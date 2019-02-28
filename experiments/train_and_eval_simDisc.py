import os
import glob
import sacred as sc
import cv2
import scipy.misc
import numpy as np
import tensorflow as tf
from sacred.utils import apply_backspaces_and_linefeeds
from experiments.utils import get_observer, load_data
from xview.datasets import Cityscapes
from experiments.evaluation import evaluate, import_weights_into_network
from xview.datasets import get_dataset
from xview.models import get_model
from xview.settings import EXP_OUT
from tests.evaluationFunctions import computePRvalues, computeIOU, computePatchSSIM, ShannonEntropy
import sys
import shutil

class Helper:
    name = 'A'

a = Helper()
b = Helper()

def error_mask(segm_image, gt_image):
    mask_3d = (segm_image == gt_image)
    mask = np.logical_and(mask_3d[:,:,2],np.logical_and(mask_3d[:,:,0],mask_3d[:,:,1]))
    return ~mask

def create_directories(run_id, experiment):
    """
    Make sure directories for storing diagnostics are created and clean.

    Args:
        run_id: ID of the current sacred run, you can get it from _run._id in a captured
            function.
        experiment: The sacred experiment object
    Returns:
        The path to the created output directory you can store your diagnostics to.
    """
    root = EXP_OUT
    # create temporary directory for output files
    if not os.path.exists(root):
        os.makedirs(root)
    # The id of this experiment is stored in the magical _run object we get from the
    # decorator.
    output_dir = '{}/{}'.format(root, run_id)
    if os.path.exists(output_dir):
        # Directory may already exist if run_id is None (in case of an unobserved
        # test-run)
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    # Tell the experiment that this output dir is also used for tensorflow summaries
    experiment.info.setdefault("tensorflow", {}).setdefault("logdirs", [])\
        .append(output_dir)
    return output_dir

def predict_network(net, output_dir, paths, data_desc, flag_entropy, num_classes):
    """
    Predict on a given dataset.

    Args:
        net: An instance of a `base_model` class.
        output_dir: A directory path. This function will add all files found at this path
            as artifacts to the experiment.
        paths: A list of paths to images to be evaluated
    """

    segm = np.zeros((paths['rgb'].shape))
    segm_gt = np.zeros((paths['rgb'].shape))
    mask = np.zeros((paths['rgb'].shape[0],paths['rgb'].shape[1],paths['rgb'].shape[2]))
    outLabel = np.zeros((paths['rgb'].shape[0],paths['rgb'].shape[1],paths['rgb'].shape[2]))
    probs = np.zeros((paths['rgb'].shape[0],paths['rgb'].shape[1],paths['rgb'].shape[2],num_classes))


    for i in range(paths['rgb'].shape[0]):
        img = np.expand_dims(paths['rgb'][i,:,:,:], axis=0)
        data = {'rgb': img, 'depth': tf.zeros(shape=[img.shape[0],img.shape[1],img.shape[2],1],dtype=tf.float32),
                            'labels': tf.zeros(shape=img.shape[0:-1],dtype=tf.int32),
                            'mask': tf.zeros(shape=img.shape[0:-1],dtype=tf.float32)}
        output = net.predict(data)
        outLabel[i,:,:] = output[0,:,:]
        outputColor = data_desc.coloured_labels(labels=output)
        outputColor = outputColor[0,:,:,:]
        segm[i,:,:,:] = outputColor[...,::-1]
        outputColor = data_desc.coloured_labels(labels=paths['labels'][i,:,:])
        segm_gt[i,:,:,:] = outputColor[...,::-1]
        mask[i,:,:] = error_mask(segm[i,:,:,:], segm_gt[i,:,:,:])

        if flag_entropy:
            out_prob = net.predict(data, output_attr='prob')
            probs[i,:,:,:] = out_prob[0,:,:,:]


    if 'mask' in paths:
        return segm, paths['rgb'], paths['mask'], segm_gt, outLabel, probs
    else:
        return segm, paths['rgb'], mask, segm_gt, outLabel, probs

ex = sc.Experiment()
# reduce output of progress bars
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(get_observer())

@ex.capture
def predict_output(net, output_dir, paths, data_desc, flag_entropy, num_classes, _run):
    """Predict data on a given network"""
    return predict_network(net, output_dir, paths, data_desc, flag_entropy, num_classes)

@ex.main
def main(modelname, net_config, gan_config, disc_config, datasetSem, datasetGAN, datasetDisc, starting_weights, flag_measure, output_mat, flag_entropy, thresholds, _run):
    for key in gan_config:
        setattr(a, key, gan_config[key])
    for key in disc_config:
        setattr(b, key, disc_config[key])
    setattr(a,'EXP_OUT',EXP_OUT)
    setattr(a,'RUN_id',_run._id)
    setattr(b,'EXP_OUT',EXP_OUT)
    setattr(b,'RUN_id',_run._id)
    disc_data_path = os.path.join(datasetDisc['image_input_dir'],str(gan_config['checkpoint'])+"_full")
    data_id=str(gan_config['checkpoint'])
    setattr(b,'DATA_id',data_id)
    # Set up the directories for diagnostics
    output_dir = create_directories(_run._id, ex)

    # load the data for the data description
    data_desc = get_dataset(datasetSem['name'])

    model = get_model(modelname)
    net = model(data_description=data_desc.get_data_description(),
                output_dir=output_dir, **net_config)
    net.import_weights(filepath=starting_weights)
    print("INFO: SemSegNet Imported weights succesfully")

    GAN_graph = tf.Graph()
    with GAN_graph.as_default():
        # create the network
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        GAN_sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        if gan_config['type'] == 'cascRef':
            dataGAN = get_dataset('cityscapes_cascGAN')
            cGAN_model = get_model('cascGAN')
            if a.checkpoint is not None:
                ckp = os.path.join(a.EXP_OUT,str(a.checkpoint))
            modelGAN = cGAN_model(GAN_sess,dataset_name='cityscapes_cascGAN',image_size=disc_config['input_image_size'],
                               checkpoint_dir=output_dir,
                               data_desc=dataGAN.get_data_description(),
                               is_training=False, checkpoint=ckp, vgg_checkpoint="/cluster/work/riner/users/haldavid/Checkpoints/VGG_Model/imagenet-vgg-verydeep-19.mat")
        else:
            # load the dataset class
            dataGAN = get_dataset(datasetGAN['name'])
            # data = data(**datasetGAN)
            cGAN_model = get_model('cGAN')
            modelGAN = cGAN_model(GAN_sess, checkpoint_dir=output_dir,
                            data_desc=dataGAN.get_data_description(),
                            feature_matching=gan_config['feature_matching'],
                            checkpoint=os.path.join(a.EXP_OUT,str(a.checkpoint)),
                            gen_type=gan_config['type'],use_grayscale=gan_config['use_grayscale'])
        print("INFO: Generative model imported weights succesfully")

    Disc_graph = tf.Graph()
    with Disc_graph.as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        sessD = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        dataD = get_dataset(datasetDisc['name'])
        dataD = dataD(disc_data_path,**datasetDisc)
        disc_model = get_model('simDisc')

        disc_checkpoint = None
        if disc_config['checkpoint'] is not None:
            disc_checkpoint = os.path.join(a.EXP_OUT,str(disc_config['checkpoint']))
        modelDiff=disc_model(sess=sessD, checkpoint_dir=output_dir, pos_weight=disc_config['pos_weight'],
                             data=dataD, arch=disc_config['arch'], use_grayscale=disc_config['use_grayscale'],
                             checkpoint=disc_checkpoint, use_segm=disc_config['use_segm'],
                             batch_size=disc_config['batch_size'])

        if disc_config['checkpoint'] is None:
            print("INFO: Begin training simDisc")
            tmp = modelDiff.train(b)
            _run.info['simDisc_predictions'] = tmp
            _run.info['simDisc_mean_predictions'] = np.mean(tmp, axis=0)
            _run.info['simDisc_stdDev'] = np.std(tmp, axis=0)
            print("INFO: Finished training simDisc")
        else:
            print("INFO: Init and loaded checpoint for simDisc")

    if flag_measure:
        benchmarks = ['measure']
    else:
        benchmarks = ['wilddash','posneg','valid','measure']
    data_SemSeg = data_desc(**datasetSem)

    # thresholds = [0.2,0.4,0.6,0.8]
    # thresholds = [0.85,0.9,0.95,0.99]
    _run.info['thresholds'] = thresholds
    for set in benchmarks:
        if set == "measure":
            dataset = data_SemSeg.get_measureset(tf_dataset=False)
        elif set == "valid":
            dataset = data_SemSeg.get_validation_set(tf_dataset=False)
        else:
            data = get_dataset(set)
            head, _ = os.path.split(datasetDisc['image_input_dir'])
            data = data(os.path.join(head,set))
            dataset = data.get_validation_set(tf_dataset=False)

        sem_seg_images, rgb_images, masks, gt_seg_images, seg_seg_labels, output_probs = predict_output(net,output_dir,dataset,data_SemSeg,flag_entropy,net_config['num_classes'])

        with GAN_sess.as_default():
            with GAN_graph.as_default():
                if gan_config['type'] == 'cascRef':
                    synth_images = modelGAN.transform(a,seg_seg_labels)
                else:
                    synth_images = modelGAN.transform(a,sem_seg_images)

        with sessD.as_default():
            with Disc_graph.as_default():
                simMat = modelDiff.transform(rgb_images, synth_images, sem_seg_images)

############################################################################################
        # Perhaps need to also give number of patches per dimension.
        simMatSSIM = computePatchSSIM(rgb_images,synth_images,datasetDisc['ppd'])

############################################################################################

        temp_iou = computeIOU(simMat, masks, thresholds)
        temp_pr = computePRvalues(simMat, masks, thresholds)
        _run.info[set+'_IOU'] = temp_iou
        _run.info[set+'_PRvals'] = temp_pr
        _run.info[set+'_F1score'] = 2*np.asarray(temp_pr[1])*np.asarray(temp_pr[2])/(np.asarray(temp_pr[1])+np.asarray(temp_pr[2]))

        # _run.info[set+'_SSIM_IOU'] = computeIOU(simMatSSIM, masks, thresholds)
        # _run.info[set+'_SSIM_PRvals'] = computePRvalues(simMatSSIM, masks)

        if flag_entropy and set is not 'posneg':
            entropy = ShannonEntropy(output_probs)
            # _run.info[set+'_meanVarEntropyOoD'] = [np.mean(entropy[masks.astype(bool)]),np.var(entropy[masks.astype(bool)],ddof=1)]
            # _run.info[set+'_meanVarEntropyID'] = [np.mean(entropy[~masks.astype(bool)]),np.var(entropy[~masks.astype(bool)],ddof=1)]
            temp_iou = computeIOU(entropy, masks, thresholds)
            temp_pr = computePRvalues(entropy, masks, thresholds)
            _run.info[set+'_entropy_IOU'] = temp_iou
            _run.info[set+'_entropy_PRvals'] = temp_pr
            _run.info[set+'_entropy_F1score'] = 2*np.asarray(temp_pr[1])*np.asarray(temp_pr[2])/(np.asarray(temp_pr[1])+np.asarray(temp_pr[2]))


        k = masks.shape[0]

        if output_mat and set is not 'posneg':

            if not os.path.exists(os.path.join(output_dir,set)):
                os.makedirs(os.path.join(output_dir,set))
            if set is 'measure':
                k=25

            matrix_path = os.path.join(output_dir,set,"mskMat.npy")
            np.save(matrix_path, masks[0:k, ...])

            matrix_path = os.path.join(output_dir,set,"simMat.npy")
            np.save(matrix_path, simMat[0:k, ...])

            # matrix_path = os.path.join(output_dir,set,"ssmMat.npy")
            # np.save(matrix_path, simMatSSIM[0:k, ...])

            matrix_path = os.path.join(output_dir,set,"rgbMat.npy")
            np.save(matrix_path, rgb_images[0:k, ...])

            matrix_path = os.path.join(output_dir,set,"synMat.npy")
            np.save(matrix_path, synth_images[0:k, ...])

            matrix_path = os.path.join(output_dir,set,"semMat.npy")
            np.save(matrix_path, sem_seg_images[0:k, ...])

            matrix_path = os.path.join(output_dir,set,"gtsMat.npy")
            np.save(matrix_path, gt_seg_images[0:k, ...])

            if flag_entropy:
                matrix_path = os.path.join(output_dir,set,"entMat.npy")
                np.save(matrix_path, entropy)

if __name__ == '__main__':
    ex.run_commandline()
    # for some reason we have processes running in the background that won't stop
    # this is the only way to kill them
    os._exit(os.EX_OK)
