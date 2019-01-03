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
from tests.evaluationFunctions import computePRvalues, computeIOU
import sys
import shutil

class Helper:
    name = 'A'

a = Helper()
b = Helper()

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

def predict_network(net, output_dir, paths, data_desc):
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
    for i in range(paths['rgb'].shape[0]):
        img = np.expand_dims(paths['rgb'][i,:,:,:], axis=0)
        data = {'rgb': img, 'depth': tf.zeros(shape=[img.shape[0],img.shape[1],img.shape[2],1],dtype=tf.float32),
                            'labels': tf.zeros(shape=img.shape[0:-1],dtype=tf.int32),
                            'mask': tf.zeros(shape=img.shape[0:-1],dtype=tf.float32)}
        output = net.predict(data)
        outputColor = data_desc.coloured_labels(labels=output)
        outputColor = outputColor[0,:,:,:]
        segm[i,:,:,:] = outputColor[...,::-1]

        outputColor = data_desc.coloured_labels(labels=paths['labels'][i,:,:])
        # outputColor = outputColor[:,:,:]
        segm_gt[i,:,:,:] = outputColor[...,::-1]
    return segm, paths['rgb'], paths['mask'], segm_gt

ex = sc.Experiment()
# reduce output of progress bars
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(get_observer())

@ex.capture
def predict_output(net, output_dir, paths, data_desc, _run):
    """Predict data on a given network"""
    return predict_network(net, output_dir, paths, data_desc)

@ex.main
def main(modelname, net_config, gan_config, disc_config, datasetSem, datasetGAN, datasetDisc, starting_weights, input_folder, _run):
    for key in gan_config:
        setattr(a, key, gan_config[key])
    for key in disc_config:
        setattr(b, key, disc_config[key])
    setattr(a,'EXP_OUT',EXP_OUT)
    setattr(a,'RUN_id',_run._id)
    setattr(b,'EXP_OUT',EXP_OUT)
    setattr(b,'RUN_id',_run._id)
    data_id=datasetDisc['image_input_dir'].split('/')[-1].split('_')[0]
    setattr(b,'DATA_id',data_id)    # Set up the directories for diagnostics
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
        # load the dataset class
        dataGAN = get_dataset(datasetGAN['name'])
        # data = data(**datasetGAN)
        cGAN_model = get_model('cGAN')
        modelGAN = cGAN_model(GAN_sess, checkpoint_dir=output_dir,
                        data_desc=dataGAN.get_data_description(),
                        checkpoint=os.path.join(a.EXP_OUT,str(a.checkpoint)))
        print("INFO: GAN Imported weights succesfully")

    Disc_graph = tf.Graph()
    with Disc_graph.as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        sessD = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        dataD = get_dataset(datasetDisc['name'])
        dataD = dataD(datasetDisc['image_input_dir'],**datasetDisc)
        disc_model = get_model('simDisc')
        modelDiff=disc_model(sess=sessD, checkpoint_dir=output_dir, data=dataD)
        print("INFO: Begin training simDisc")
        tmp = modelDiff.train(b)
        _run.info['simDisc_predictions'] = tmp
        _run.info['simDisc_mean_predictions'] = np.mean(tmp, axis=0)
        print("INFO: Finished training simDisc")


    benchmarks = ['valid','wilddash','pos_neg_set','measure']
    for set in benchmarks:
        # if set == "wilddash":
        #     #TODO Get wilddash images and load them into 4D array
        #     #Put into dic: labels,rgb and mask -> after SemSeg, if semSeg is off -> mask 1, else 0
        #     #TODO Don't forget to transform labels like cityscapes -> coloured_labels func
        if set == "measure":
            data = data_desc(**datasetSem)
            dataset = data.get_measureset(tf_dataset=False)
        elif set == "valid":
            data = data_desc(**datasetSem)
            dataset = data.get_validation_set(tf_dataset=False)
        # else:
        #     #TODO Get nine/nine images and load them into 4D array
        #     #Put into dic: labels empty,rgb images and mask -> for 9 neg, 1 mask, for 9 pos 0 mask



        sem_seg_images, rgb_images, masks, _ = predict_output(net,output_dir,dataset,data)

        with GAN_sess.as_default():
            with GAN_graph.as_default():
                synth_images = modelGAN.transform(a,sem_seg_images)


        with sessD.as_default():
            with Disc_graph.as_default():
                simMat = modelDiff.transform(rgb_images, synth_images, sem_seg_images)

        _run.info[set+'_IOU'] = computeIOU(simMat, masks)
        _run.info[set+'_PRvals'] = computePRvalues(simMat, masks)



if __name__ == '__main__':
    ex.run_commandline()
    # for some reason we have processes running in the background that won't stop
    # this is the only way to kill them
    os._exit(os.EX_OK)
