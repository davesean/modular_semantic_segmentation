from os import path, environ
import os
import glob
import sacred as sc
import cv2
import scipy.misc
import numpy as np
import tensorflow as tf
import zipfile
from sacred.utils import apply_backspaces_and_linefeeds
from experiments.utils import get_observer, load_data
from xview.datasets import Cityscapes
from experiments.evaluation import evaluate, import_weights_into_network
from xview.datasets import get_dataset
from xview.models import get_model
from xview.settings import EXP_OUT, DATA_BASEPATH
from tests.evaluationFunctions import computePRvalues, computeIOU, computePatchSSIM, ShannonEntropy
import sys
import shutil
from sys import stdout


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

ex = sc.Experiment()
# reduce output of progress bars
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(get_observer())

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
    # net.import_weights(filepath=starting_weights)
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
                             batch_size=disc_config['batch_size'],feature_extractor=os.path.join(a.EXP_OUT,str(a.checkpoint)))

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

    _run.info['thresholds'] = thresholds

    ###########################################################################
    # mapping from Deeplab classes to Adapnet classes
    original_labelinfo = {
            0: {'name': 'road', 'mapping': 'road'},
            1: {'name': 'sidewalk', 'mapping': 'sidewalk'},
            2: {'name': 'building', 'mapping': 'building'},
            3: {'name': 'wall', 'mapping': 'building'},
            4: {'name': 'fence', 'mapping': 'fence'},
            5: {'name': 'pole', 'mapping': 'pole'},
            6: {'name': 'traffic light', 'mapping': 'void'},
            7: {'name': 'traffic sign', 'mapping': 'traffic sign'},
            8: {'name': 'vegetation', 'mapping': 'vegetation'},
            9: {'name': 'terrain', 'mapping': 'vegetation'},
            10: {'name': 'sky', 'mapping': 'sky'},
            11: {'name': 'person', 'mapping': 'person'},
            12: {'name': 'rider', 'mapping': 'person'},
            13: {'name': 'car', 'mapping': 'vehicle'},
            14: {'name': 'truck', 'mapping': 'vehicle'},
            15: {'name': 'bus', 'mapping': 'vehicle'},
            16: {'name': 'train', 'mapping': 'vehicle'},
            17: {'name': 'motorcycle', 'mapping': 'vehicle'},
            18: {'name': 'bicycle', 'mapping': 'bicycle'},
            255: {'name': 'void', 'mapping': 'void'}
    }

    labelinfo = {
        0: {'name': 'void', 'color': [0, 0, 0]},
        1: {'name': 'sky', 'color': [70, 130, 180]},
        2: {'name': 'building', 'color': [70, 70, 70]},
        3: {'name': 'road', 'color': [128, 64, 128]},
        4: {'name': 'sidewalk', 'color': [244, 35, 232]},
        5: {'name': 'fence', 'color': [190, 153, 153]},
        6: {'name': 'vegetation', 'color': [107, 142, 35]},
        7: {'name': 'pole', 'color': [153, 153, 153]},
        8: {'name': 'vehicle', 'color': [0,  0, 142]},
        9: {'name': 'traffic sign', 'color': [220, 220, 0]},
        10: {'name': 'person', 'color': [220, 20, 60]},
        11: {'name': 'bicycle', 'color': [119, 11, 32]}
    }

    label_lookup = [next(i for i in labelinfo
                         if labelinfo[i]['name'] == k['mapping'])
                         for _, k in original_labelinfo.items()]

    base_path = path.join(DATA_BASEPATH, 'fishyscapes_newfog')
    if 'TMPDIR' in environ:
        print('INFO loading dataset into machine ... ')
        # first load the zipfile into a closer memory location, then load all the
        # images
        zip = zipfile.ZipFile(path.join(base_path, 'testset.zip'), 'r')
        localtmp = environ['TMPDIR']
        zip.extractall(localtmp)
        zip.close()
        base_path = localtmp

    print('DONE loading dataset into machine ... ')

    ###########################################################################

    set_size = 1000
    h_orig = 1024
    w_orig = 2048

    sub_size = 1

    semseg_path = "/cluster/work/riner/users/blumh/fishyscapes_deeplab_predictions_newfog"
    out_path = "/cluster/work/riner/users/blumh/resultsDH"


    # for k in range(int(set_size/sub_size)):
    for k in range(0):
        # counter = 0
        kb = k*sub_size
        if k>0:
            print('Done %d images' %(kb))
            stdout.flush()
        img_array = np.zeros((sub_size,256,256,3))
        for i in range(sub_size):
            img = cv2.imread(path.join(base_path,'testset', str(i+kb)+'_rgb.png'))
            dl_labels =  np.expand_dims(cv2.imread(path.join(semseg_path,str(i+kb)+'_predict.png'))[:,:,0],axis=0)
            cs_labels = np.asarray(label_lookup, dtype='int32')[dl_labels]

            lookup = np.array([labelinfo[i]['color'] for i in range(max(labelinfo.keys()) + 1)]).astype(int)
            segm = np.array(lookup[cs_labels[:]]).astype('uint8')[...,::-1]


            #mask = cv2.imread(path.join(base_path, str(i)+'_mask.png'), cv2.IMREAD_ANYDEPTH)
            # blob['labels'] = cv2.imread(labels_filename, cv2.IMREAD_ANYDEPTH)
            # # apply label mapping
            # blob['labels'] = np.asarray(self.label_lookup, dtype='int32')[blob['labels']]

            img_array[i,...] = cv2.resize(img, (256, 256),interpolation=cv2.INTER_LINEAR)
            segm = np.expand_dims(cv2.resize(segm[0,...], (256, 256),interpolation=cv2.INTER_NEAREST),axis=0)

            filename = path.join(out_path,str(i+kb)+'_segm.png')
            cv2.imwrite(filename,segm)


        # data = {'rgb': tf.to_float(img_array), 'depth': tf.zeros(shape=[img_array.shape[0],img_array.shape[1],img_array.shape[2],1],dtype=tf.float32),
        #                     'labels': tf.zeros(shape=img_array.shape[0:-1],dtype=tf.int32),
        #                     'mask': tf.zeros(shape=img_array.shape[0:-1],dtype=tf.float32)}
        # output = net.predict(data)

        # outputColor = data_SemSeg.coloured_labels(labels=output)
        # segm = outputColor[...,::-1]

        # with GAN_sess.as_default():
        #     with GAN_graph.as_default():
        #         synth_images = modelGAN.transform(a,segm)
        #
        # with sessD.as_default():
        #     with Disc_graph.as_default():
        #         simMat = modelDiff.transform(img_array, synth_images, segm)
        #
        # filename = os.path.join(output_dir,"rgb"+str(k)+".png")
        # cv2.imwrite(filename, img_array[0,...,::-1])
        # filename = os.path.join(output_dir,"segm"+str(k)+".png")
        # cv2.imwrite(filename, segm[0,...])
        # filename = os.path.join(output_dir,"synth"+str(k)+".png")
        # cv2.imwrite(filename, synth_images[0,...])
        # filename = os.path.join(output_dir,"sim"+str(k)+".png")
        # cv2.imwrite(filename, simMat[0,...]*255)






if __name__ == '__main__':
    ex.run_commandline()
    # for some reason we have processes running in the background that won't stop
    # this is the only way to kill them
    os._exit(os.EX_OK)
