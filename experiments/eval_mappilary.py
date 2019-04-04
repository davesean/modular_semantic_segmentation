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
import json
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
def main(modelname, net_config, gan_config, disc_config, datasetSem, datasetGAN, datasetDisc, starting_weights, flag_measure, output_mat, flag_entropy, thresholds, start, _run):
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
    # mapping from Mapillary classes to Deeplab classes
    if start==0:
        with open('/cluster/work/riner/users/haldavid/config.json') as f:
            config = json.load(f)
    else:
        with open('/Volumes/Netti HD /Master Thesis/mapillary/config.json') as f:
            config = json.load(f)

    def map_class(name):
        """Map a Mapillary class onto one of the 19 cityscapes classes or void."""
        direct_mapping = {
            'construction--barrier--fence': 'fence',
            'construction--barrier--wall': 'wall',
            'construction--flat--road': 'road',
            'construction--flat--sidewalk': 'sidewalk',
            'construction--structure--building': 'building',
            'human--person': 'person',
            'nature--sky': 'sky',
            'nature--terrain': 'terrain',
            'nature--vegetation': 'vegetation',
            'object--support--pole': 'pole',
            'object--support--utility-pole': 'pole',
            'object--traffic-light': 'traffic light',
            'object--traffic-sign--front': 'traffic sign',
            'object--vehicle--bicycle': 'bicycle',
            'object--vehicle--bus': 'bus',
            'object--vehicle--car': 'car',
            'object--vehicle--motorcycle': 'motorcycle',
            'object--vehicle--on-rails': 'train',
            'object--vehicle--truck': 'truck',
        }
        if name in direct_mapping:
            return direct_mapping[name]
        elif name.startswith('human--rider'):
            return 'rider'
        elif name.startswith('marking'):
            return 'road'
        else:
            return 'void'

    original_labels_mapi = {
        i: {'name': v['name'], 'color': v['color'], 'mapping': map_class(v['name'])}
        for i, v in enumerate(config['labels'])}

    # array to look up the label_id of a given color
    color_map = np.ndarray(shape=(256**3), dtype='int32')
    color_map[:] = -1
    for c, v in original_labels_mapi.items():
        rgb = v['color']
        rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
        color_map[rgb] = c

    # apply same label mapping as for original cityscapes
    labelinfo_mapi = {
        -1: {'name': 'void', 'color': [0, 0, 0]},
        0: {'name': 'road', 'color': [128, 64, 128]},
        1: {'name': 'sidewalk', 'color': [244, 35, 232]},
        2: {'name': 'building', 'color': [70, 70, 70]},
        3: {'name': 'wall', 'color': [70, 70, 70]},
        4: {'name': 'fence', 'color': [190, 153, 153]},
        5: {'name': 'pole', 'color': [153, 153, 153]},
        6: {'name': 'traffic light', 'color': [0, 0, 0]},
        7: {'name': 'traffic sign', 'color': [220, 220, 0]},
        8: {'name': 'vegetation', 'color': [107, 142, 35]},
        9: {'name': 'terrain', 'color': [107, 142, 35]},
        10: {'name': 'sky', 'color': [70, 130, 180]},
        11: {'name': 'person', 'color': [220, 20, 60]},
        12: {'name': 'rider', 'color': [220, 20, 60]},
        13: {'name': 'car', 'color': [0,  0, 142]},
        14: {'name': 'truck', 'color': [0,  0, 142]},
        15: {'name': 'bus', 'color': [0,  0, 142]},
        16: {'name': 'train', 'color': [0,  0, 142]},
        17: {'name': 'motorcycle', 'color': [0,  0, 142]},
        18: {'name': 'bicycle', 'color': [119, 11, 32]}
    }

    label_lookup_mapi = [next(i for i in labelinfo_mapi if labelinfo_mapi[i]['name'] == v['mapping'])
                         for v in original_labels_mapi.values()]

    lookup_mapi = np.array([labelinfo_mapi[i]['color'] for i in range(max(labelinfo_mapi.keys()) + 1)]).astype(int)


    ###########################################################################
    # mapping from Deeplab classes to Adapnet classes
    original_labelinfo_dl = {
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

    label_lookup_dl = [next(i for i in labelinfo
                         if labelinfo[i]['name'] == k['mapping'])
                         for _, k in original_labelinfo_dl.items()]

    lookup = np.array([labelinfo[i]['color'] for i in range(max(labelinfo.keys()) + 1)]).astype(int)

    # base_path = path.join(DATA_BASEPATH, 'fishyscapes_newfog')
    # if 'TMPDIR' in environ:
    #     print('INFO loading dataset into machine ... ')
    #     # first load the zipfile into a closer memory location, then load all the
    #     # images
    #     zip = zipfile.ZipFile(path.join(base_path, 'testset.zip'), 'r')
    #     localtmp = environ['TMPDIR']
    #     zip.extractall(localtmp)
    #     zip.close()
    #     base_path = localtmp
    #
    # print('DONE loading dataset into machine ... ')

    ###########################################################################

    sub_size = 100

    if start==0:
        input_path = "/cluster/work/riner/users/blumh/mapillary_evaluation_set"
        out_path = path.join("/cluster/work/riner/users/haldavid/MapillaryResultsCropped",disc_config['arch'])
    else:
        input_path = "/Users/David/Downloads/mapillary_evaluation_set"
        out_path = path.join("/Users/David/Desktop/out",disc_config['arch'])

    if not os.path.exists(out_path):
        os.makedirs(out_path)
                                #N H W C
    img_array = np.zeros((sub_size,256,256,3))
    segm_array = np.zeros((sub_size,256,256,3))
    gt_array = np.zeros((sub_size,256,256,3))
    mask_array = np.zeros((sub_size,256,256))
    out_mask_array = np.zeros((sub_size,256,256))

    for i in range(sub_size):
        #RGB
        img = cv2.imread(path.join(input_path, str(i)+'_rgb.png'))
        #
        dl_labels =  np.expand_dims(cv2.imread(path.join(input_path,str(i)+'_predict.png'))[:,:,0],axis=0)
        cs_labels = np.asarray(label_lookup_dl, dtype='int32')[dl_labels]
        segm = np.array(lookup[cs_labels[:]]).astype('uint8')[...,::-1]

        gt_labels = cv2.imread(path.join(input_path,str(i)+'_segm.png'))[...,::-1]
        gt_labels = gt_labels.dot(np.array([65536, 256, 1], dtype='int32'))
        gt_segm = color_map[gt_labels]
        # apply mapping

        gt_segm = np.asarray(label_lookup_mapi, dtype='int32')[gt_segm]

        gt_segm = np.array(lookup_mapi[gt_segm[:]]).astype('uint8')[...,::-1]

        mask = cv2.imread(path.join(input_path, str(i)+'_mask.png'))[...,0]

        #Crop all in the same manor 4:3 we crop to 4:2
        crop_margin = int(img.shape[0]/6)
        img = img[crop_margin:-crop_margin,...]
        segm = segm[0,crop_margin:-crop_margin,...]
        gt_segm = gt_segm[crop_margin:-crop_margin,...]
        mask = mask[crop_margin:-crop_margin,...]

        img_array[i,...] = cv2.resize(img, (256, 256),interpolation=cv2.INTER_LINEAR)
        segm_array[i,...] = cv2.resize(segm, (256, 256),interpolation=cv2.INTER_NEAREST)
        gt_array[i,...] = cv2.resize(gt_segm, (256, 256),interpolation=cv2.INTER_NEAREST)
        mask_array[i,...] = cv2.resize(mask, (256, 256),interpolation=cv2.INTER_NEAREST)

    with GAN_sess.as_default():
        with GAN_graph.as_default():
            synth_images, pred_descriminator = modelGAN.transform_withD(a,segm_array, img_array)

    with sessD.as_default():
        with Disc_graph.as_default():
            simMat = modelDiff.transform(img_array, synth_images, segm_array)

    for i in range(sub_size):
        gt_mask = error_mask(segm_array[i],gt_array[i])
        out_mask = np.logical_or(gt_mask,mask_array[i])
        out_mask_array[i,...] = out_mask

        # filename = path.join(out_path,str(i)+'_dissim.npy')
        # np.save(filename,cv2.resize(simMat[i,...], (2048, 1024),interpolation=cv2.INTER_LINEAR))
        # if disc_config['arch']=='arch13':
        #     filename = path.join(out_path,str(i)+'_discsim.npy')
        #     np.save(filename,cv2.resize(pred_descriminator[i,...], (2048, 1024),interpolation=cv2.INTER_NEAREST))
        #     filename = path.join(out_path,str(i)+'_mask.npy')
        #     np.save(filename,cv2.resize(out_mask, (2048, 1024),interpolation=cv2.INTER_NEAREST))
        #     filename = path.join(out_path,str(i)+'_rgb.npy')
        #     np.save(filename,cv2.resize(img_array[i,...], (2048, 1024),interpolation=cv2.INTER_LINEAR))

    filename = path.join(out_path,str(i)+'_dissim.npy')
    np.save(filename,simMat)
    if disc_config['arch']=='arch13':
        filename = path.join(out_path,str(i)+'_discsim.npy')
        np.save(filename,pred_descriminator)
        filename = path.join(out_path,str(i)+'_mask.npy')
        np.save(filename,out_mask_array)
        filename = path.join(out_path,str(i)+'_rgb.npy')
        np.save(filename,img_array)
        filename = path.join(out_path,str(i)+'_synth.npy')
        np.save(filename,synth_images)
        filename = path.join(out_path,str(i)+'_segm.npy')
        np.save(filename,segm_array)
        filename = path.join(out_path,str(i)+'_gt.npy')
        np.save(filename,gt_array)



if __name__ == '__main__':
    ex.run_commandline()
    # for some reason we have processes running in the background that won't stop
    # this is the only way to kill them
    os._exit(os.EX_OK)
