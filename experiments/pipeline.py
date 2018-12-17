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
import sys
sys.path.insert(0, '../pix2pix-tensorflow')
from model import pix2pix
sys.path.insert(0, '../Discrim')
from diffDiscrim import DiffDiscrim
import shutil

class Helper:
    name = 'A'

a = Helper()

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

def predict_network(net, output_dir, paths, data_desc, dataFlag):
    """
    Predict on a given dataset.

    Args:
        net: An instance of a `base_model` class.
        output_dir: A directory path. This function will add all files found at this path
            as artifacts to the experiment.
        paths: A list of paths to images to be evaluated
    """
    if dataFlag:
        data_class = data_desc(in_memory=False)
        seg = []
        inp = []
        img_w=256
        img_h=256
        for path in paths:
            try:
                image = np.float32(np.expand_dims(cv2.resize(cv2.imread(path),(img_w,img_h),interpolation=cv2.INTER_LINEAR), axis=0))
                inp.append(image)
                data = {'rgb': image, 'depth': tf.zeros(shape=[1, img_w, img_h, 1],dtype=tf.float32), 'labels': tf.zeros(shape=[1, img_w, img_h],dtype=tf.int32)}
                output = net.predict(data)
                outputColor = data_class.coloured_labels(labels=output)
                seg.append(outputColor)

            except KeyboardInterrupt:
                print('WARNING: Got Keyboard Interrupt')
                break
        # return np.concatenate(ret)
        return seg, inp
    else:
        segm = np.zeros((paths['rgb'].shape))
        for i in range(paths['rgb'].shape[0]):
            img = np.expand_dims(paths['rgb'][i,:,:,:], axis=0)
            data = {'rgb': img, 'depth': tf.zeros(shape=[img.shape[0],img.shape[1],img.shape[2],1],dtype=tf.float32), 'labels': tf.zeros(shape=img.shape[0:-1],dtype=tf.int32)}
            output = net.predict(data)
            outputColor = data_desc.coloured_labels(labels=output)
            outputColor = outputColor[0,:,:,:]
            segm[i,:,:,:] = outputColor[...,::-1]
        return segm, paths['rgb'], paths['labels']

    # To end the experiment, we collect all produced output files and store them.
    # for filename in os.listdir(output_dir):
    #     experiment.add_artifact(os.path.join(output_dir, filename))

def load_list_path(input_path):
    tmp = glob.glob(os.path.join(input_path,"*.png"))
    tmp.sort()
    return tmp

ex = sc.Experiment()
# reduce output of progress bars
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(get_observer())

@ex.capture
def predict_output(net, output_dir, paths, data_desc, dataFlag, _run):
    """Predict data on a given network"""
    return predict_network(net, output_dir, paths, data_desc, dataFlag)


@ex.main
def main(modelname, net_config, gan_config, disc_config, datasetSem, datasetGAN, datasetDisc, starting_weights, input_folder, sets, _run):
    for key in gan_config:
        setattr(a, key, gan_config[key])

    setattr(a,'EXP_OUT',EXP_OUT)
    setattr(a,'RUN_id',_run._id)

    # Set up the directories for diagnostics
    output_dir = create_directories(_run._id, ex)

    # load the data for the data description
    data_desc = get_dataset(datasetSem['name'])
    dataFlag = (input_folder is not None)

    if dataFlag:
        # load images in input_folder
        eval_image_paths = load_list_path(input_folder)
    elif sets == "measure":
        data = data_desc(**datasetSem)
        dataset = data.get_measureset(tf_dataset=False)
    else:
        data = data_desc(**datasetSem)
        dataset = data.get_validation_set(tf_dataset=False)

    # create the network
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # load the dataset class
    dataGAN = get_dataset(datasetGAN['name'])
    # data = data(**datasetGAN)
    modelGAN = pix2pix(sess, checkpoint_dir=output_dir,
                    data_desc=dataGAN.get_data_description(),
                    checkpoint=os.path.join(a.EXP_OUT,str(a.checkpoint)))
    print("INFO: GAN Imported weights succesfully")

    model = get_model(modelname)
    net = model(data_description=data_desc.get_data_description(),
                output_dir=output_dir, **net_config)
    net.import_weights(filepath=starting_weights)
    print("INFO: Imported weights succesfully")

    ss_run_id = starting_weights.split('/')[-2]
    gan_run_id = str(a.checkpoint)
    folder_name = ss_run_id + "_" + gan_run_id + "_" + str(disc_config['checkpoint'])
    base_output_path = os.path.join(a.file_output_dir,folder_name)
    if not os.path.exists(base_output_path):
        os.makedirs(base_output_path)

    if dataFlag:
        sem_seg_images, rgb_images = predict_output(net,output_dir,eval_image_paths,data_desc,dataFlag=dataFlag)
    else:
        sem_seg_images, rgb_images, masks = predict_output(net,output_dir,dataset,data,dataFlag=dataFlag)
        matrix_path = os.path.join(base_output_path,"mskMat.npy")
        np.save(matrix_path, masks)
    print("Done with prediction of semantic segmentation")

    synth_images = modelGAN.transform(a,sem_seg_images)
    print("Done with prediction of GAN")

    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sessD = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    dataD = get_dataset(datasetDisc['name'])
    # data = data(dataset['image_input_dir'], ppd=dataset['ppd'])
    dataD = dataD(datasetDisc['image_input_dir'],**datasetDisc)

    modelDiff=DiffDiscrim(sess=sessD, checkpoint_dir=output_dir, data=dataD,
                      checkpoint=os.path.join(a.EXP_OUT,str(disc_config['checkpoint'])))
    print("INFO: Disc Imported weights succesfully")

    simMat = modelDiff.transform(rgb_images, synth_images, sem_seg_images)

    matrix_path = os.path.join(base_output_path,"simMat.npy")
    np.save(matrix_path, simMat)

    matrix_path = os.path.join(base_output_path,"rgbMat.npy")
    np.save(matrix_path, rgb_images)

    matrix_path = os.path.join(base_output_path,"synMat.npy")
    np.save(matrix_path, synth_images)

    matrix_path = os.path.join(base_output_path,"semMat.npy")
    np.save(matrix_path, sem_seg_images)

if __name__ == '__main__':
    ex.run_commandline()
    # for some reason we have processes running in the background that won't stop
    # this is the only way to kill them
    os._exit(os.EX_OK)
