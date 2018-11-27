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

def predict_network(net, output_dir, paths, data_desc):
    """
    Predict on a given dataset.

    Args:
        net: An instance of a `base_model` class.
        output_dir: A directory path. This function will add all files found at this path
            as artifacts to the experiment.
        paths: A list of paths to images to be evaluated
    """
    seg = []
    inp = []
    img_w=256
    img_h=256
    data_class = data_desc()
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

    # To end the experiment, we collect all produced output files and store them.
    # for filename in os.listdir(output_dir):
    #     experiment.add_artifact(os.path.join(output_dir, filename))

def load_list_path(input_path):
    return glob.glob(os.path.join(input_path,"*.png"))

ex = sc.Experiment()
# reduce output of progress bars
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(get_observer())

@ex.capture
def predict_output(net, output_dir, paths, data_desc, _run):
    """Predict data on a given network"""
    return predict_network(net, output_dir, paths, data_desc)


@ex.main
def main(modelname, net_config, gan_config, dataset, datasetGAN, starting_weights, input_folder, _run):
    for key in gan_config:
        setattr(a, key, gan_config[key])

    setattr(a,'EXP_OUT',EXP_OUT)
    setattr(a,'RUN_id',_run._id)

    # Set up the directories for diagnostics
    output_dir = create_directories(_run._id, ex)

    # load the data for the data description
    data_desc = get_dataset(dataset['name'])

    # load images in input_folder
    eval_image_paths = load_list_path(input_folder)

    # create the network
    model = get_model(modelname)
    with model(data_description=data_desc.get_data_description(),
               output_dir=output_dir, **net_config) as net:
        net.import_weights(filepath=starting_weights)
        print("INFO: Imported weights succesfully")
        sem_seg_images, rgb_images = predict_output(net,output_dir,eval_image_paths,data_desc)


    print("Done with prediction of semantic segmentation")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # load the dataset class
        data = get_dataset(datasetGAN['name'])
        data = data(**dataset)
        model = pix2pix(sess, image_size=a.input_image_size, batch_size=a.batch_size,
                        output_size=a.input_image_size, dataset_name=dataset['name'],
                        checkpoint_dir=output_dir, data=data, momentum=a.batch_momentum,
                        L1_lambda=int(a.l1_weight/a.gan_weight), gf_dim=a.ngf,
                        df_dim=a.ndf,label_smoothing=a.label_smoothing,
                        noise_std_dev=a.noise_std_dev)
        model.predict(a,sem_seg_images, rgb_images)
        print("Done with prediction of GAN")


if __name__ == '__main__':
    ex.run_commandline()
    # for some reason we have processes running in the background that won't stop
    # this is the only way to kill them
    os._exit(os.EX_OK)
