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
import shutil

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
    img_w=256
    img_h=256
    data_class = data_desc()
    for path in paths:
        try:
            run_id = path.split('/')[-1].split('_')[0]
            NameNumber = path.split('/')[-1].split('_')[1].split('.')[0]
            filename = run_id+"_AdapNetPredicted_"+NameNumber+".png"
            image = np.float32(np.expand_dims(cv2.resize(cv2.imread(path),(img_w,img_h),interpolation=cv2.INTER_LINEAR), axis=0))
            data = {'rgb': image, 'depth': tf.zeros(shape=[1, img_w, img_h, 1],dtype=tf.float32), 'labels': tf.zeros(shape=[1, img_w, img_h],dtype=tf.int32)}
            output = net.predict(data)
            outputColor = data_class.coloured_labels(labels=output)
            cv2.imwrite(os.path.join(output_dir,filename), cv2.resize(cv2.cvtColor(outputColor[0,:,:,:],cv2.COLOR_RGB2BGR),(256,256),interpolation=cv2.INTER_NEAREST), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        except KeyboardInterrupt:
            print('WARNING: Got Keyboard Interrupt')
            break

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
    predict_network(net, output_dir, paths, data_desc)


@ex.main
def main(modelname, net_config, dataset, starting_weights, input_folder, _run):
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
        predict_output(net,output_dir,eval_image_paths,data_desc)


if __name__ == '__main__':
    ex.run_commandline()
    # for some reason we have processes running in the background that won't stop
    # this is the only way to kill them
    os._exit(os.EX_OK)
