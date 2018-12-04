from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from experiments.utils import get_observer, load_data
from xview.datasets import get_dataset
from experiments.evaluation import evaluate, import_weights_into_network
from xview.models import get_model
from xview.settings import EXP_OUT
import os
import sys
import shutil
import cv2
import glob
import numpy as np
import tensorflow as tf


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


def predict_network(net, output_dir, data, image_dir, starting_weights, experiment,
                  additional_eval_data={},image_lists=None):
    """
    Train a network on a given dataset.

    Args:
        net: An instance of a `base_model` class.
        output_dir: A directory path. This function will add all files foudn at this path
            as artifacts to the experiment.
        data: A dataset in one of the formats accepted by xview.models.base_model
        starting_weights: Desriptor for weight sto load into network. If not false or
            empty, will load weights as described in `evaluation.py`.
        experiment: The current sacred experiment.
    """

    # Train the given network
    if starting_weights:
        net.import_weights(filepath=starting_weights)


    # Create a folder for the output
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    folder_path = os.path.join(image_dir,starting_weights.split('/')[-2]+"_full")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    validation_path = os.path.join(folder_path,"validation")
    measure_path = os.path.join(folder_path,"measure")
    training_path = os.path.join(folder_path,"training")

    if not os.path.exists(validation_path):
        os.makedirs(validation_path)

    if not os.path.exists(measure_path):
        os.makedirs(measure_path)

    if not os.path.exists(training_path):
        os.makedirs(training_path)


    if image_lists is None:
        # validation_images = net.predict(data.get_validation_set())
        #
        # for i in range(validation_images.shape[0]):
        #     outputColor = data.coloured_labels(labels=validation_images[i,:,:])
        #     filename = "input_validation"+str(i+1)+".png"
        #     cv2.imwrite(os.path.join(validation_path,filename), cv2.resize(cv2.cvtColor(outputColor,cv2.COLOR_RGB2BGR),(256,256),interpolation=cv2.INTER_NEAREST), [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # measure_images = net.predict(data.get_measureset())
        #
        # for i in range(measure_images.shape[0]):
        #     outputColor = data.coloured_labels(labels=measure_images[i,:,:])
        #     filename = "input_measure"+str(i+1)+".png"
        #     cv2.imwrite(os.path.join(measure_path,filename), cv2.resize(cv2.cvtColor(outputColor,cv2.COLOR_RGB2BGR),(256,256),interpolation=cv2.INTER_NEAREST), [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        training_images = net.predict(data.get_trainset())

        for i in range(training_images.shape[0]):
            outputColor = data.coloured_labels(labels=training_images[i,:,:])
            filename = "input_training"+str(i+1)+".png"
            cv2.imwrite(os.path.join(training_path,filename), cv2.resize(cv2.cvtColor(outputColor,cv2.COLOR_RGB2BGR),(256,256),interpolation=cv2.INTER_NEAREST), [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    else:

        data_class = data
        for image in image_lists:
            num = image.split('/')[-1].split('.')[0].split('g')[-1]
            input = np.expand_dims(cv2.imread(image), axis=0)
            data = {'labels': tf.zeros_like(input[:,:,:,0], dtype=tf.int32), 'rgb': tf.to_float(input), 'depth': tf.zeros(shape=[1, 256, 256, 1],dtype=tf.float32) }

            output = net.predict(data)

            outputColor = data_class.coloured_labels(labels=output)

            filename = "input_training"+num+".png"
            cv2.imwrite(os.path.join(training_path,filename), cv2.resize(cv2.cvtColor(outputColor[0,:,:,:],cv2.COLOR_RGB2BGR),(256,256),interpolation=cv2.INTER_NEAREST), [int(cv2.IMWRITE_JPEG_QUALITY), 90])



ex = Experiment()
# reduce output of progress bars
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(get_observer())


@ex.capture
def predict_training(net, output_dir, data, image_dir, starting_weights, _run):
    """Output all training images given the network."""
    predict_network(net, output_dir, data, image_dir, starting_weights, ex)

@ex.capture
def predict_inputs(net, output_dir, data, image_dir, target_list, starting_weights, _run):
    """Output all training images given the network."""
    predict_network(net, output_dir, data, image_dir, starting_weights, ex, image_lists=target_list)

@ex.main
def main(modelname, dataset, net_config, _run):
    # Set up the directories for diagnostics
    output_dir = create_directories(_run._id, ex)

    # load the dataset class, but don't instantiate it
    data = get_dataset(dataset['name'])

    # create the network
    model = get_model(modelname)
    with model(data_description=data.get_data_description(), output_dir=output_dir,
               **net_config) as net:
        # now we can load the dataset inside the scope of the network graph
        if net_config['input_dir'] is not None:
            data = data()
            target_list = glob.glob(os.path.join(net_config['input_dir'],"target_training*.png"))
            target_list.sort()
            print(len(target_list))
            predict_inputs(net, output_dir, data, net_config['file_output_dir'],target_list)
        else:
            data = data(**dataset)
            predict_training(net, output_dir, data, net_config['file_output_dir'])


if __name__ == '__main__':
    ex.run_commandline()
    # for some reason we have processes running in the background that won't stop
    # this is the only way to kill them
    os._exit(os.EX_OK)
