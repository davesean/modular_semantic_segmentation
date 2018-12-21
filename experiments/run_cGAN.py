from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from experiments.utils import get_observer

from xview.datasets import Cityscapes_GAN
from xview.datasets import get_dataset
from xview.models import get_model
from xview.settings import EXP_OUT

import os
import sacred as sc
from sacred.utils import apply_backspaces_and_linefeeds
import scipy.misc
import numpy as np
import shutil
import tensorflow as tf
import argparse
import json
import glob
import random
import collections
import math
import time
import scipy
import cv2
from copy import deepcopy
from sys import stdout
from skimage.measure import compare_ssim

class Helper:
    name = 'A'

a = Helper()

EPS = 1e-12
num_test_images = 20

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
def main(dataset, net_config, _run):
    # Add all of the config into the helper class
    for key in net_config:
        setattr(a, key, net_config[key])

    setattr(a,'EXP_OUT',EXP_OUT)
    setattr(a,'RUN_id',_run._id)

    output_dir = create_directories(_run._id, ex)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # load the dataset class
        data = get_dataset(dataset['name'])
        data = data(**dataset)
        cGAN_model = get_model('cGAN')
        if a.checkpoint is not None:
            ckp = os.path.join(a.EXP_OUT,str(a.checkpoint))
        else:
            ckp = None
        model = cGAN_model(sess, image_size=a.input_image_size, batch_size=a.batch_size,
                        output_size=a.input_image_size, dataset_name=dataset['name'],
                        checkpoint_dir=output_dir, data=data,
                        data_desc=data.get_data_description(), momentum=a.batch_momentum,
                        L1_lambda=int(a.l1_weight/a.gan_weight), gf_dim=a.ngf,
                        df_dim=a.ndf,label_smoothing=a.label_smoothing,
                        noise_std_dev=a.noise_std_dev,
                        checkpoint=ckp)

        if a.mode == 'train':
            tmp = model.train(a)
            _run.info['predictions'] = tmp
            _run.info['mean_predictions'] = np.mean(tmp, axis=0)
        elif a.mode == 'valid':
            tmp = model.validate(a)
            _run.info['predictions'] = tmp
            _run.info['mean_predictions'] = np.mean(tmp, axis=0)
        else:
            model.outputAllData(a)


if __name__ == '__main__':
    ex.run_commandline()
    # for some reason we have processes running in the background that won't stop
    # this is the only way to kill them
    os._exit(os.EX_OK)
