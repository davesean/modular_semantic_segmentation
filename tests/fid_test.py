from __future__ import absolute_import, division, print_function
import os
import glob
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import fid
from scipy.misc import imread
import tensorflow as tf
import cv2
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="path to folder with images")
parser.add_argument("--n", help="path to folder with images")
a = parser.parse_args()

inception_path = fid.check_path_inception("/Users/David/Downloads/inception-2015-12-05") # download inception network

fid.create_inception_graph(inception_path)  # load the graph into the current TF graph

def get_images(path,prefix):
    paths = glob.glob(os.path.join(path,prefix+"*.png"))
    test_load = cv2.imread(paths[0])
    img_size = test_load.shape[-2]
    c_channel = test_load.shape[-1]
    if prefix.split('_')[-1] == "training":
        num_samples = a.n

        random.seed(42)
        paths = random.sample(paths,num_samples)
    else:
        num_samples = len(paths)
    images = np.zeros((num_samples,img_size,img_size,c_channel))



    for i,path in enumerate(paths):
        images[i,:,:,:]= cv2.imread(path)[...,::-1]
    return images
base_path = a.path
num = base_path.split('/')[-2].split('_')[0]
ending = base_path.split('/')[-1]
set_real = get_images(base_path, num+"_"+ending)
set_fake = get_images(base_path, "target_"+ending)

assert(set_real.shape[0] is not 0 and set_fake.shape[0] is not 0)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mu_real, sigma_real = fid.calculate_activation_statistics(set_real, sess, batch_size=set_real.shape[0])
    mu_gen, sigma_gen = fid.calculate_activation_statistics(set_fake, sess, batch_size=set_fake.shape[0])

fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
print("FID for %s set of run %s: %s" % (ending,num,fid_value))
