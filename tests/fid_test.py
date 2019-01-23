from __future__ import absolute_import, division, print_function
import os
import glob
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import fid
from scipy.misc import imread
import tensorflow as tf
import cv2
inception_path = fid.check_path_inception("/Users/David/Downloads/inception-2015-12-05") # download inception network

fid.create_inception_graph(inception_path)  # load the graph into the current TF graph

def get_images(path,prefix):
    paths = glob.glob(os.path.join(path,prefix+"*.png"))
    num_samples = len(paths)
    images = np.zeros((num_samples,256,256,3))

    for i,path in enumerate(paths):
        images[i,:,:,:]= cv2.imread(path)[...,::-1]
    return images
base_path = "/Users/David/masterThesis/pix2pix-tensorflow/dir/264_full/validation"
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
print("FID: %s" % fid_value)
