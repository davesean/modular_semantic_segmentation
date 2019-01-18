import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

# Concatenating 2 tensors
def concatenate(x, y):
	X_shape = x.get_shape()
	Y_shape = y.get_shape()
	# concatenating on feature map axis
	return tf.concat([x, y], axis=3)

# Define activation function for the network
def lrelu_layer(x, leak=0.2, name="lrelu"):
	return tf.maximum(x, leak*x)

# Function for fully connected layer
# def linear_layer(x, output_size, scope=None, stddev=0.2, bias_start=0.0):
	# shape = x.get_shape().as_list()
	# with tf.variable_scope(scope or "Linear"):
	# 	matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
	# 	bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
	# 	if with_w: # return values along with parameters of fc_layer
	# 		return tf.matmul(x, matrix) + bias, matrix, bias
	# 	else:
	# 		return tf.matmul(x, matrix) + bias

def linear_layer(x, output_size, shape ,scope=None, stddev=0.2, bias_start=0.0):

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape, output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        return tf.matmul(tf.layers.flatten(x), matrix) + bias

# Function for BatchNormalization layer
def bn_layer(x, is_training, scope):
	return layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training, scope=scope)

# Function for 2D convolutional layer
def conv2d_layer(x, num_filters, filter_height, filter_width, stride_height, stride_width, stddev=0.2, name="conv2d"):
	with tf.variable_scope(name):
		w = tf.get_variable('weight', [filter_height, filter_width, x.get_shape()[-1], num_filters], initializer=tf.truncated_normal_initializer(stddev=stddev)) #weights
		s = [1, stride_height, stride_width, 1] # stride

		if name == 'res_convd1' or name == 'res_convd2':
			conv = tf.nn.conv2d(x, w, s, padding='SAME')
		else:
			conv = tf.nn.conv2d(x, w, s, padding='SAME')
			biases = tf.get_variable('bias', [num_filters], initializer=tf.constant_initializer(0.0))
			conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

		return conv


# Function for 2D Deconvolutional layer
def deconv2d_layer(x, out_channel, filter_height, filter_width, stride_height, stride_width, stddev=0.2, name="deconv2d"):
	with tf.variable_scope(name):
		in_channel = x.get_shape()[-1]
		out_shape = [int(x.get_shape()[0]), int(x.get_shape()[1]*stride_height), int(x.get_shape()[2]*stride_width), out_channel]
		#out_shape = tf.convert_to_tensor(out_shape)
		w = tf.get_variable("weight", [filter_height, filter_width, out_channel, x.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=stddev))
		s = [1, stride_height, stride_width, 1]
		deconv = tf.nn.conv2d_transpose(x, w, out_shape, s, padding='SAME')
		biases = tf.get_variable('bias', out_channel, initializer=tf.constant_initializer(0.0))
		deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

		return deconv

# Function for Residual Blocks
def residual_block1(input, num_filters, filter_size, is_training, name="res_block"):
	with tf.variable_scope(name):
	    x_shortcut = x
	    x = lrelu_layer(bn_layer(conv2d_layer(x, num_filters, filter_size, filter_size, 2, 2, name='res_convd1'), is_training=is_training, scope='ebn_1'))
	    x = bn_layer(conv2d_layer(x, num_filters, 1, 1, 1, 1, name='res_convd2'), is_training=is_training, scope='ebn_2')
	    x_shortcut = bn_layer(conv2d_layer(x_shortcut, num_filters, 1, 1, 1, 1, name='skip'), is_training=is_training, scope='ebn_3')
	    res = tf.nn.relu(x + x_shortcut)
	    return res

# Function for Residual Blocks
def residual_block(input, num_filters, filter_size, is_training, name="res_block"):
	with tf.variable_scope(name):
		in_filter = input.get_shape()[-1]
		out_filter = num_filters
		x_shortcut = input
		x = lrelu_layer(bn_layer(conv2d_layer(input, out_filter, 1, 1, 2, 2, name='sub_res_1'), is_training=is_training, scope='bn_1')) # 64 x 64 x 128
		x = lrelu_layer(bn_layer(conv2d_layer(x, out_filter, filter_size, filter_size, 1, 1, name='sub_res_2'), is_training=is_training, scope='bn_2')) # 64 x 64 x 128
		x = bn_layer(conv2d_layer(x, out_filter, 1, 1, 1, 1, name='sub_res_3'), is_training=is_training, scope='bn_3') # 64 x 64 x 128
		x_shortcut = bn_layer(conv2d_layer(x_shortcut, out_filter, 1, 1, 2, 2, name='res_skip'), is_training=is_training, scope='bn_skip') # 64 x 64 x 128

		res = tf.nn.relu(x + x_shortcut)

		return res

#
#
#
# import tensorflow as tf
# import tensorflow.contrib.layers as tfl
#
# def lrelu(x, leak=0.2, name="lrelu"):
#     """Summary
#     Parameters
#     ----------
#     x : TYPE
#         Description
#     leak : float, optional
#         Description
#     name : str, optional
#         Description
#     Returns
#     -------
#     TYPE
#         Description
#     """
#     with tf.variable_scope(name):
#         return tf.maximum(x, leak * x)
#
#
# def instance_norm(x, epsilon=1e-5):
#     """Instance Normalization.
#     See Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2016).
#     Instance Normalization: The Missing Ingredient for Fast Stylization,
#     Retrieved from http://arxiv.org/abs/1607.08022
#     Parameters
#     ----------
#     x : TYPE
#         Description
#     epsilon : float, optional
#         Description
#     Returns
#     -------
#     TYPE
#         Description
#     """
#     with tf.variable_scope('instance_norm'):
#         mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
#         scale = tf.get_variable(
#             name='scale',
#             shape=[x.get_shape()[-1]],
#             initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
#         offset = tf.get_variable(
#             name='offset',
#             shape=[x.get_shape()[-1]],
#             initializer=tf.constant_initializer(0.0))
#         out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset
#         return out
#
#
# def conv2d(inputs,
#            activation_fn=lrelu,
#            normalizer_fn=instance_norm,
#            scope='conv2d',
#            **kwargs):
#     """Summary
#     Parameters
#     ----------
#     inputs : TYPE
#         Description
#     activation_fn : TYPE, optional
#         Description
#     normalizer_fn : TYPE, optional
#         Description
#     scope : str, optional
#         Description
#     **kwargs
#         Description
#     Returns
#     -------
#     TYPE
#         Description
#     """
#     with tf.variable_scope(scope or 'conv2d'):
#         h = tfl.conv2d(
#             inputs=inputs,
#             activation_fn=None,
#             normalizer_fn=None,
#             weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
#             biases_initializer=None,
#             **kwargs)
#         if normalizer_fn:
#             h = normalizer_fn(h)
#         if activation_fn:
#             h = activation_fn(h)
#         return h
#
#
# def conv2d_transpose(inputs,
#                      activation_fn=lrelu,
#                      normalizer_fn=instance_norm,
#                      scope='conv2d_transpose',
#                      **kwargs):
#     """Summary
#     Parameters
#     ----------
#     inputs : TYPE
#         Description
#     activation_fn : TYPE, optional
#         Description
#     normalizer_fn : TYPE, optional
#         Description
#     scope : str, optional
#         Description
#     **kwargs
#         Description
#     Returns
#     -------
#     TYPE
#         Description
#     """
#     with tf.variable_scope(scope or 'conv2d_transpose'):
#         h = tfl.conv2d_transpose(
#             inputs=inputs,
#             activation_fn=None,
#             normalizer_fn=None,
#             weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
#             biases_initializer=None,
#             **kwargs)
#         if normalizer_fn:
#             h = normalizer_fn(h)
#         if activation_fn:
#             h = activation_fn(h)
#         return h
#
# def encoder(x, n_filters=32, k_size=3, normalizer_fn=instance_norm,
#         activation_fn=lrelu, scope=None, reuse=None):
#     with tf.variable_scope(scope or 'encoder', reuse=reuse):
#         h = tf.pad(x, [[0, 0], [k_size, k_size], [k_size, k_size], [0, 0]],
#                 "REFLECT")
#         h = tfl.conv2d(
#                 inputs=h,
#                 num_outputs=n_filters,
#                 kernel_size=7,
#                 stride=1,
#                 padding='VALID',
#                 weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
#                 biases_initializer=None,
#                 normalizer_fn=normalizer_fn,
#                 activation_fn=activation_fn,
#                 scope='1',
#                 reuse=reuse)
#         h = tfl.conv2d(
#                 inputs=h,
#                 num_outputs=n_filters * 2,
#                 kernel_size=k_size,
#                 stride=2,
#                 weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
#                 biases_initializer=None,
#                 normalizer_fn=normalizer_fn,
#                 activation_fn=activation_fn,
#                 scope='2',
#                 reuse=reuse)
#         h = tfl.conv2d(
#                 inputs=h,
#                 num_outputs=n_filters * 4,
#                 kernel_size=k_size,
#                 stride=2,
#                 weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
#                 biases_initializer=None,
#                 normalizer_fn=normalizer_fn,
#                 activation_fn=activation_fn,
#                 scope='3',
#                 reuse=reuse)
#     return h
#
# def residual_block(x, n_channels=128, normalizer_fn=instance_norm,
#         activation_fn=lrelu, kernel_size=3, scope=None, reuse=None):
#     with tf.variable_scope(scope or 'residual', reuse=reuse):
#         h = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
#         h = tfl.conv2d(
#                 inputs=h,
#                 num_outputs=n_channels,
#                 kernel_size=kernel_size,
#                 weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
#                 biases_initializer=None,
#                 normalizer_fn=normalizer_fn,
#                 padding='VALID',
#                 activation_fn=activation_fn,
#                 scope='1',
#                 reuse=reuse)
#         h = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
#         h = tfl.conv2d(
#                 inputs=h,
#                 num_outputs=n_channels,
#                 kernel_size=kernel_size,
#                 weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
#                 biases_initializer=None,
#                 normalizer_fn=normalizer_fn,
#                 padding='VALID',
#                 activation_fn=None,
#                 scope='2',
#                 reuse=reuse)
#         h = tf.add(x, h)
#     return h
#
# def transform(x, img_size=256, reuse=None):
#     h = x
#     if img_size >= 256:
#         n_blocks = 9
#     else:
#         n_blocks = 6
#     for block_i in range(n_blocks):
#         with tf.variable_scope('block_{}'.format(block_i), reuse=reuse):
#             h = residual_block(h, reuse=reuse)
#     return h
#
# def decoder(x, n_filters=32, k_size=3, normalizer_fn=instance_norm,
#     activation_fn=lrelu, scope=None, reuse=None):
#     with tf.variable_scope(scope or 'decoder', reuse=reuse):
#         h = tfl.conv2d_transpose(
#                 inputs=x,
#                 num_outputs=n_filters * 2,
#                 kernel_size=k_size,
#                 stride=2,
#                 weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
#                 biases_initializer=None,
#                 normalizer_fn=normalizer_fn,
#                 activation_fn=activation_fn,
#                 scope='1',
#                 reuse=reuse)
#         h = tfl.conv2d_transpose(
#                 inputs=h,
#                 num_outputs=n_filters,
#                 kernel_size=k_size,
#                 stride=2,
#                 weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
#                 biases_initializer=None,
#                 normalizer_fn=normalizer_fn,
#                 activation_fn=activation_fn,
#                 scope='2',
#                 reuse=reuse)
#         h = tf.pad(h, [[0, 0], [k_size, k_size], [k_size, k_size], [0, 0]],
#                 "REFLECT")
#         h = tfl.conv2d(
#                 inputs=h,
#                 num_outputs=3,
#                 kernel_size=7,
#                 stride=1,
#                 weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
#                 biases_initializer=None,
#                 padding='VALID',
#                 normalizer_fn=normalizer_fn,
#                 activation_fn=tf.nn.tanh,
#                 scope='3',
#                 reuse=reuse)
#     return h
#
# def generator(x, scope=None, reuse=None):
#     with tf.variable_scope(scope or 'generator', reuse=reuse):
#         h = encoder(x, reuse=reuse)
#         h = transform(h, reuse=reuse)
#         h = decoder(h, reuse=reuse)
#     return h
#
# def discriminator(x, n_filters=64, k_size=4, activation_fn=lrelu,
#     normalizer_fn=instance_norm, scope=None, reuse=None):
#     with tf.variable_scope(scope or 'discriminator', reuse=reuse):
#         h = tfl.conv2d(
#                 inputs=x,
#                 num_outputs=n_filters,
#                 kernel_size=k_size,
#                 stride=2,
#                 weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
#                 biases_initializer=None,
#                 activation_fn=activation_fn,
#                 normalizer_fn=None,
#                 scope='1',
#                 reuse=reuse)
#         h = tfl.conv2d(
#                 inputs=h,
#                 num_outputs=n_filters * 2,
#                 kernel_size=k_size,
#                 stride=2,
#                 weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
#                 biases_initializer=None,
#                 activation_fn=activation_fn,
#                 normalizer_fn=normalizer_fn,
#                 scope='2',
#                 reuse=reuse)
#         h = tfl.conv2d(
#                 inputs=h,
#                 num_outputs=n_filters * 4,
#                 kernel_size=k_size,
#                 stride=2,
#                 weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
#                 biases_initializer=None,
#                 activation_fn=activation_fn,
#                 normalizer_fn=normalizer_fn,
#                 scope='3',
#                 reuse=reuse)
#         h = tfl.conv2d(
#                 inputs=h,
#                 num_outputs=n_filters * 8,
#                 kernel_size=k_size,
#                 stride=1,
#                 weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
#                 biases_initializer=None,
#                 activation_fn=activation_fn,
#                 normalizer_fn=normalizer_fn,
#                 scope='4',
#                 reuse=reuse)
#         h = tfl.conv2d(
#                 inputs=h,
#                 num_outputs=1,
#                 kernel_size=k_size,
#                 stride=1,
#                 weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
#                 biases_initializer=None,
#                 activation_fn=tf.nn.sigmoid,
#                 scope='5',
#                 reuse=reuse)
#     return h
