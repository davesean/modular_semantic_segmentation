import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

class batch_norm(object):
            # h1 = lrelu(tf.contrib.layers.batch_norm(conv2d(h0, self.df_dim*2, name='d_h1_conv'),decay=0.9,updates_collections=None,epsilon=0.00001,scale=True,scope="d_h1_conv"))
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, scope=self.name)

def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.

    For brevity, let `x = `, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                              (1. - targets) * tf.log(1. - preds + eps)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv2d(input_, output_dim,
           k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02,
           name="conv2d",pad="SAME"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        if pad=="VALID":
            conv = tf.pad(input_, [[0,0],[1,1],[1,1],[0,0]], mode="CONSTANT")
            conv = tf.nn.conv2d(conv, w, strides=[1, d_h, d_w, 1], padding=pad)
        else:
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=pad)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        return conv

def deconv2d(input_, output_shape, filters=None,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False, pad="SAME"):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        try:
            w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                                initializer=tf.random_normal_initializer(stddev=stddev))
            biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))

        except TypeError:
            w = tf.get_variable('w', [k_h, k_w, filters, input_.get_shape()[-1]],
                                initializer=tf.random_normal_initializer(stddev=stddev))
            biases = tf.get_variable('biases', [filters], initializer=tf.constant_initializer(0.0))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1], padding=pad)

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])
        try:
            deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        except:
            deconv = deconv + biases

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    # shape = input_.get_shape().as_list()
    shape = 31 * 31 * 64 * 8
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape, output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def instance_norm(x, epsilon=1e-5, scope='instance_norm'):
    """Instance Normalization.
    See Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2016).
    Instance Normalization: The Missing Ingredient for Fast Stylization,
    Retrieved from http://arxiv.org/abs/1607.08022
    Parameters
    ----------
    x : TYPE
        Description
    epsilon : float, optional
        Description
    Returns
    -------
    TYPE
        Description
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable(
            name='scale',
            shape=[x.get_shape()[-1]],
            initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable(
            name='offset',
            shape=[x.get_shape()[-1]],
            initializer=tf.constant_initializer(0.0))
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset
        return out


def residual_block(x, n_channels=128, normalizer_fn=instance_norm,
        activation_fn=lrelu, kernel_size=3, scope=None, reuse=None):
    with tf.variable_scope(scope or 'residual', reuse=reuse):
        h = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        h = tf.contrib.layers.conv2d(
                inputs=h,
                num_outputs=n_channels,
                kernel_size=kernel_size,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                biases_initializer=None,
                normalizer_fn=normalizer_fn,
                padding='VALID',
                activation_fn=activation_fn,
                scope='1',
                reuse=reuse)
        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        h = tf.contrib.layers.conv2d(
                inputs=h,
                num_outputs=n_channels,
                kernel_size=kernel_size,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                biases_initializer=None,
                normalizer_fn=normalizer_fn,
                padding='VALID',
                activation_fn=None,
                scope='2',
                reuse=reuse)
        h = tf.add(x, h)
    return h
