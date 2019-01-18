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

def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False, pad="SAME"):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1], padding=pad)

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

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
    with tf.variable_scope(scope):
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

# def define_G(x, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, n_blocks_local=3, norm='instance'):
#     norm_layer = instance_norm
#     if netG == 'global':
#         netG = GlobalGenerator(x, output_nc, ngf, n_downsample_global,n_blocks_global,norm_layer)
#     elif netG == 'local':
#         netG = LocalEnhancer(x, output_nc, ngf, n_downsample_global,n_blocks_global,n_local_enhancers,n_blocks_local,norm_layer)
#     elif netG == 'encoder':
#         netG = Encoder(x, output_nc, ngf, n_downsample_global, norm_layer)
#     else:
#         raise('Unknown generator called! netG: '+str(netG))
#     return netG
#
# def define_D(x, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getItermFeat=False):
#     norm_layer = instance_norm
#     netD = MultiscaleDiscriminator(x,ndf,n_layers_D,norm_layer,use_sigmoid,num_D,getItermFeat)
#     return netD
#
# def LocalEnhancer(x, output_nc, ngf=32, n_downsample_global=3,n_blocks_global=9,n_local_enhancers=1,n_blocks_local=3,norm_layer=instance_norm, reuse=tf.AUTO_REUSE):
#     ngf_global = ngf * (2**n_local_enhancers)
#     _, h = GlobalGenerator(x, output_nc, ngf_global, n_downsample_global,n_blocks_local, norm_layer, reuse=reuse)
#     padding = 3
#
#     local_enhancer_outs = {}
#     with tf.variable_scope('local_enhancer'):
#         for n in range(1, n_local_enhancers+1):
#             ngf_global = ngf * (2**(n_local_enhancers-n))
#             h = tf.pad(h, [[0, 0], [padding, padding], [padding, padding], [0, 0]], "REFLECT")
#             h = conv2d(h, ngf_global, k_h=7, k_w=7,name="le_"+str(n)+"_1_conv2d",pad="VALID")
#             h = norm_layer(h)
#             h = lrelu(h)
#             h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
#             h = conv2d(h, ngf_global*2, k_h=3, k_w=3, d_h=2, d_w=2,name="le_"+str(n)+"_2_conv2d",pad="VALID")
#             h = norm_layer(h)
#             h = lrelu(h)
#
#             mult = 2**n_downsampling
#             for i in range(n_blocks_local):
#                 with tf.variable_scope('block_{}'.format(i), reuse=reuse):
#                     h = residual_block(h, reuse=reuse)
#
#             h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
#             h = deconv2d(h, ngf_global,k_h=3, k_w=3, d_h=2, d_w=2,name="gg_+"str(n_downsampling-1-i)"+_deconv2d",pad="VALID")
#             h = norm_layer(h)
#             h = lrelu(h)
#
#             if n == n_local_enhancers:
#                 h = tf.pad(h, [[0, 0], [padding, padding], [padding, padding], [0, 0]], "REFLECT")
#                 h = conv2d(h, output_nc, k_h=7, k_w=7,name="le_out_conv2d",pad="VALID")
#                 h = tf.nn.tanh(h)
#
#         h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
#         return tf.nn.avg_pool(h,ksize=[[0, 0], [padding, padding], [padding, padding], [0, 0]],
#                                 strides=[[0, 0], [2, 2], [2, 2], [0, 0]], padding="VALID",name="avg_pool")
#
# def GlobalGenerator(x, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=instance_norm, reuse=tf.AUTO_REUSE):
#     assert(n_blocks>=0)
#     activation=lrelu
#     padding = 3
#     with tf.variable_scope('global_generator'):
#         h = tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]], "REFLECT")
#         h = conv2d(h, ngf, k_h=7, k_w=7,name="gg_0_conv2d",pad="VALID")
#         h = norm_layer(h)
#         h = activation(h)
#
#         for i in range(n_downsampling):
#             h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
#             mult = 2**(i+1)
#             h = conv2d(h, ngf * mult, k_h=3, k_w=3, d_h=2, d_w=2,name="gg_"+str(i+1)+"_conv2d",pad="VALID")
#             h = norm_layer(h)
#             h = activation(h)
#
#         mult = 2**n_downsampling
#         for i in range(n_blocks):
#             with tf.variable_scope('block_{}'.format(i), reuse=reuse):
#                 h = residual_block(h, reuse=reuse)
#
#         for i in range(n_downsampling):
#             mult = 2**(n_downsampling - i)
#             h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
#             h = deconv2d(h, int(ngf*mult/2),k_h=3, k_w=3, d_h=2, d_w=2,name="gg_+"str(n_downsampling-1-i)"+_deconv2d",pad="VALID")
#             h = norm_layer(h)
#             h = activation(h)
#
#         h0 = h
#         h = tf.pad(h, [[0, 0], [padding, padding], [padding, padding], [0, 0]], "REFLECT")
#         h = conv2d(h, output_nc, k_h=7, k_w=7,name="gg_out_conv2d",pad="VALID")
#
#
#         return tf.nn.tanh(h), h0
#
# def Encoder(x, output_nc, ngf=32, n_downsampling=4, norm_layer=instance_norm):
#     padding = 3
#
#     with tf.variable_scope('encoder'):
#         h = tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]], "REFLECT")
#         h = conv2d(h, ngf, k_h=7, k_w=7,name="en_0_conv2d",pad="VALID")
#         h = norm_layer(h)
#         h = lrelu(h)
#
#         for i in range(1,n_downsampling+1):
#             mult = 2**i
#             h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
#             h = conv2d(h, ngf*mult, k_h=3, k_w=3, d_h=2, d_w=2,name="en_"+str(n)+"_conv2d",pad="VALID")
#             h = norm_layer(h)
#             h = lrelu(h)
#
#         for i in range(1,n_downsampling+1):
#             mult = 2**(n_downsampling-i+1)
#             h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
#             h = deconv2d(h, int(nfg*mult/2),k_h=3, k_w=3, d_h=2, d_w=2,name="en_+"str(i)"+_deconv2d",pad="VALID")
#             h = norm_layer(h)
#             h = lrelu(h)
#
#         h = tf.pad(h, [[0, 0], [padding, padding], [padding, padding], [0, 0]], "REFLECT")
#         h = conv2d(h, output_nc, k_h=7, k_w=7,name="en_out_conv2d",pad="VALID")
#
#         return tf.nn.tanh(h)


def residual_block(x, n_channels=128, normalizer_fn=instance_norm,
        activation_fn=lrelu, kernel_size=3, scope=None, reuse=None):
    with tf.variable_scope(scope or 'residual', reuse=reuse):
        h = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        h = tfl.conv2d(
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
        h = tfl.conv2d(
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

# def discriminatorHD(self, image, y=None, reuse=False):
#
#     with tf.variable_scope("discriminator") as scope:
#
#         # image is 256 x 256 x (input_c_dim + output_c_dim)
#         if reuse:
#             tf.get_variable_scope().reuse_variables()
#         else:
#             assert tf.get_variable_scope().reuse == False
#
#         h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
#         # h0 is (128 x 128 x self.df_dim)
#         h1 = lrelu(instance_norm(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
#         # h1 is (64 x 64 x self.df_dim*2)
#         h2 = lrelu(instance_norm(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
#         # h2 is (32x 32 x self.df_dim*4)
#         h3 = lrelu(instance_norm(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv',pad="VALID")))
#         # h3 is (31 x 31 x self.df_dim*8)
#         h4 = conv2d(h3, 1, d_h=1, d_w=1, name='d_h4_conv',pad="VALID")
#         # h4 is (30 x 30 x 1)
#
#         return tf.nn.sigmoid(h4), h4
