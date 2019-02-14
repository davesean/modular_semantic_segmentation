import tensorflow as tf
from .vgg16 import vgg16
from tensorflow.python.layers.layers import max_pooling2d
from xview.models.cGAN_ops import residual_block, deconv2d, instance_norm

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.layers.batch_normalization(x,momentum=self.momentum, epsilon=self.epsilon, name=self.name, training=train)
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)
def conv2d(input_, output_dim,
           k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02,
           name="conv2d",pad="SAME"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        if pad=="VALID":
            conv = tf.pad(input_, [[0,0],[1,1],[1,1],[0,0]], mode="CONSTANT")
            conv = tf.nn.conv2d(conv, w, strides=[1, d_h, d_w, 1], padding=pad)
        elif pad=="VALID_NOPAD":
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='VALID')
        else:
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=pad)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        return conv
def maxPool2d(input_,
           k_h=4, k_w=4, d_h=2, d_w=2,
           name="maxpool2d",pad="SAME"):
           return tf.layers.max_pooling2d(inputs=input_, pool_size=[k_h,k_w], strides=[d_h,d_w], padding=pad)
def dense(input_, output_size, input_size, num_channels, name="dense", reuse=False, stddev=0.02, bias_start=0.0):
    shape = input_size * input_size * num_channels
    with tf.variable_scope(name):
        matrix = tf.get_variable("Matrix", [shape, output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        return tf.matmul(tf.layers.flatten(input_), matrix) + bias



class simArch(object):

    def arch1(self, image, params, y=None, reuse=False, is_training=True, var_scope="sim_disc"):
        # image is 256 x 256 x (input_c_dim + input_c_dim)
        with tf.variable_scope(var_scope) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, k_h=4, k_w=4, d_h=4, d_w=4, name='s_h0_conv'))
            # h0 is (64 x 64 x self.df_dim)
            # h1 = lrelu(self.s_bn1(conv2d(h0, self.df_dim*2, k_h=4, k_w=4, d_h=4, d_w=4, name='s_h1_conv'),train=is_training))
            h1 = lrelu(conv2d(h0, self.df_dim*2, k_h=4, k_w=4, d_h=4, d_w=4, name='s_h1_conv'))
            # h1 is (16 x 16 x self.df_dim*2)
            h2 = conv2d(h1, 1, k_h=2, k_w=2, d_h=2, d_w=2, name='s_h2_conv')
            # h2 is (8 x 8 x 1)

            return tf.nn.sigmoid(h2), h2, params['entropy']

    archs = {
        'arch1': arch1
    }

    def arch2(self, image, params, y=None, reuse=False, is_training=True, var_scope="sim_disc"):
        # image is 256 x 256 x (input_c_dim + input_c_dim)
        with tf.variable_scope(var_scope) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, k_h=1, k_w=1, d_h=1, d_w=1, name='s_h0_conv'))
            # h0 is (256 x 256 x self.df_dim)
            h1 = lrelu(conv2d(h0, self.df_dim*2, k_h=4, k_w=4, d_h=4, d_w=4, name='s_h1_conv'))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = maxPool2d(input_=h1, k_h=2, k_w=2, d_h=2, d_w=2, name="s_h2_maxpool2d")
            # h2 is (32 x 32 x self.df_dim*2)
            h3 = lrelu(conv2d(h2, self.df_dim*4, k_h=2, k_w=2, d_h=2, d_w=2, name='s_h3_conv'))
            # h3 is (16 x 16 x self.df_dim*4)
            h4 = conv2d(h3, 1, k_h=2, k_w=2, d_h=2, d_w=2, name='s_h4_conv')
            # h4 is (8 x 8 x 1)

            return tf.nn.sigmoid(h4), h4, params['entropy']

    archs['arch2'] = arch2

    def arch4(self, image, params, y=None, reuse=False, is_training=True, var_scope="sim_disc"):
        with tf.variable_scope(var_scope) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            h0 = lrelu(conv2d(image, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h0_conv'))
            h1 = lrelu(conv2d(h0, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h1_conv'))
            pool1 = max_pooling2d(h1, [2, 2], [2, 2], name='s_pool1')
            h2 = lrelu(conv2d(pool1, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h2_conv'))
            h3 = lrelu(conv2d(h2, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h3_conv'))
            pool2 = max_pooling2d(h3, [2, 2], [2, 2], name='s_pool2')
            h4 = lrelu(conv2d(pool2, 256, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h4_conv'))
            h5 = conv2d(h4, 1, k_h=8, k_w=8, d_h=8, d_w=8, name='s_h5_conv')

            return tf.nn.sigmoid(h5), h5, params['entropy']

    archs['arch4'] = arch4

    def arch5(self, image, params, y=None, reuse=False, is_training=True, var_scope="sim_disc"):
        "2ch pytorch impl from https://github.com/szagoruyko/cvpr15deepcompare/blob/master/pytorch/eval.py"
        with tf.variable_scope(var_scope) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            image = (image+1)/2

            h0 = lrelu(conv2d(image, 96, k_h=7, k_w=7, d_h=3, d_w=3, name='s_h0_conv'))
            pool1 = max_pooling2d(h0, [2, 2], [2, 2], name='s_pool1', padding='same')
            h1 = lrelu(conv2d(pool1, 192, k_h=5, k_w=5, d_h=1, d_w=1, name='s_h1_conv'))
            pool2 = max_pooling2d(h1, [2, 2], [2, 2], name='s_pool2', padding='same')
            h2 = lrelu(conv2d(pool2, 256, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h2_conv'))


            out = dense(h2, input_size=3, output_size=1, num_channels=256,
                        reuse=reuse, name='s_dense_out')
            return tf.nn.sigmoid(out), out, params['entropy']

    archs['arch5'] = arch5

    def arch6(self, image, params, y=None, reuse=False, is_training=True, var_scope="sim_disc"):
        with tf.variable_scope(var_scope) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            h = (image+1)/2

            h = tf.pad(h, [[0, 0], [2, 2], [2, 2], [0, 0]], "REFLECT")

            h0 = lrelu(conv2d(h, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h0_conv',pad="VALID_NOPAD"))
            h1 = lrelu(conv2d(h0, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h1_conv',pad="VALID_NOPAD"))

            # flat = tf.layers.flatten(h1,name='s_flatten')
            # out = dense(flat, input_size=32, output_size=1024, num_channels=128,
            #             reuse=reuse, name='s_dense_out')
            out = (conv2d(h1, 1, k_h=1, k_w=1, d_h=1, d_w=1, name='s_h2_conv'))

            return tf.nn.sigmoid(out), out, params['entropy']

    archs['arch6'] = arch6

    def arch7(self, image, params, y=None, reuse=False, is_training=True, var_scope="sim_disc"):
        with tf.variable_scope(var_scope) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            h = (image+1)/2

            h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            h0 = lrelu(conv2d(h, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h0_conv',pad="VALID_NOPAD"))
            h1 = lrelu(conv2d(h0, 128, k_h=5, k_w=5, d_h=1, d_w=1, name='s_h1_conv',pad="VALID_NOPAD"))
            h = h1
            for i in range(3):
                h = residual_block(h, n_channels=128, kernel_size=3, scope="resBlock_"+str(i), reuse=reuse)
            h = lrelu(conv2d(h, 64, k_h=1, k_w=1, d_h=1, d_w=1, name='s_h2_conv'))
            h = (conv2d(h, 1, k_h=1, k_w=1, d_h=1, d_w=1, name='s_h3_conv'))
            return tf.nn.sigmoid(h), h, params['entropy']


    archs['arch7'] = arch7

    def arch8(self, image, params, y=None, reuse=False, is_training=True, var_scope="sim_disc"):
        with tf.variable_scope(var_scope) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            h = (image+1)/2

            h = tf.pad(h, [[0, 0], [8, 8], [8, 8], [0, 0]], "REFLECT")

            h0 = lrelu(conv2d(h, 128, k_h=15, k_w=15, d_h=1, d_w=1, name='s_h0_conv',pad="VALID_NOPAD"))
            h1 = lrelu(conv2d(h0, 256, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h1_conv',pad="VALID_NOPAD"))
            h = h1
            for i in range(5):
                h = residual_block(h, n_channels=256, kernel_size=3, scope="resBlock_"+str(i), reuse=reuse)
            h = lrelu(conv2d(h, 128, k_h=1, k_w=1, d_h=1, d_w=1, name='s_h2_conv'))
            h = (conv2d(h, 1, k_h=1, k_w=1, d_h=1, d_w=1, name='s_h3_conv'))
            return tf.nn.sigmoid(h), h, params['entropy']


    archs['arch8'] = arch8

    def arch9(self, image, params, y=None, reuse=False, is_training=True, var_scope="sim_disc"):
        with tf.variable_scope(var_scope) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            h = ((image+1)/2)*0.79375

            h0 = lrelu(conv2d(h, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h0_conv'))
            h1 = lrelu(conv2d(h0, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h1_conv'))
            pool1 = max_pooling2d(h1, [2, 2], [2, 2], name='s_pool1')
            h2 = lrelu(conv2d(pool1, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h2_conv'))
            h3 = lrelu(conv2d(h2, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h3_conv'))
            pool2 = max_pooling2d(h3, [2, 2], [2, 2], name='s_pool2')
            h4 = lrelu(conv2d(pool2, 256, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h4_conv'))
            # 8*8*256
            # flat = tf.layers.flatten(h4,name='s_flatten')
            d1 = lrelu(dense(h4, input_size=8, output_size=1024, num_channels=256,
                        reuse=reuse, name='s_dense_out1'))
            d3 = dense(d1, input_size=2, output_size=1, num_channels=256,
                        reuse=reuse, name='s_dense_out3')
            # return tf.nn.softmax(d3), d3, 'softmax'
            return tf.nn.sigmoid(d3), d3, params['entropy']

    archs['arch9'] = arch9

    def arch10(self, image, params, y=None, reuse=False, is_training=True, var_scope="sim_disc"):
        with tf.variable_scope(var_scope) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            h = ((image+1)/2)*0.79375

            h0 = lrelu(conv2d(h, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h0_conv'))
            h1 = lrelu(conv2d(h0, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h1_conv'))
            pool1 = max_pooling2d(h1, [2, 2], [2, 2], name='s_pool1')
            h2 = lrelu(conv2d(pool1, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h2_conv'))
            h3 = lrelu(conv2d(h2, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h3_conv'))
            pool2 = max_pooling2d(h3, [2, 2], [2, 2], name='s_pool2')
            h4 = lrelu(conv2d(pool2, 256, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h4_conv'))
            # 8*8*256
            # flat = tf.layers.flatten(h4,name='s_flatten')
            d1 = lrelu(dense(h4, input_size=8, output_size=1024, num_channels=256,
                        reuse=reuse, name='s_dense_out1'))
            d2 = lrelu(dense(d1, input_size=2, output_size=1024, num_channels=256,
                        reuse=reuse, name='s_dense_out2'))
            d3 = dense(d2, input_size=2, output_size=1, num_channels=256,
                        reuse=reuse, name='s_dense_out3')
            # return tf.nn.softmax(d3), d3, 'softmax'
            return tf.nn.sigmoid(d3), d3, params['entropy']

    archs['arch10'] = arch10

    def arch12(self, image, params, y=None, reuse=False, is_training=True, var_scope="sim_disc"):
        with tf.variable_scope(var_scope) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            h = ((image+1)/2)*0.79375

            h0 = lrelu(conv2d(h, self.df_dim, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h0_conv'))
            h1 = lrelu(conv2d(h0, self.df_dim, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h1_conv'))
            pool1 = max_pooling2d(h1, [2, 2], [2, 2], name='s_pool1')
            h2 = lrelu(conv2d(pool1, self.df_dim*2, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h2_conv'))
            h3 = lrelu(conv2d(h2, self.df_dim*2, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h3_conv'))
            pool2 = max_pooling2d(h3, [2, 2], [2, 2], name='s_pool2')
            h4 = lrelu(conv2d(pool2, self.df_dim*4, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h4_conv'))
            h5 = lrelu(conv2d(h4, self.df_dim*4, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h5_conv'))
            h6 = lrelu(conv2d(h5, self.df_dim*4, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h6_conv'))
            pool3 = max_pooling2d(h6, [2, 2], [2, 2], name='s_pool2')
            h7 = lrelu(conv2d(pool3, self.df_dim*4, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h7_conv'))

            # 4x4x256
            dh1_out_shape = tf.stack([tf.shape(image)[0], tf.to_int32(tf.shape(image)[1]/4), tf.to_int32(tf.shape(image)[2]/4), self.df_dim*2])
            dh1 = lrelu(deconv2d(h7,output_shape=dh1_out_shape, name='s_dh1', filters=self.df_dim*2))
            # 8x8x128
            dh2_out_shape = tf.stack([tf.shape(image)[0], tf.to_int32(tf.shape(image)[1]/2), tf.to_int32(tf.shape(image)[2]/2), self.df_dim])
            dh2 = lrelu(deconv2d(dh1,output_shape=dh2_out_shape, name='s_dh2', filters=self.df_dim))
            # 16x16x64
            dh3_out_shape = tf.stack([tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2], 1])
            dh3 = deconv2d(dh2,output_shape=dh3_out_shape, name='s_dh3', filters=1)
            # 32x32x1

            return tf.nn.sigmoid(dh3), dh3, params['entropy']

    archs['arch12'] = arch12

    def arch13(self, image, params, y=None, reuse=False, is_training=True, var_scope="sim_disc"):
        with tf.variable_scope(var_scope) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            h = ((image+1)/2)*0.79375
            bp = 11
            h = tf.pad(h, [[0, 0], [bp, bp], [bp, bp], [0, 0]], "REFLECT")

            h0 = lrelu(conv2d(h, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h0_conv',pad="VALID_NOPAD"))
            h1 = lrelu(conv2d(h0, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h1_conv',pad="VALID_NOPAD"))
            pool1 = max_pooling2d(h1, [2, 2], [2, 2], name='s_pool1',padding='VALID')
            h2 = lrelu(conv2d(pool1, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h2_conv',pad="VALID_NOPAD"))
            h3 = lrelu(conv2d(h2, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h3_conv',pad="VALID_NOPAD"))
            pool2 = max_pooling2d(h3, [2, 2], [2, 2], name='s_pool2',padding='VALID')
            h4 = lrelu(conv2d(pool2, 256, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h4_conv',pad="VALID_NOPAD"))
            # # 8x8x256

            # h5 = lrelu(conv2d(h4, 256, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h5_conv'))
            # h6 = lrelu(conv2d(h5, 256, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h6_conv'))
            # pool3 = max_pooling2d(h6, [2, 2], [2, 2], name='s_pool2')
            # h7 = lrelu(conv2d(pool3, 256, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h7_conv'))

            # # 4x4x256
            # dh1 = lrelu(deconv2d(h7,output_shape=[params['batch_size'], 8, 8, 128], name='s_dh1'))
            # 8x8x128
            dh2_out_shape = tf.stack([tf.shape(image)[0], tf.to_int32(tf.shape(image)[1]/2), tf.to_int32(tf.shape(image)[2]/2), self.df_dim])
            dh2 = lrelu(deconv2d(h4,output_shape=dh2_out_shape, name='s_dh2', filters=self.df_dim))
            # 16x16x64
            dh3_out_shape = tf.stack([tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2], 1])
            dh3 = deconv2d(dh2,output_shape=dh3_out_shape, name='s_dh3', filters=1)
            # 32x32x1

            return tf.nn.sigmoid(dh3), dh3, params['entropy']

    archs['arch13'] = arch13

    def arch14(self, image, params, y=None, reuse=False, is_training=True, var_scope="sim_disc"):
        with tf.variable_scope(var_scope) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            h = ((image+1)/2)*0.79375
            bp = 11
            h = tf.pad(h, [[0, 0], [bp, bp], [bp, bp], [0, 0]], "REFLECT")

            h0 = lrelu(conv2d(h, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h0_conv',pad="VALID_NOPAD"))
            h1 = lrelu(conv2d(h0, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h1_conv',pad="VALID_NOPAD"))
            pool1 = max_pooling2d(h1, [2, 2], [2, 2], name='s_pool1',padding='VALID')
            h2 = lrelu(conv2d(pool1, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h2_conv',pad="VALID_NOPAD"))
            h3 = lrelu(conv2d(h2, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h3_conv',pad="VALID_NOPAD"))
            pool2 = max_pooling2d(h3, [2, 2], [2, 2], name='s_pool2',padding='VALID')
            h4 = lrelu(conv2d(pool2, 256, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h4_conv',pad="VALID_NOPAD"))
            h4 = conv2d(h4, 1, k_h=1, k_w=1, d_h=1, d_w=1, name='s_h5_conv')

            out = tf.image.resize_images(h4, [tf.shape(image)[1], tf.shape(image)[2]])

            return tf.nn.sigmoid(out), out, params['entropy']

    archs['arch14'] = arch14

    def arch15(self, image, params, y=None, reuse=False, is_training=True, var_scope="sim_disc"):
        with tf.variable_scope(var_scope) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            # 32x32x6
            h = ((image+1)/2)

            h0 = lrelu(conv2d(h, self.df_dim, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h0_conv'))
            h1 = lrelu(conv2d(h0, self.df_dim, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h1_conv'))
            pool1 = max_pooling2d(h1, [2, 2], [2, 2], name='s_pool1')
            h2 = lrelu(conv2d(pool1, self.df_dim*2, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h2_conv'))
            h3 = lrelu(conv2d(h2, self.df_dim*2, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h3_conv'))
            pool2 = max_pooling2d(h3, [2, 2], [2, 2], name='s_pool2')
            h4 = lrelu(conv2d(pool2, self.df_dim*4, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h4_conv'))
            h5 = lrelu(conv2d(h4, self.df_dim*4, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h5_conv'))
            h6 = lrelu(conv2d(h5, self.df_dim*4, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h6_conv'))

            # 8x8x128
            dh2_out_shape = tf.stack([tf.shape(image)[0], tf.to_int32(tf.shape(image)[1]/2), tf.to_int32(tf.shape(image)[2]/2), self.df_dim*2])
            dh2 = lrelu(deconv2d(h6,output_shape=dh2_out_shape, name='s_dh2', filters=self.df_dim*2))
            # 16x16x64
            dh3_out_shape = tf.stack([tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2], 1])
            dh3 = deconv2d(dh2,output_shape=dh3_out_shape, name='s_dh3', filters=1)
            # 32x32x1

            return tf.nn.sigmoid(dh3), dh3, params['entropy']

    archs['arch15'] = arch15

    def arch16(self, image, params, y=None, reuse=False, is_training=True, var_scope="sim_disc"):
        with tf.variable_scope(var_scope) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            h = ((image+1)/2)*0.79375
            #32x32x6
            h0 = lrelu(conv2d(h, self.df_dim, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h0_conv'))
            #32x32xself.df_dim
            h1 = lrelu(conv2d(h0, self.df_dim, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h1_conv'))
            #32x32xself.df_dim
            pool1 = max_pooling2d(h1, [2, 2], [2, 2], name='s_pool1')
            #16x16xself.df_dim
            h2 = lrelu(conv2d(pool1, self.df_dim*2, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h2_conv'))
            #16x16xself.df_dim*2
            h3 = lrelu(conv2d(h2, self.df_dim*2, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h3_conv'))
            #16x16xself.df_dim*2
            pool2 = max_pooling2d(h3, [2, 2], [2, 2], name='s_pool2')
            #8x8xself.df_dim*2
            h4 = lrelu(conv2d(pool2, self.df_dim*4, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h4_conv'))
            h5 = lrelu(conv2d(h4, self.df_dim*4, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h5_conv'))
            h6 = lrelu(conv2d(h5, self.df_dim*4, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h6_conv'))
            #8x8xself.df_dim*4
            pool3 = max_pooling2d(h6, [2, 2], [2, 2], name='s_pool2')
            #4x4xself.df_dim*4
            h7 = lrelu(conv2d(pool3, self.df_dim*4, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h7_conv'))

            # 4x4x256
            dh1_out_shape = tf.stack([tf.shape(image)[0], tf.to_int32(tf.shape(image)[1]/4), tf.to_int32(tf.shape(image)[2]/4), self.df_dim*2])
            dh1 = lrelu(deconv2d(h7,output_shape=dh1_out_shape, name='s_dh1', filters=self.df_dim*2))
            dh1 = tf.concat([dh1, pool2], 3)
            # 8x8x128 x2
            dh2_out_shape = tf.stack([tf.shape(image)[0], tf.to_int32(tf.shape(image)[1]/2), tf.to_int32(tf.shape(image)[2]/2), self.df_dim])
            dh2 = lrelu(deconv2d(dh1,output_shape=dh2_out_shape, name='s_dh2', filters=self.df_dim))
            dh2 = tf.concat([dh2, pool1], 3)
            # 16x16x64 x2
            dh3_out_shape = tf.stack([tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2], 1])
            dh3 = deconv2d(dh2,output_shape=dh3_out_shape, name='s_dh3', filters=1)
            # 32x32x1

            return tf.nn.sigmoid(dh3), dh3, params['entropy']

    archs['arch16'] = arch16

    def arch17(self, image, params, y=None, reuse=False, is_training=True, var_scope="sim_disc"):
        with tf.variable_scope(var_scope) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            h = ((image+1)/2)*0.79375
            #32x32x6
            h0 = lrelu(conv2d(h, self.df_dim, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h0_conv'))
            #32x32xself.df_dim
            h1 = lrelu(instance_norm(conv2d(h0, self.df_dim, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h1_conv'), scope="s_h1_IN"))
            #32x32xself.df_dim
            pool1 = max_pooling2d(h1, [2, 2], [2, 2], name='s_pool1')
            #16x16xself.df_dim
            h2 = lrelu(instance_norm(conv2d(pool1, self.df_dim*2, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h2_conv'), scope="s_h2_IN"))
            #16x16xself.df_dim*2
            h3 = lrelu(instance_norm(conv2d(h2, self.df_dim*2, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h3_conv'), scope="s_h3_IN"))
            #16x16xself.df_dim*2
            pool2 = max_pooling2d(h3, [2, 2], [2, 2], name='s_pool2')
            #8x8xself.df_dim*2
            h4 = lrelu(instance_norm(conv2d(pool2, self.df_dim*4, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h4_conv'), scope="s_h4_IN"))
            h5 = lrelu(instance_norm(conv2d(h4, self.df_dim*4, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h5_conv'), scope="s_h5_IN"))
            h6 = lrelu(instance_norm(conv2d(h5, self.df_dim*4, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h6_conv'), scope="s_h6_IN"))
            #8x8xself.df_dim*4
            pool3 = max_pooling2d(h6, [2, 2], [2, 2], name='s_pool2')
            #4x4xself.df_dim*4
            h7 = lrelu(instance_norm(conv2d(pool3, self.df_dim*4, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h7_conv'), scope="s_h7_IN"))

            # 4x4x256
            dh1_out_shape = tf.stack([tf.shape(image)[0], tf.to_int32(tf.shape(image)[1]/4), tf.to_int32(tf.shape(image)[2]/4), self.df_dim*2])
            dh1 = lrelu(instance_norm(deconv2d(h7,output_shape=dh1_out_shape, name='s_dh1', filters=self.df_dim*2), scope="s_dh1_IN"))
            dh1 = tf.concat([dh1, pool2], 3)
            # 8x8x128 x2
            dh2_out_shape = tf.stack([tf.shape(image)[0], tf.to_int32(tf.shape(image)[1]/2), tf.to_int32(tf.shape(image)[2]/2), self.df_dim])
            dh2 = lrelu(instance_norm(deconv2d(dh1,output_shape=dh2_out_shape, name='s_dh2', filters=self.df_dim),scope="s_dh2_IN"))
            dh2 = tf.concat([dh2, pool1], 3)
            # 16x16x64 x2
            dh3_out_shape = tf.stack([tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2], 1])
            dh3 = deconv2d(dh2,output_shape=dh3_out_shape, name='s_dh3', filters=1)
            # 32x32x1

            return tf.nn.sigmoid(dh3), dh3, params['entropy']

    archs['arch17'] = arch17

    def arch18(self, image, params, y=None, reuse=False, is_training=True, var_scope="sim_disc"):
        with tf.variable_scope(var_scope) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            h = ((image+1)/2)*0.79375
            #32x32x6
            h0 = lrelu(conv2d(h, self.df_dim, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h0_conv'))
            #32x32xself.df_dim
            h1 = lrelu(instance_norm(conv2d(h0, self.df_dim, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h1_conv'), scope="s_h1_IN"))
            #32x32xself.df_dim
            pool1 = max_pooling2d(h1, [2, 2], [2, 2], name='s_pool1')
            #16x16xself.df_dim
            h2 = lrelu(instance_norm(conv2d(pool1, self.df_dim*2, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h2_conv'), scope="s_h2_IN"))
            #16x16xself.df_dim*2
            h3 = lrelu(instance_norm(conv2d(h2, self.df_dim*2, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h3_conv'), scope="s_h3_IN"))
            #16x16xself.df_dim*2
            pool2 = max_pooling2d(h3, [2, 2], [2, 2], name='s_pool2')
            #8x8xself.df_dim*2
            h4 = lrelu(instance_norm(conv2d(pool2, self.df_dim*4, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h4_conv'), scope="s_h4_IN"))
            h5 = lrelu(instance_norm(conv2d(h4, self.df_dim*4, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h5_conv'), scope="s_h5_IN"))
            h6 = lrelu(instance_norm(conv2d(h5, self.df_dim*4, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h6_conv'), scope="s_h6_IN"))
            #8x8xself.df_dim*4
            pool3 = max_pooling2d(h6, [2, 2], [2, 2], name='s_pool2')
            #4x4xself.df_dim*4
            h7 = lrelu(instance_norm(conv2d(pool3, self.df_dim*4, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h7_conv'), scope="s_h7_IN"))

            # 4x4x256
            dh1_out_shape = tf.stack([tf.shape(image)[0], tf.to_int32(tf.shape(image)[1]/4), tf.to_int32(tf.shape(image)[2]/4), self.df_dim*2])
            dh1 = lrelu(instance_norm(tf.nn.dropout(deconv2d(h7,output_shape=dh1_out_shape, name='s_dh1', filters=self.df_dim*2), 0.5), scope="s_dh1_IN"))
            dh1 = tf.concat([dh1, pool2], 3)
            # 8x8x128 x2
            dh2_out_shape = tf.stack([tf.shape(image)[0], tf.to_int32(tf.shape(image)[1]/2), tf.to_int32(tf.shape(image)[2]/2), self.df_dim])
            dh2 = lrelu(instance_norm(deconv2d(dh1,output_shape=dh2_out_shape, name='s_dh2', filters=self.df_dim),scope="s_dh2_IN"))
            dh2 = tf.concat([dh2, pool1], 3)
            # 16x16x64 x2
            dh3_out_shape = tf.stack([tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2], 1])
            dh3 = deconv2d(dh2,output_shape=dh3_out_shape, name='s_dh3', filters=1)
            # 32x32x1

            return tf.nn.sigmoid(dh3), dh3, params['entropy']

    archs['arch18'] = arch18

    def arch19(self, image, params, y=None, reuse=False, is_training=True, var_scope="sim_disc"):
        with tf.variable_scope(var_scope) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            h = ((image+1)/2)*0.79375
            # 32x32x6
            h0 = lrelu(conv2d(h, self.df_dim, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h0_conv'))
            h1 = lrelu(conv2d(h0, self.df_dim, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h1_conv'))
            pool1 = max_pooling2d(h1, [2, 2], [2, 2], name='s_pool1')
            h2 = lrelu(conv2d(pool1, self.df_dim*2, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h2_conv'))
            h3 = lrelu(conv2d(h2, self.df_dim*2, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h3_conv'))
            pool2 = max_pooling2d(h3, [2, 2], [2, 2], name='s_pool2')
            h4 = lrelu(conv2d(pool2, self.df_dim*4, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h4_conv'))
            h5 = lrelu(conv2d(h4, self.df_dim*4, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h5_conv'))

            # 8x8x128
            dh2_out_shape = tf.stack([tf.shape(image)[0], tf.to_int32(tf.shape(image)[1]/2), tf.to_int32(tf.shape(image)[2]/2), self.df_dim])
            dh2 = lrelu(deconv2d(h5,output_shape=dh2_out_shape, name='s_dh2', filters=self.df_dim))
            # 16x16x64
            dh3_out_shape = tf.stack([tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2], 1])
            dh2 = tf.concat([dh2, h3], 3)
            dh3 = deconv2d(dh2,output_shape=dh3_out_shape, name='s_dh3', filters=1)
            # 32x32x1

            return tf.nn.sigmoid(dh3), dh3, params['entropy']

    archs['arch19'] = arch19


    def __init__(self, df_dim=64, batch_momentum=0.9, arch='arch1', archs=archs):
        """
        Args:
            df_dim: Number of filters in the first layer. Doubled with each following layer.
            momentum: Parameter for momentum in batch normalization.
            arch: Name of architecture to be used
            batch_size: Number of samples per batch
        """
        self.df_dim = df_dim
        self.batch_momentum = batch_momentum
        self.arch = arch
        self.archs = archs


    def get_output(self, image, reuse=False, is_training=True, bn=False, bs=64):
        params = {'activation': tf.nn.relu, 'padding': 'same',
                  'batch_normalization': bn, 'entropy': 'sigmoid', 'batch_size': bs}
        return self.archs[self.arch](self, image, reuse=reuse, is_training=is_training, params=params)
