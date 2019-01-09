import tensorflow as tf

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
        else:
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=pad)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        return conv
def maxPool2d(input_,
           k_h=4, k_w=4, d_h=2, d_w=2,
           name="maxpool2d",pad="SAME"):
           return tf.layers.max_pooling2d(inputs=input_, pool_size=[k_h,k_w], strides=[d_h,d_w], padding=pad)
def dense(input_, output_size, num_channels, name="dense", reuse=False, stddev=0.02, bias_start=0.0):
    shape = 16 * 16 * num_channels
    with tf.variable_scope(name):
        matrix = tf.get_variable("Matrix", [shape, output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        return tf.matmul(tf.layers.flatten(input_), matrix) + bias


class simArch(object):

    def arch1(self, image, y=None, reuse=False, is_training=True):
        # image is 256 x 256 x (input_c_dim + input_c_dim)
        with tf.variable_scope("sim_disc") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, k_h=4, k_w=4, d_h=4, d_w=4, name='s_h0_conv'))
            # h0 is (64 x 64 x self.df_dim)
            h1 = lrelu(self.s_bn1(conv2d(h0, self.df_dim*2, k_h=4, k_w=4, d_h=4, d_w=4, name='s_h1_conv'),train=is_training))
            # h1 is (16 x 16 x self.df_dim*2)
            h2 = conv2d(h1, 1, k_h=2, k_w=2, d_h=2, d_w=2, name='s_h2_conv')
            # h2 is (8 x 8 x 1)

            return tf.nn.sigmoid(h2), h2

    def arch2(self, image, y=None, reuse=False, is_training=True):
        # image is 256 x 256 x (input_c_dim + input_c_dim)
        with tf.variable_scope("sim_disc") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, k_h=1, k_w=1, d_h=1, d_w=1, name='s_h0_conv'))
            # h0 is (256 x 256 x self.df_dim)
            h1 = lrelu(self.s_bn1(conv2d(h0, self.df_dim*2, k_h=4, k_w=4, d_h=4, d_w=4, name='s_h1_conv'),train=is_training))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = maxPool2d(input_=h1, k_h=2, k_w=2, d_h=2, d_w=2, name="s_h2_maxpool2d")
            # h2 is (32 x 32 x self.df_dim*2)
            h3 = lrelu(self.s_bn2(conv2d(h2, self.df_dim*4, k_h=2, k_w=2, d_h=2, d_w=2, name='s_h3_conv'),train=is_training))
            # h3 is (16 x 16 x self.df_dim*4)
            h4 = conv2d(h3, 1, k_h=2, k_w=2, d_h=2, d_w=2, name='s_h4_conv')
            # h4 is (8 x 8 x 1)

            return tf.nn.sigmoid(h4), h4

    def arch3(self, image, y=None, reuse=False, is_training=True):
        # image is 256 x 256 x (input_c_dim + input_c_dim)
        with tf.variable_scope("sim_disc") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h0_conv'))
            # h0 is (256 x 256 x self.df_dim)
            h1 = lrelu(self.s_bn1(conv2d(h0, self.df_dim*2, k_h=4, k_w=4, d_h=2, d_w=2, name='s_h1_conv'),train=is_training))
            # h1 is (128 x 128 x self.df_dim*2)
            h2 = lrelu(self.s_bn2(conv2d(h1, self.df_dim*4, k_h=4, k_w=4, d_h=4, d_w=4, name='s_h2_conv'),train=is_training))
            # h2 is (32 x 32 x self.df_dim*4)
            h3 = maxPool2d(input_=h2, k_h=2, k_w=2, d_h=2, d_w=2, name="s_h3_maxpool2d")
            # h3 is (16 x 16 x self.df_dim*4)
            h4 = conv2d(h3, 1, k_h=2, k_w=2, d_h=2, d_w=2, name='s_h4_conv')
            # h4 is (8 x 8 x 1)

            return tf.nn.sigmoid(h4), h4

    archs = {
        'arch1': arch1,
        'arch2': arch2,
        'arch3': arch3
    }

    def __init__(self, df_dim=64, batch_momentum=0.9, arch='arch1', archs=archs):
        """
        Args:
            df_dim: Number of filters in the first layer. Doubled with each following layer.
            momentum: Parameter for momentum in batch normalization.
            arch: Name of architecture to be used
        """
        self.df_dim = df_dim
        self.batch_momentum = batch_momentum
        self.arch = arch
        self.s_bn1 = batch_norm(name='s_bn1', momentum=batch_momentum)
        self.s_bn2 = batch_norm(name='s_bn2', momentum=batch_momentum)
        self.d_bn3 = batch_norm(name='s_bn3', momentum=batch_momentum)
        self.archs = archs


    def get_output(self, image, reuse=False, is_training=True):
        return self.archs[self.arch](self, image, reuse=reuse, is_training=is_training)
