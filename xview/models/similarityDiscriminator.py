import math
import numpy as np
import tensorflow as tf
import glob
import sys
from sys import stdout
import os
import time
import cv2
from PIL import Image

from tensorflow.python.framework import ops

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 255] => [-1, 1]
        return image/255 * 2 - 1
def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 255]
        return ((image + 1) / 2)*255
class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        # return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, scope=self.name)
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

class DiffDiscrim(object):
    def __init__(self, sess, image_size=256,
                 batch_size=64, df_dim=64, ppd=8,
                 input_c_dim=3, is_training=True,
                 checkpoint_dir=None, data=None, momentum=0.9, checkpoint=None):
        """
        Args:
            sess: TensorFlow session
            image_size: Width and height of image. Should be square image.
            batch_size: The size of batch. Should be specified before training.
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            is_training: Flag for batch normalization to know
            df_dim: Number of filters in the first layer. Doubled with each following layer.
            checkpoint_dir: Directory where the checkpoint will be saved.
            data: Data object, used to get the shape of data and called to return datasets.
            momentum: Parametre for momentum in batch normalization.
            checkpoint: Directory where the current checkpoint is that will be loaded
        """
        self.sess = sess
        self.ppd = 8
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size

        self.input_c_dim = input_c_dim
        self.df_dim = df_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.s_bn1 = batch_norm(name='s_bn1', momentum=momentum)
        self.s_bn2 = batch_norm(name='s_bn2', momentum=momentum)
        #self.d_bn3 = batch_norm(name='s_bn3', momentum=momentum)


        self.data = data
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_loaded = False

        data_description = data.get_data_description()
        # Create an iterator for the data
        data_description = data.get_data_description()
        data_description = [data_description[0], {
            key: [None, *description]
            for key, description in data_description[1].items()}]

        self.iter_handle = tf.placeholder(tf.string, shape=[],
                                              name='training_placeholder')
        iterator = tf.data.Iterator.from_string_handle(
            self.iter_handle, *data_description)
        training_batch = iterator.get_next()

        self.build_model(training_batch['labels'],training_batch['pos'],training_batch['neg'],training_batch['pos_segm'],training_batch['neg_segm'])

        if checkpoint is not None and not(checkpoint.split('/')[-1] == "None"):
            self.load(checkpoint)
            self.checkpoint_loaded = True
            if not is_training:
                assign_flag_op = self.train_flag.assign(False)
                self.sess.run(assign_flag_op)

    def build_model(self, target, pos, neg, pos_segm, neg_segm):
        self.target_placeholder = preprocess(target)
        self.pos_placeholder = preprocess(pos)
        self.neg_placeholder = preprocess(neg)
        self.pos_segm_placeholder = preprocess(pos_segm)
        self.neg_segm_placeholder = preprocess(neg_segm)
        self.train_flag = tf.Variable(True, name="Train_flag")

        # PosExample = tf.concat([self.target_placeholder, self.pos_placeholder, self.pos_segm_placeholder], 3)
        # NegExample = tf.concat([self.target_placeholder, self.neg_placeholder, self.neg_segm_placeholder], 3)

        PosExample = tf.concat([self.target_placeholder, self.pos_placeholder], 3)
        NegExample = tf.concat([self.target_placeholder, self.neg_placeholder], 3)

        self.D, self.D_logits = self.discriminator(PosExample)
        self.D_, self.D_logits_ = self.discriminator(NegExample,reuse=True)

        self.d_loss_pos = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.zeros_like(self.D)))
        self.d_loss_neg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        self.d_loss = self.d_loss_pos + self.d_loss_neg
        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.target_sum = tf.summary.image("Target", target[...,::-1])
        self.positiv_sum = tf.summary.image("Positiv", pos[...,::-1])
        self.negativ_sum = tf.summary.image("Negativ", neg[...,::-1])

        self.d_loss_pos_sum = tf.summary.scalar("d_loss_pos", self.d_loss_pos)
        self.d_loss_neg_sum = tf.summary.scalar("d_loss_neg", self.d_loss_neg)

        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 's_' in var.name]

        self.saver = tf.train.Saver()


    def train(self, args):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            d_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                              .minimize(self.d_loss, var_list=self.d_vars)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.sum = tf.summary.merge([self.d__sum,self.target_sum,self.positiv_sum,
            self.negativ_sum, self.d_loss_pos_sum, self.d_sum, self.d_loss_neg_sum, self.d_loss_sum])

        self.writer = tf.summary.FileWriter(self.checkpoint_dir, self.sess.graph)
        if not self.checkpoint_loaded:
            if args.checkpoint is not None and self.load(os.path.join(args.EXP_OUT,str(args.checkpoint))):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        input_data = self.data.get_trainset()
        data_iterator = input_data.repeat(args.max_epochs).batch(args.batch_size).make_one_shot_iterator()
        data_handle = self.sess.run(data_iterator.string_handle())
        globalCounter = 1
        localCounter = 1
        start_time = time.time()

        while True:
            if np.mod(globalCounter, args.num_print) == 1:
                try:
                    _, summary_str, d_l = self.sess.run([d_optim, self.sum, self.d_loss],
                                                   feed_dict={ self.iter_handle: data_handle })

                except tf.errors.OutOfRangeError:
                    print("INFO: Done with all steps")
                    self.save(self.checkpoint_dir, globalCounter, args.DATA_id)
                    break

                self.writer.add_summary(summary_str, globalCounter-1)
                print("Step: [%2d] rate: %4.4f steps/sec, d_loss: %.8f" \
                    % (globalCounter,localCounter/(time.time() - start_time), np.mean(d_l)))
                stdout.flush()
                tmp = self.validation(args, out=True, loaded=True)
                mean_val_D, mean_val_D_ = np.mean(tmp, axis=0)
                absErr = mean_val_D+1-mean_val_D_
                print("Mean Validation: Same: %f \t Diff: %f \t Abs: %f" % (mean_val_D,mean_val_D_,absErr))

                abs_err = tf.Summary(value=[tf.Summary.Value(tag='Absolute Validation Error',
                                            simple_value=absErr)])
                self.writer.add_summary(abs_err, globalCounter)
                stdout.flush()
                start_time = time.time()
                localCounter = 1
            else:
                try:
                    self.sess.run(d_optim,feed_dict={ self.iter_handle: data_handle })
                except tf.errors.OutOfRangeError:
                    print("INFO: Done with all training steps")
                    self.save(self.checkpoint_dir, globalCounter, args.DATA_id)
                    break
            globalCounter += 1
            localCounter += 1

        return self.validation(args, loaded=True)

    def validation(self, args, out=False, loaded=False):
        if not loaded:
            if args.checkpoint is not None and self.load(os.path.join(args.EXP_OUT,str(args.checkpoint))):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
                raise ValueError('Could not load checkpoint and that is needed for validation')
        input_data, num_validation = self.data.get_validation_set()
        data_iterator = input_data.repeat(1).batch(args.batch_size).make_one_shot_iterator()
        data_handle = self.sess.run(data_iterator.string_handle())
        counter = 1
        start_time = time.time()
        pred_array = np.zeros((int(num_validation/args.batch_size),2))
        while(True):
            try:
                D, D_ = self.sess.run([self.D, self.D_],
                                               feed_dict={ self.iter_handle: data_handle })

                pred_array[counter-1,:] = [np.mean(D),np.mean(D_)]
                if not out:
                    print("Validation image %d: Same: %f \t Diff: %f" % (counter, np.mean(D),np.mean(D_)))
                    stdout.flush()
            except tf.errors.OutOfRangeError:
                break

            counter += 1
        return pred_array

    def predict(self, args, inputImage, ganImage, segmImage):
        """ Predict similarity between images given in lists """
        dx_h = int(args.input_image_size/self.ppd)
        dx_w = int(args.input_image_size/self.ppd)
        pred_array = np.zeros((len(inputImage),2))

        # Check that a checkpoint directory is given, to load from
        if not self.checkpoint_loaded:
            assert(args.checkpoint is not None)
            self.load(os.path.join(args.EXP_OUT,str(args.checkpoint)))

        if not os.path.exists(os.path.join(args.file_output_dir,str(args.checkpoint))):
            os.makedirs(os.path.join(args.file_output_dir,str(args.checkpoint)))

        for k,image_path in enumerate(inputImage):
            input = np.expand_dims(cv2.imread(image_path), axis=0)
            synth = np.expand_dims(cv2.imread(ganImage[k]), axis=0)
            segm = np.expand_dims(cv2.imread(segmImage[k]), axis=0)

            input_patch = []
            synth_patch = []
            segm_patch = []

            output_image = np.zeros((self.ppd,self.ppd))

            for j in range(self.ppd):
                for i in range(self.ppd):
                    if (i<1 and j<1):
                        input_patch = input[:,j*dx_h:(j+1)*dx_h,i*dx_w:(i+1)*dx_w,:]
                        synth_patch = synth[:,j*dx_h:(j+1)*dx_h,i*dx_w:(i+1)*dx_w,:]
                        segm_patch = segm[:,j*dx_h:(j+1)*dx_h,i*dx_w:(i+1)*dx_w,:]
                    else:
                        input_patch=np.concatenate((input_patch,input[:,j*dx_h:(j+1)*dx_h,i*dx_w:(i+1)*dx_w,:]),axis=0)
                        synth_patch=np.concatenate((synth_patch,synth[:,j*dx_h:(j+1)*dx_h,i*dx_w:(i+1)*dx_w,:]),axis=0)
                        segm_patch=np.concatenate((segm_patch,segm[:,j*dx_h:(j+1)*dx_h,i*dx_w:(i+1)*dx_w,:]),axis=0)

            data = {'labels': tf.to_float(input_patch), 'pos': tf.to_float(synth_patch), 'neg': tf.zeros_like(tf.to_float(synth_patch)),'pos_segm': tf.to_float(segm_patch), 'neg_segm': tf.zeros_like(tf.to_float(segm_patch)) }

            iterator = tf.data.Dataset.from_tensor_slices(data)\
                       .batch(self.ppd*self.ppd).make_one_shot_iterator()
            handle = self.sess.run(iterator.string_handle())

            output = self.sess.run(self.D, feed_dict={self.iter_handle: handle})
            output = output[:,0,0,0].reshape((self.ppd,self.ppd))
            pred_array[k,:] = [k+1, np.mean(output)]

            if (k+1) == 1:
                output_matrix = np.expand_dims(output, axis=0)
            else:
                output_matrix = np.concatenate((output_matrix, np.expand_dims(output, axis=0)),axis=0)

            filename = "simGrid_"+str(k+1)+".png"
            cv2.imwrite(os.path.join(args.file_output_dir,str(args.checkpoint),filename), cv2.resize(255*output,(args.input_image_size,args.input_image_size),interpolation=cv2.INTER_NEAREST))

        matrix_path = os.path.join(args.file_output_dir,str(args.checkpoint),"mat.npy")
        np.save(matrix_path, output_matrix)

        txt_path = os.path.join(args.file_output_dir,str(args.checkpoint),"pred.txt")
        text_file = open(txt_path,'w')
        for i in range(len(inputImage)):
            print("%d. \t %f" % (pred_array[i,0],pred_array[i,1]))
            text_file.write("%d. \t %f \n" % (pred_array[i,0],pred_array[i,1]))
        text_file.close()

    def transform(self, realImages, synthImages, segmImages):
        """ Predict similarity between images """
        counter = 1
        dx_h = int(realImages.shape[1]/self.ppd)
        dx_w = int(realImages.shape[1]/self.ppd)

        # Check that a checkpoint directory is given, to load from
        if not self.checkpoint_loaded:
            assert(args.checkpoint is not None)
            self.load(os.path.join(args.EXP_OUT,str(args.checkpoint)))

        for k in range(realImages.shape[0]):
            input = np.expand_dims(realImages[k,:,:,:],axis=0)
            synth = np.expand_dims(synthImages[k,:,:,:],axis=0)
            segm = np.expand_dims(segmImages[k,:,:,:],axis=0)

            input_patch = []
            synth_patch = []
            segm_patch = []

            for j in range(self.ppd):
                for i in range(self.ppd):
                    if (i<1 and j<1):
                        input_patch = input[:,j*dx_h:(j+1)*dx_h,i*dx_w:(i+1)*dx_w,:]
                        synth_patch = synth[:,j*dx_h:(j+1)*dx_h,i*dx_w:(i+1)*dx_w,:]
                        segm_patch = segm[:,j*dx_h:(j+1)*dx_h,i*dx_w:(i+1)*dx_w,:]
                    else:
                        input_patch=np.concatenate((input_patch,input[:,j*dx_h:(j+1)*dx_h,i*dx_w:(i+1)*dx_w,:]),axis=0)
                        synth_patch=np.concatenate((synth_patch,synth[:,j*dx_h:(j+1)*dx_h,i*dx_w:(i+1)*dx_w,:]),axis=0)
                        segm_patch=np.concatenate((segm_patch,segm[:,j*dx_h:(j+1)*dx_h,i*dx_w:(i+1)*dx_w,:]),axis=0)

            data = {'labels': tf.to_float(input_patch), 'pos': tf.to_float(synth_patch), 'neg': tf.zeros_like(tf.to_float(synth_patch)),'pos_segm': tf.to_float(segm_patch), 'neg_segm': tf.zeros_like(tf.to_float(segm_patch)) }

            iterator = tf.data.Dataset.from_tensor_slices(data)\
                       .batch(self.ppd*self.ppd).make_one_shot_iterator()
            handle = self.sess.run(iterator.string_handle())

            output = self.sess.run(self.D, feed_dict={self.iter_handle: handle})
            output = output[:,0,0,0].reshape((self.ppd,self.ppd))

            if counter == 1:
                output_matrix = np.expand_dims(output, axis=0)
            else:
                output_matrix = np.concatenate((output_matrix, np.expand_dims(output, axis=0)),axis=0)

            counter += 1
        return output_matrix



    def discriminator(self, image, y=None, reuse=False):
        # image is 256 x 256 x (input_c_dim + input_c_dim)
        with tf.variable_scope("sim_disc") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, k_h=4, k_w=4, d_h=4, d_w=4, name='s_h0_conv'))
            # h0 is (64 x 64 x self.df_dim)
            h1 = lrelu(self.s_bn1(conv2d(h0, self.df_dim*2, k_h=4, k_w=4, d_h=4, d_w=4, name='s_h1_conv'),train=self.train_flag))
            # h1 is (16 x 16 x self.df_dim*2)
            h2 = conv2d(h1, 1, k_h=2, k_w=2, d_h=2, d_w=2, name='s_h2_conv')
            # h2 is (8 x 8 x 1)

            return tf.nn.sigmoid(h2), h2

    def discriminator2(self, image, y=None, reuse=False):
        # image is 256 x 256 x (input_c_dim + input_c_dim)
        with tf.variable_scope("sim_disc") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, k_h=1, k_w=1, d_h=1, d_w=1, name='s_h0_conv'))
            # h0 is (256 x 256 x self.df_dim)
            h1 = lrelu(self.s_bn1(conv2d(h0, self.df_dim*2, k_h=4, k_w=4, d_h=4, d_w=4, name='s_h1_conv'),train=self.train_flag))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = maxPool2d(input_=h1, k_h=2, k_w=2, d_h=2, d_w=2, name="s_h2_maxpool2d")
            # h2 is (32 x 32 x self.df_dim*2)
            h3 = lrelu(self.s_bn2(conv2d(h2, self.df_dim*4, k_h=2, k_w=2, d_h=2, d_w=2, name='s_h3_conv'),train=self.train_flag))
            # h3 is (16 x 16 x self.df_dim*4)
            h4 = conv2d(h3, 1, k_h=2, k_w=2, d_h=2, d_w=2, name='s_h4_conv')
            # h4 is (8 x 8 x 1)

            return tf.nn.sigmoid(h4), h4

    def discriminator3(self, image, y=None, reuse=False):
        # image is 256 x 256 x (input_c_dim + input_c_dim)
        with tf.variable_scope("sim_disc") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, k_h=3, k_w=3, d_h=1, d_w=1, name='s_h0_conv'))
            # h0 is (256 x 256 x self.df_dim)
            h1 = lrelu(self.s_bn1(conv2d(h0, self.df_dim*2, k_h=4, k_w=4, d_h=2, d_w=2, name='s_h1_conv'),train=self.train_flag))
            # h1 is (128 x 128 x self.df_dim*2)
            h2 = lrelu(self.s_bn2(conv2d(h1, self.df_dim*4, k_h=4, k_w=4, d_h=4, d_w=4, name='s_h2_conv'),train=self.train_flag))
            # h2 is (32 x 32 x self.df_dim*4)
            h3 = maxPool2d(input_=h2, k_h=2, k_w=2, d_h=2, d_w=2, name="s_h3_maxpool2d")
            # h3 is (16 x 16 x self.df_dim*4)
            h4 = conv2d(h3, 1, k_h=2, k_w=2, d_h=2, d_w=2, name='s_h4_conv')
            # h4 is (8 x 8 x 1)

            return tf.nn.sigmoid(h4), h4

    def save(self, checkpoint_dir, step, id):
        model_name = "diffDiscrim"+id+".model"
        self.saver.save(self.sess,
                        os.path.join(self.checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        self.saver.restore(self.sess, checkpoint)
        self.graph = tf.get_default_graph()
        return True
