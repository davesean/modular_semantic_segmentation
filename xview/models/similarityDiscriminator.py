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
from xview.models.similarityArchitectures import simArch

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 255] => [-1, 1]
        return image/255 * 2 - 1
# def deprocess(image):
#     with tf.name_scope("deprocess"):
#         # [-1, 1] => [0, 255]
#         return ((image + 1) / 2)*255

class DiffDiscrim(object):
    def __init__(self, sess, image_size=256, seed=42,
                 batch_size=64, df_dim=64, ppd=8,
                 input_c_dim=3, is_training=True, arch='arch1', batch_norm=False,
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
            momentum: Parameter for momentum in batch normalization.
            checkpoint: Directory where the current checkpoint is that will be loaded
            arch: define: which architecture should be used
            batch_norm: Define if batch normalization should be used in conv2d
        """
        self.sess = sess
        self.seed = seed
        self.ppd = 8
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.batch_norm = batch_norm

        self.input_c_dim = input_c_dim
        self.df_dim = df_dim

        self.batch_momentum = momentum
        self.archDisc = simArch(df_dim=df_dim, arch=arch)

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

    def build_model(self, target, pos, neg, pos_segm, neg_segm):
        tf.set_random_seed(self.seed)
        self.target_placeholder = preprocess(target)
        self.pos_placeholder = preprocess(pos)
        self.neg_placeholder = preprocess(neg)
        self.pos_segm_placeholder = preprocess(pos_segm)
        self.neg_segm_placeholder = preprocess(neg_segm)
        self.train_flag = tf.placeholder(tf.bool, name="Train_flag")

        # PosExample = tf.concat([self.target_placeholder, self.pos_placeholder, self.pos_segm_placeholder], 3)
        # NegExample = tf.concat([self.target_placeholder, self.neg_placeholder, self.neg_segm_placeholder], 3)

        PosExample = tf.concat([self.target_placeholder, self.pos_placeholder], 3)
        NegExample = tf.concat([self.target_placeholder, self.neg_placeholder], 3)

        # self.D, self.D_logits = self.discriminator(PosExample)
        # self.D_, self.D_logits_ = self.discriminator(NegExample,reuse=True)

        self.D, self.D_logits = self.archDisc.get_output(image=PosExample,
                                                         is_training=self.train_flag,
                                                         bn=self.batch_norm)
        self.D_, self.D_logits_ = self.archDisc.get_output(image=NegExample,
                                                           reuse=True,
                                                           is_training=self.train_flag,
                                                           bn=self.batch_norm)

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
                                                   feed_dict={ self.iter_handle: data_handle,
                                                               self.train_flag: True})

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
                std_D, std_D_ = np.std(tmp, axis=0)
                absErr = mean_val_D+1-mean_val_D_
                totStd = std_D+std_D_
                print("Mean Validation: Same: %f \t Diff: %f \t Abs: %f \t sum_Std: %f" % (mean_val_D,mean_val_D_,absErr,totStd))

                abs_err = tf.Summary(value=[tf.Summary.Value(tag='Absolute Validation Error',
                                            simple_value=absErr)])
                tot_std = tf.Summary(value=[tf.Summary.Value(tag='Sum Standard Deviations',
                                            simple_value=totStd)])
                self.writer.add_summary(abs_err, globalCounter)
                self.writer.add_summary(tot_std, globalCounter)
                stdout.flush()
                start_time = time.time()
                localCounter = 1
            else:
                try:
                    self.sess.run(d_optim,feed_dict={ self.iter_handle: data_handle,
                                                      self.train_flag: True })
                except tf.errors.OutOfRangeError:
                    print("INFO: Done with all training steps")
                    self.save(self.checkpoint_dir, globalCounter, args.DATA_id)
                    break
            globalCounter += 1
            localCounter += 1
        self.checkpoint_loaded = True
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
                                               feed_dict={ self.iter_handle: data_handle,
                                                           self.train_flag: False })

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

            output = self.sess.run(self.D, feed_dict={self.iter_handle: handle,
                                                      self.train_flag: False })
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
            assert(False, "No checkpoint loaded, load one at init of model.")

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

            output = self.sess.run(self.D, feed_dict={self.iter_handle: handle,
                                                      self.train_flag: True })

            output = np.squeeze(output).reshape((self.ppd,self.ppd))

            if counter == 1:
                output_matrix = np.expand_dims(output, axis=0)
            else:
                output_matrix = np.concatenate((output_matrix, np.expand_dims(output, axis=0)),axis=0)

            counter += 1
        return output_matrix

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
