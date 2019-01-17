from __future__ import division
import os
import time
from sys import stdout
from glob import glob
import tensorflow as tf
import numpy as np
import cv2
from six.moves import xrange
import sys

from xview.models.vgg16 import vgg16
from xview.models.cycleGAN_ops import *

def add_noise(image, noise=0.1):
    with tf.name_scope("add_noise"):
        return image+tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=noise, dtype=tf.float32)

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 255] => [-1, 1]
        return image/255 * 2 - 1
def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 255]
        return ((image + 1) / 2)*255

class cycleGAN(object):
    def __init__(self, sess, image_size=256, capacity=50, batch_size=1,
                 dataset_name='cityscapes_GAN',
                 checkpoint_dir=None, data=None, data_desc=None,
                 checkpoint=None):
        """
        https://colab.research.google.com/drive/1Enc-pKlP4Q3cimEBfcQv0B_6hUvjVL3o?sandboxMode=true#scrollTo=p7xUHe93Xq61
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            capacity: How many fake generations to keep around. [50]
            dataset_name: Name of dataset to be loaded and trained on.
            checkpoint_dir: Path to directory where checkpoint will be saved
            checkpoint: Path to directory where a checkpoint will be loaded from
            data: Object that delivers the datasets.
            data_desc: Received the shape of data, to build the model/graph.
        """
        self.sess = sess
        self.batch_size = batch_size
        self.image_size = image_size


        # Storage for fake generations
        self.capacity = capacity
        self.fake_As = capacity * [np.zeros((1, self.image_size, self.image_size, 3), dtype=np.float32)]
        self.fake_Bs = capacity * [np.zeros((1, self.image_size, self.image_size, 3), dtype=np.float32)]

        self.data = data
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_loaded = False

        # Get the data descriptors with the shape of data coming
        data_description = data_desc
        data_description = [data_description[0], {
            key: [None, *description]
            for key, description in data_description[1].items()}]

        # Create an iterator for the data
        self.iter_handle = tf.placeholder(tf.string, shape=[],
                                              name='training_placeholder')
        iterator = tf.data.Iterator.from_string_handle(
            self.iter_handle, *data_description)
        training_batch = iterator.get_next()

        self.build_model(training_batch['labels'], training_batch['rgb'])

        if checkpoint is not None and False:
            self.load(checkpoint)
            self.checkpoint_loaded = True

    def build_model(self, input, target):
        # RGB Target
        self.real_B = preprocess(target)
        # SemSeg Label image
        self.real_A = preprocess(input)

        self.fake_A = generator(self.real_B, scope='G_ba')
        self.fake_B = generator(self.real_A, scope='G_ab')

        self.cycle_A = generator(self.fake_B, scope='G_ba', reuse=True)
        self.cycle_B = generator(self.fake_A, scope='G_ab', reuse=True)

        self.D_A_real = discriminator(self.real_A, scope='D_A')
        self.D_B_real = discriminator(self.real_B, scope='D_B')
        self.D_A_fake = discriminator(self.fake_A, scope='D_A', reuse=True)
        self.D_B_fake = discriminator(self.fake_B, scope='D_B', reuse=True)

        l1 = 10.0
        loss_cycle = tf.reduce_mean(l1 * tf.abs(self.real_A - self.cycle_A)) + \
                     tf.reduce_mean(l1 * tf.abs(self.real_B - self.cycle_B))
        loss_G_ab = tf.reduce_mean(tf.square(self.D_B_fake - 1.0)) + loss_cycle
        loss_G_ba = tf.reduce_mean(tf.square(self.D_A_fake - 1.0)) + loss_cycle
        self.loss_G = loss_G_ab + loss_G_ba

        self.fake_A_sample = tf.placeholder(name='fake_A_sample',
                shape=[None, self.image_size, self.image_size, 3], dtype=tf.float32)
        self.fake_B_sample = tf.placeholder(name='fake_B_sample',
                shape=[None, self.image_size, self.image_size, 3], dtype=tf.float32)

        self.D_A_fake_sample = discriminator(self.fake_A_sample, scope='D_A', reuse=True)
        self.D_B_fake_sample = discriminator(self.fake_B_sample, scope='D_B', reuse=True)
        #TODO /2 or not?
        self.loss_D_B = (tf.reduce_mean(tf.square(self.D_B_real - 0.9)) + \
                    tf.reduce_mean(tf.square(self.D_B_fake_sample))) / 2
        self.loss_D_A = (tf.reduce_mean(tf.square(self.D_A_real - 0.9)) + \
                    tf.reduce_mean(tf.square(self.D_A_fake_sample))) / 2

        self.dAr_sum = tf.summary.histogram("D_A_real", self.D_A_real)
        self.dBr_sum = tf.summary.histogram("D_B_real", self.D_B_real)
        self.dAf_sum = tf.summary.histogram("D_A_fake", self.D_A_fake)
        self.dBf_sum = tf.summary.histogram("D_B_fake", self.D_B_fake)

        self.real_A_sum = tf.summary.image("Input", deprocess(self.real_A)[...,::-1])
        self.real_B_sum = tf.summary.image("Target", deprocess(self.real_B)[...,::-1])
        self.fake_A_sum = tf.summary.image("Fake_Input", deprocess(self.fake_A)[...,::-1])
        self.fake_B_sum = tf.summary.image("Fake_Target", deprocess(self.fake_B)[...,::-1])
        self.cycle_A_sum = tf.summary.image("Cycle_Input", deprocess(self.cycle_A)[...,::-1])
        self.cycle_B_sum = tf.summary.image("Cycle_Target", deprocess(self.cycle_B)[...,::-1])

        self.d_A_loss_sum = tf.summary.scalar("d_A_loss", self.loss_D_A)
        self.d_B_loss_sum = tf.summary.scalar("d_B_loss", self.loss_D_B)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.loss_G)


        t_vars = tf.trainable_variables()

        self.D_A_vars = [var for var in t_vars if var.name.startswith('D_A')]
        self.D_B_vars = [var for var in t_vars if var.name.startswith('D_B')]
        G_ab_vars = [var for var in t_vars if var.name.startswith('G_ab')]
        G_ba_vars = [var for var in t_vars if var.name.startswith('G_ba')]
        self.G_vars = G_ab_vars + G_ba_vars

        self.saver = tf.train.Saver()

    def train(self, args):
        """Train cycleGAN"""
        #TODO check if these two lines are needed/improve results
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):

        d_A_optim = tf.train.AdamOptimizer(args.lr,args.beta1).minimize(
                self.loss_D_A, var_list=self.D_A_vars)
        d_B_optim = tf.train.AdamOptimizer(args.lr,args.beta1).minimize(
                self.loss_D_B, var_list=self.D_B_vars)
        g_optim = tf.train.AdamOptimizer(args.lr,args.beta1).minimize(
                self.loss_G, var_list=self.G_vars)

        train_op = tf.group(g_optim, d_B_optim, d_A_optim)
        init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
        self.sess.run(init_op)

        if args.checkpoint is not None and self.load(os.path.join(args.EXP_OUT,str(args.checkpoint))):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")



        self.sum = tf.summary.merge([self.dAr_sum,self.dBr_sum,self.dAf_sum,self.dBf_sum,self.real_A_sum,
                                     self.real_B_sum,self.fake_A_sum, self.fake_B_sum,self.cycle_A_sum,self.cycle_B_sum,
                                     self.d_A_loss_sum, self.d_B_loss_sum, self.g_loss_sum])

        self.writer = tf.summary.FileWriter(self.checkpoint_dir, self.sess.graph)

        input_data = self.data.get_trainset()
        data_iterator = input_data.repeat(args.max_epochs).batch(args.batch_size).make_one_shot_iterator()
        data_handle = self.sess.run(data_iterator.string_handle())

        train_step = 0
        idx = 0
        start_time = time.time()

        while True:
            try:
                if train_step == 0:
                    A_fake, B_fake = self.sess.run([self.fake_A,self.fake_B],
                                              feed_dict={ self.iter_handle: data_handle })

                if train_step < self.capacity:
                    self.fake_As[idx] = A_fake
                    self.fake_Bs[idx] = B_fake
                    idx = (idx+1)%self.capacity
                elif np.random.random() > 0.5:
                    rand_idx = np.random.randint(0,self.capacity)
                    self.fake_As[rand_idx], A_fake = A_fake, self.fake_As[rand_idx]
                    self.fake_Bs[rand_idx], B_fake = B_fake, self.fake_Bs[rand_idx]
                else:
                    pass

                _, dAl, dBl, gl, A_fake_new, B_fake_new  = self.sess.run([train_op,self.loss_D_A,self.loss_D_B,self.loss_G, self.fake_A,self.fake_B],
                                                      feed_dict={ self.iter_handle: data_handle,
                                                                self.fake_A_sample: A_fake,
                                                                self.fake_B_sample: B_fake})

            except tf.errors.OutOfRangeError:
                print("INFO: Done with all steps")
                self.save(self.checkpoint_dir, train_step)
                break

            if train_step % args.num_print == 0:
                print("Step: [%2d] rate: %4.4f steps/sec, dA_loss: %.8f, dB_loss: %.8f, g_loss: %.8f" \
                    % (train_step,args.num_print/(time.time() - start_time),dAl,dBl,gl))
                stdout.flush()

                summary = self.sess.run(self.sum,
                                  feed_dict={ self.iter_handle: data_handle,
                                            self.fake_A_sample: A_fake,
                                            self.fake_B_sample: B_fake})

                self.writer.add_summary(summary, train_step)
                start_time = time.time()

            A_fake = A_fake_new
            B_fake = B_fake_new
            train_step += 1

    def validate(self, args):
        """Validate cycleGAN"""
        pred_array = np.zeros((15,2))
        counter = 1

        if not self.checkpoint_loaded:
            self.load(os.path.join(args.EXP_OUT,str(args.checkpoint)))

        if not os.path.exists(os.path.join(args.file_output_dir,str(args.checkpoint))):
            os.makedirs(os.path.join(args.file_output_dir,str(args.checkpoint)))

        validation_data = self.data.get_validation_set()
        valid_iterator = validation_data.batch(args.batch_size).make_one_shot_iterator()
        valid_handle = self.sess.run(valid_iterator.string_handle())

        while True:
            # Retrieve everything you want out of the graph
            try:
                outImage, inpt, target, real_val, fake_val = self.sess.run([self.fake_B,self.real_A,self.real_B,self.D,self.D_],
                                               feed_dict={ self.iter_handle: valid_handle })
            # When tf dataset is empty this error is thrown
            except tf.errors.OutOfRangeError:
                print("INFO: Done with all steps")
                break
            pred_array[counter-1,:] = [np.mean(real_val[0]),np.mean(fake_val[0])]

            # Save the 30 x 30 output of the discriminator
            filename = str(args.checkpoint)+"_realfield" + str(counter) + ".png"
            cv2.imwrite(os.path.join(args.file_output_dir,str(args.checkpoint),filename), cv2.resize(255*real_val[0,:,:,0],(args.input_image_size,args.input_image_size),interpolation=cv2.INTER_NEAREST))
            filename = str(args.checkpoint)+"_fakefield" + str(counter) + ".png"
            cv2.imwrite(os.path.join(args.file_output_dir,str(args.checkpoint),filename), cv2.resize(255*fake_val[0,:,:,0],(args.input_image_size,args.input_image_size),interpolation=cv2.INTER_NEAREST))
            # Save the output of the generator
            filename = str(args.checkpoint)+"_validation" + str(counter) + ".png"
            cv2.imwrite(os.path.join(args.file_output_dir,str(args.checkpoint),filename), deprocess(outImage[0,:,:,:]), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            # If needed, save the input and target
            if args.val_target_output == True:
                filename = "input_validation" + str(counter) + ".png"
                cv2.imwrite(os.path.join(args.file_output_dir,filename), deprocess(inpt[0,:,:,:]), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                filename = "target_validation" + str(counter) + ".png"
                cv2.imwrite(os.path.join(args.file_output_dir,filename), deprocess(target[0,:,:,:]), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            counter += 1
        print(pred_array)
        return pred_array

    def transform(self, args, images):
        """Transforms dataset from single images
        This is used for pipeline usage.
        """
        # Check that a checkpoint was loaded at init
        assert(self.checkpoint_loaded is not False)

        synth = np.zeros((images.shape))

        data = {'rgb': tf.zeros_like(images, dtype=tf.float32), 'labels': tf.to_float(images)}

        iterator = tf.data.Dataset.from_tensor_slices(data)\
                   .batch(1).make_one_shot_iterator()
        handle = self.sess.run(iterator.string_handle())

        for i in range(images.shape[0]):
            outImage = self.sess.run([self.fake_B],
                                           feed_dict={ self.iter_handle: handle })

            synth[i,:,:,:] = deprocess(outImage[0])

        return synth

    def transformDatasets(self, args):
        """Transforms complete dataset
        """
        # Check that a checkpoint was loaded at init
        assert(self.checkpoint_loaded is not False)
        set_handles = {}
        subsets = ['validation','measure','training']

        validation_data = self.data.get_validation_set()
        valid_iterator = validation_data.batch(1).make_one_shot_iterator()
        set_handles['validation'] = self.sess.run(valid_iterator.string_handle())

        measure_data = self.data.get_measureset()
        measure_iterator = measure_data.batch(1).make_one_shot_iterator()
        set_handles['measure'] = self.sess.run(measure_iterator.string_handle())

        training_data = self.data.get_trainset()
        training_iterator = training_data.batch(1).make_one_shot_iterator()
        set_handles['training'] = self.sess.run(training_iterator.string_handle())

        head_folder = os.path.join(args.file_output_dir,str(args.checkpoint)+"_full")
        if not os.path.exists(head_folder):
            os.makedirs(head_folder)

        for sets in subsets:
            counter = 1
            local_folder = os.path.join(head_folder,sets)
            if not os.path.exists(local_folder):
                os.makedirs(local_folder)
            while True:
                # Retrieve everything you want from of the graph
                try:
                    outImage, inpt, target = self.sess.run([self.fake_B,self.real_A,self.real_B],
                                                   feed_dict={ self.iter_handle: set_handles[sets] })
                # When tf dataset is empty this error is thrown
                except tf.errors.OutOfRangeError:
                    print("INFO: Done with "+sets+" set")
                    break

                # Save the output of the generator
                filename = str(args.checkpoint)+"_"+ sets + str(counter) + ".png"
                cv2.imwrite(os.path.join(local_folder,filename), deprocess(outImage[0,:,:,:]), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                filename = "input_" + sets + str(counter) + ".png"
                cv2.imwrite(os.path.join(local_folder,filename), deprocess(inpt[0,:,:,:]), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                filename = "target_"+ sets + str(counter) + ".png"
                cv2.imwrite(os.path.join(local_folder,filename), deprocess(target[0,:,:,:]), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                counter +=1


    def save(self, checkpoint_dir, step):
        model_name = "cycleGAN.model"
        # model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        # checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)

        self.saver.save(self.sess,
                        os.path.join(self.checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        self.saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint_dir))
        self.graph = tf.get_default_graph()
        # uninitialized_vars = [var for var in (self.D_A_vars + self.D_B_vars + self.G_vars) if not self.sess.run(tf.is_variable_initialized(var))]
        # self.sess.run(tf.variables_initializer(var_list=uninitialized_vars))

        return True
