from __future__ import division
import os
import time
from sys import stdout
from glob import glob
import tensorflow as tf
import numpy as np
import cv2
import sys

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
    def __init__(self, sess, image_size=256, batch_size=1, reconst_coeff=10,
                 dataset_name='cityscapes_GAN', z_dim=8, latent_coeff=0.5, kl_coeff=0.01,
                 checkpoint_dir=None, data=None, data_desc=None,
                 checkpoint=None):
        """
        https://colab.research.google.com/drive/1Enc-pKlP4Q3cimEBfcQv0B_6hUvjVL3o?sandboxMode=true#scrollTo=p7xUHe93Xq61
        now
        https://github.com/prakashpandey9/BicycleGAN/blob/master/model.py
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            capacity: How many fake generations to keep around. [50]
            dataset_name: Name of dataset to be loaded and trained on.
            checkpoint_dir: Path to directory where checkpoint will be saved
            checkpoint: Path to directory where a checkpoint will be loaded from
            z_dim: size of latent vector [8]
            data: Object that delivers the datasets.
            data_desc: Received the shape of data, to build the model/graph.
        """
        self.sess = sess
        self.batch_size = batch_size
        self.image_size = image_size

        self.Z_dim=z_dim
        self.reconst_coeff = reconst_coeff
        self.latent_coeff = latent_coeff
        self.kl_coeff = kl_coeff

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

        self.z = tf.placeholder(tf.float32, [self.batch_size, self.Z_dim], name='latent_vector')

        self.encoded_true_img, self.encoded_mu, self.encoded_log_sigma = self.Encoder(self.real_B)
        self.desired_gen_img = self.Generator(self.real_A, self.encoded_true_img)

        self.LR_desired_img = self.Generator(self.real_A, self.z)
        self.reconst_z, self.reconst_mu, self.reconst_log_sigma = self.Encoder(self.LR_desired_img)

        self.P_real = self.Discriminator(self.real_B) # Probability of ground_truth/real image (B) as real/fake
        self.P_fake = self.Discriminator(self.LR_desired_img) # Probability of generated output images (G(A, N(z)) as real/fake
        self.P_fake_encoded = self.Discriminator(self.desired_gen_img) # Probability of generated output images (G(A, Q(z|B)) as real/fake

        self.loss_vae_gan_D = (tf.reduce_mean(tf.squared_difference(self.P_real, 0.9)) + tf.reduce_mean(tf.square(self.P_fake_encoded)))

        self.loss_lr_gan_D = (tf.reduce_mean(tf.squared_difference(self.P_real, 0.9)) + tf.reduce_mean(tf.square(self.P_fake)))

        self.loss_vae_gan_GE = tf.reduce_mean(tf.squared_difference(self.P_fake_encoded, 0.9)) #G

        self.loss_gan_G = tf.reduce_mean(tf.squared_difference(self.P_fake, 0.9))

        self.loss_vae_GE = tf.reduce_mean(tf.abs(self.real_B - self.desired_gen_img)) #G

        self.loss_latent_GE = tf.reduce_mean(tf.abs(self.z - self.reconst_z)) #G

        self.loss_kl_E = 0.5 * tf.reduce_mean(-1 - self.encoded_log_sigma + self.encoded_mu ** 2 + tf.exp(self.encoded_log_sigma))

        self.loss_D = self.loss_vae_gan_D + self.loss_lr_gan_D - tf.reduce_mean(tf.squared_difference(self.P_real, 0.9))
        self.loss_G = self.loss_vae_gan_GE + self.loss_gan_G + self.reconst_coeff*self.loss_vae_GE + self.latent_coeff*self.loss_latent_GE
        self.loss_E = self.loss_vae_gan_GE + self.reconst_coeff*self.loss_vae_GE + self.latent_coeff*self.loss_latent_GE + self.kl_coeff*self.loss_kl_E

        self.d_loss_sum = tf.summary.scalar("d_loss", self.loss_D)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.loss_G)
        self.e_loss_sum = tf.summary.scalar("e_loss", self.loss_E)

        self.Pr_sum = tf.summary.histogram("P_real", self.P_real)
        self.Pf_sum = tf.summary.histogram("P_fake", self.P_fake)
        self.Pfe_sum = tf.summary.histogram("P_fake_encoded", self.P_fake_encoded)

        self.real_A_sum = tf.summary.image("Input", deprocess(self.real_A)[...,::-1])
        self.real_B_sum = tf.summary.image("Target", deprocess(self.real_B)[...,::-1])
        self.Des_sum = tf.summary.image("Desired_Target", deprocess(self.desired_gen_img)[...,::-1])
        self.Gen_sum = tf.summary.image("Generated_Image", deprocess(self.LR_desired_img)[...,::-1])

        self.saver = tf.train.Saver()

    def train(self, args):
        """Train cycleGAN"""

        # Optimizer
        self.dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator")
        self.gen_var= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator")
        self.enc_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Encoder")
        opt = tf.train.AdamOptimizer(args.lr, beta1=0.5)



        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.D_solver = opt.minimize(self.loss_D, var_list = self.dis_var)

        with tf.control_dependencies([self.D_solver]):
            self.G_solver = opt.minimize(self.loss_G, var_list = self.gen_var)

        with tf.control_dependencies([self.G_solver]):
            self.E_solver = opt.minimize(self.loss_E, var_list = self.enc_var)


        init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
        self.sess.run(init_op)

        if args.checkpoint is not None and self.load(os.path.join(args.EXP_OUT,str(args.checkpoint))):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        self.sum = tf.summary.merge([self.d_loss_sum,self.g_loss_sum,self.e_loss_sum,
                                     self.Pr_sum,self.Pf_sum,self.Pfe_sum,
                                     self.real_A_sum, self.real_B_sum, self.Des_sum, self.Gen_sum])

        self.writer = tf.summary.FileWriter(self.checkpoint_dir, self.sess.graph)

        input_data = self.data.get_trainset()
        data_iterator = input_data.repeat(args.max_epochs).batch(args.batch_size).make_one_shot_iterator()
        data_handle = self.sess.run(data_iterator.string_handle())

        train_step = 0
        idx = 0
        start_time = time.time()

        while True:
            batch_z = np.random.normal(size=(self.batch_size, self.Z_dim))
            try:
                if train_step % args.num_print == 0:
                    summary, _, loss_d, loss_g, loss_e   = self.sess.run([self.sum, self.E_solver,self.loss_D, self.loss_G, self.loss_E],
                                                          feed_dict={ self.iter_handle: data_handle,
                                                                      self.z: batch_z})
                    self.writer.add_summary(summary, train_step)
                    print("Step: [%2d] rate: %4.4f steps/sec, d_loss: %.8f, g_loss: %.8f, e_loss: %.8f" \
                        % (train_step,(train_step+1)/(time.time() - start_time),loss_d,loss_g,loss_e))
                    stdout.flush()
                else:
                    _ = self.sess.run(self.E_solver,
                                      feed_dict={ self.iter_handle: data_handle,
                                                  self.z: batch_z})

            except tf.errors.OutOfRangeError:
                print("INFO: Done with all steps")
                self.save(self.checkpoint_dir, train_step)
                break

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
        model_name = "bicycleGAN.model"

        self.saver.save(self.sess,
                        os.path.join(self.checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        self.saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint_dir))
        self.graph = tf.get_default_graph()

        return True

    def Discriminator(self, x, is_training=True, reuse=True):
        with tf.variable_scope("Discriminator", reuse=tf.AUTO_REUSE):
            x = lrelu_layer(conv2d_layer(x, 64, 4, 4, 2, 2, name='d_conv1'))
            x = lrelu_layer(bn_layer(conv2d_layer(x, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
            x = lrelu_layer(bn_layer(conv2d_layer(x, 256, 4, 4, 2, 2, name='d_conv3'), is_training=is_training, scope='d_bn3'))
            x = lrelu_layer(bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='d_conv4'), is_training=is_training, scope='d_bn4'))
            x = lrelu_layer(bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='d_conv5'), is_training=is_training, scope='d_bn5'))
            x = lrelu_layer(bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='d_conv6'), is_training=is_training, scope='d_bn6'))
            x = conv2d_layer(x, 1, 4, 4, 1, 1, name='d_conv7')
            x = tf.reshape(x, [self.batch_size, -1]) # Can use tf.reduce_mean(x, axis=[1, 2, 3])
            x = linear_layer(x, 1, 16, scope='d_fc8')
            x = tf.nn.sigmoid(x)

        return x

    def Generator(self, x, z, is_training=True, reuse=True):
        with tf.variable_scope("Generator", reuse=tf.AUTO_REUSE):
            conv_layer = []
            z = tf.reshape(z, [self.batch_size, 1, 1, self.Z_dim])
            z = tf.tile(z, [1, self.image_size, self.image_size, 1])
            x = tf.concat([x, z], axis=3)
            x = lrelu_layer(conv2d_layer(x, 64, 4, 4, 2, 2, name='g_conv1'))
            conv_layer.append(x)
            x = lrelu_layer(bn_layer(conv2d_layer(x, 128, 4, 4, 2, 2, name='g_conv2'), is_training=is_training, scope='g_bn2'))
            conv_layer.append(x)
            x = lrelu_layer(bn_layer(conv2d_layer(x, 256, 4, 4, 2, 2, name='g_conv3'), is_training=is_training, scope='g_bn3'))
            conv_layer.append(x)
            x = lrelu_layer(bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='g_conv4'), is_training=is_training, scope='g_bn4'))
            conv_layer.append(x)
            x = lrelu_layer(bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='g_conv5'), is_training=is_training, scope='g_bn5'))
            conv_layer.append(x)
            x = lrelu_layer(bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='g_conv6'), is_training=is_training, scope='g_bn6'))
            conv_layer.append(x)
            x = lrelu_layer(bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='g_conv7'), is_training=is_training, scope='g_bn7'))
            conv_layer.append(x)
            x = lrelu_layer(bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='g_conv8'), is_training=is_training, scope='g_bn8'))


            x = lrelu_layer(bn_layer(deconv2d_layer(x, 512, 4, 4, 2, 2, name='g_dconv1'), is_training=is_training, scope='gd_bn1'))
            x = tf.concat([x, conv_layer.pop()], axis=3)
            x = lrelu_layer(bn_layer(deconv2d_layer(x, 512, 4, 4, 2, 2, name='g_dconv2'), is_training=is_training, scope='gd_bn2'))
            x = tf.concat([x, conv_layer.pop()], axis=3)
            x = lrelu_layer(bn_layer(deconv2d_layer(x, 512, 4, 4, 2, 2, name='g_dconv3'), is_training=is_training, scope='gd_bn3'))
            x = tf.concat([x, conv_layer.pop()], axis=3)
            x = lrelu_layer(bn_layer(deconv2d_layer(x, 512, 4, 4, 2, 2, name='g_dconv4'), is_training=is_training, scope='gd_bn4'))
            x = tf.concat([x, conv_layer.pop()], axis=3)
            x = lrelu_layer(bn_layer(deconv2d_layer(x, 256, 4, 4, 2, 2, name='g_dconv5'), is_training=is_training, scope='gd_bn5'))
            x = tf.concat([x, conv_layer.pop()], axis=3)
            x = lrelu_layer(bn_layer(deconv2d_layer(x, 128, 4, 4, 2, 2, name='g_dconv6'), is_training=is_training, scope='gd_bn6'))
            x = tf.concat([x, conv_layer.pop()], axis=3)
            x = lrelu_layer(bn_layer(deconv2d_layer(x, 64, 4, 4, 2, 2, name='g_dconv7'), is_training=is_training, scope='gd_bn7'))
            x = tf.concat([x, conv_layer.pop()], axis=3)
            x = lrelu_layer(bn_layer(deconv2d_layer(x, 3, 4, 4, 2, 2, name='g_dconv8'), is_training=is_training, scope='gd_bn8'))
            x = tf.tanh(x)

        return x

    def Encoder(self, x, is_training=True, reuse=True):
        with tf.variable_scope("Encoder", reuse=tf.AUTO_REUSE):
            x = lrelu_layer(conv2d_layer(x, 64, 4, 4, 2, 2, name='e_conv1'))

            x = residual_block(x, 128, 3, is_training=is_training, name='res_1')
            x = tf.contrib.layers.avg_pool2d(x, 2, 2, padding='SAME')

            x = residual_block(x, 256, 3, is_training=is_training, name='res_2')
            x = tf.contrib.layers.avg_pool2d(x, 2, 2, padding='SAME')

            x = residual_block(x, 512, 3, is_training=is_training, name='res_3')
            x = tf.contrib.layers.avg_pool2d(x, 2, 2, padding='SAME')

            x = residual_block(x, 512, 3, is_training=is_training, name='res_4')
            x = tf.contrib.layers.avg_pool2d(x, 2, 2, padding='SAME')

            x = residual_block(x, 512, 3, is_training=is_training, name='res_5')
            x = tf.contrib.layers.avg_pool2d(x, 2, 2, padding='SAME')

            x = tf.contrib.layers.avg_pool2d(x, 8, 8, padding='SAME')
            # x = tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])]) # Flattening
            x = tf.contrib.layers.flatten(x)

            mu = linear_layer(x, self.Z_dim, 512,scope='e_fc1')

            log_sigma = linear_layer(x, self.Z_dim, 512,scope='e_fc2')

            z = mu + tf.random_normal(shape=tf.shape(self.Z_dim)) * tf.exp(log_sigma)

        return z, mu, log_sigma
