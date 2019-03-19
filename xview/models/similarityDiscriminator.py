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

def preprocess(image, grayscale=False):
    with tf.name_scope("preprocess"):
        if grayscale:
            image = tf.image.rgb_to_grayscale(image[...,::-1])
        # [0, 255] => [-1, 1]
        return image/255 * 2 - 1
# def deprocess(image):
#     with tf.name_scope("deprocess"):
#         # [-1, 1] => [0, 255]
#         return ((image + 1) / 2)*255

class DiffDiscrim(object):
    def __init__(self, sess, image_size=256, seed=42, use_segm=False,
                 batch_size=64, df_dim=64, ppd=8, pos_weight=1, use_grayscale=False,
                 input_c_dim=3, is_training=True, arch='arch1', batch_norm=False,
                 checkpoint_dir=None, data=None, momentum=0.9, checkpoint=None,
                 feature_extractor=None):
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
        self.batch_size = batch_size
        self.image_size = image_size
        self.batch_norm = batch_norm
        self.pos_weight = pos_weight
        self.use_grayscale = use_grayscale
        self.use_segm = use_segm
        self.feature_extractor = feature_extractor

        self.df_dim = df_dim

        self.batch_momentum = momentum
        self.arch = arch
        self.archDisc = simArch(df_dim=df_dim, arch=arch, ckpt=self.feature_extractor)

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
        if arch[:4] == 'arch':
            self.build_model_arch(training_batch['labels'],training_batch['pos'],training_batch['neg'],training_batch['pos_segm'],training_batch['neg_segm'])
        else:
            self.build_model_feat(training_batch['labels'],training_batch['pos'],training_batch['neg'],training_batch['pos_segm'],training_batch['neg_segm'])

        if checkpoint is not None and not(checkpoint.split('/')[-1] == "None"):
            self.load(checkpoint)
            self.checkpoint_loaded = True


    def build_model_feat(self, target, pos, neg, pos_segm, neg_segm):

        def lrelu(x, leak=0.2, name="lrelu"):
          return tf.maximum(x, leak*x)
        def dense(input_, output_size, input_size, num_channels, name="dense", reuse=False, stddev=0.02, bias_start=0.0):
            shape = input_size * input_size * num_channels
            with tf.variable_scope(name, reuse=reuse):
                matrix = tf.get_variable("Matrix", [shape, output_size], tf.float32,
                                         tf.random_normal_initializer(stddev=stddev))
                bias = tf.get_variable("bias", [output_size],
                    initializer=tf.constant_initializer(bias_start))
                return tf.matmul(tf.layers.flatten(input_), matrix) + bias

        tf.set_random_seed(self.seed)
        self.target_placeholder = preprocess(target)
        self.pos_placeholder = preprocess(pos)
        self.neg_placeholder = preprocess(neg)
        self.pos_segm_placeholder = preprocess(pos_segm)
        self.neg_segm_placeholder = preprocess(neg_segm)
        self.train_flag = tf.placeholder(tf.bool, name="Train_flag")

        # Feature Extractor
        _, self.Feature_R, _ = self.archDisc.get_output(image=self.target_placeholder,
                                                         reuse=True,
                                                         is_training=self.train_flag,
                                                         bn=self.batch_norm,
                                                         bs=self.batch_size,
                                                         image_semSeg=self.pos_segm_placeholder)
        _, self.Feature_P, _ = self.archDisc.get_output(image=self.pos_placeholder,
                                                           reuse=True,
                                                           is_training=self.train_flag,
                                                           bn=self.batch_norm,
                                                           bs=self.batch_size,
                                                           image_semSeg=self.pos_segm_placeholder)
        _, self.Feature_N, _ = self.archDisc.get_output(image=self.neg_placeholder,
                                                           reuse=True,
                                                           is_training=self.train_flag,
                                                           bn=self.batch_norm,
                                                           bs=self.batch_size,
                                                           image_semSeg=self.neg_segm_placeholder)


        PosExample = (tf.concat([self.Feature_R, self.Feature_P], 3))
        NegExample = (tf.concat([self.Feature_R, self.Feature_N], 3))
        # (32x 32 x (64*4)x2) for full image
        # (4 x 4 x (64*4)x2) for patches (256/8->32)



        # PosExample = tf.Print(PosExample, [tf.reduce_max(PosExample), tf.reduce_max(NegExample), tf.reduce_min(PosExample), tf.reduce_min(NegExample)])

        posFlat = tf.layers.flatten(PosExample)
        negFlat = tf.layers.flatten(NegExample)

        interm_nodes = 1024
        #Feat 2
        # # 8x8x64x4x2

        if self.arch[:5] == 'feat1':
            inp_nodes=int(self.df_dim*8)
        elif self.arch == 'feat2':
            inp_nodes=int(self.df_dim*32)


        dense1_pos = lrelu(dense(posFlat, interm_nodes, 4, inp_nodes, name="s_dense1"))
        dense2_pos = lrelu(dense(dense1_pos, interm_nodes, 2, int(interm_nodes/4), name="s_dense2"))
        dense3_pos = dense(dense2_pos, 1, 2, int(interm_nodes/4), name="s_dense3")

        dense1_neg = lrelu(dense(negFlat, interm_nodes, 4, inp_nodes, name="s_dense1", reuse=True))
        dense2_neg = lrelu(dense(dense1_neg, interm_nodes, 2, int(interm_nodes/4), name="s_dense2", reuse=True))
        dense3_neg = dense(dense2_neg, 1, 2, int(interm_nodes/4), name="s_dense3", reuse=True)

        self.D = tf.nn.sigmoid(dense3_pos)
        self.D_ = tf.nn.sigmoid(dense3_neg)

        self.d_loss_pos = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dense3_pos, labels=tf.zeros_like(dense3_pos)))
        self.d_loss_neg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dense3_neg, labels=tf.ones_like(dense3_neg)))

        self.d_loss = self.pos_weight*self.d_loss_pos + self.d_loss_neg
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


    def build_model_arch(self, target, pos, neg, pos_segm, neg_segm):
        tf.set_random_seed(self.seed)
        self.target_placeholder = preprocess(target,self.use_grayscale)
        self.pos_placeholder = preprocess(pos,self.use_grayscale)
        self.neg_placeholder = preprocess(neg,self.use_grayscale)
        self.pos_segm_placeholder = preprocess(pos_segm)
        self.neg_segm_placeholder = preprocess(neg_segm)
        self.train_flag = tf.placeholder(tf.bool, name="Train_flag")

        if self.use_segm:
            PosExample = tf.concat([self.target_placeholder, self.pos_placeholder, self.pos_segm_placeholder], 3)
            NegExample = tf.concat([self.target_placeholder, self.neg_placeholder, self.neg_segm_placeholder], 3)
        else:
            PosExample = tf.concat([self.target_placeholder, self.pos_placeholder], 3)
            NegExample = tf.concat([self.target_placeholder, self.neg_placeholder], 3)

        # self.D, self.D_logits = self.discriminator(PosExample)
        # self.D_, self.D_logits_ = self.discriminator(NegExample,reuse=True)

        # Feature Extractor + Decision/Metric
        self.D, self.D_logits, entropy = self.archDisc.get_output(image=PosExample,
                                                         is_training=self.train_flag,
                                                         bn=self.batch_norm,
                                                         bs=self.batch_size)
        self.D_, self.D_logits_, entropy = self.archDisc.get_output(image=NegExample,
                                                           reuse=True,
                                                           is_training=self.train_flag,
                                                           bn=self.batch_norm,
                                                           bs=self.batch_size)

        if entropy == 'softmax':
            self.d_loss_pos = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.D_logits, labels=tf.zeros_like(self.D)))
            self.d_loss_neg = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.D_logits_, labels=tf.ones_like(self.D_)))
        elif entropy == 'sigmoid':
            self.d_loss_pos = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.zeros_like(self.D)))
            self.d_loss_neg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        self.d_loss = self.pos_weight*self.d_loss_pos + self.d_loss_neg
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
        def transformConv(self, realImages, synthImages, segmImages):
            """ Predict similarity between images """
            counter = 1

            # Check that a checkpoint directory is given, to load from
            if not self.checkpoint_loaded:
                assert(False, "No checkpoint loaded, load one at init of model.")

            data = {'labels': tf.to_float(realImages), 'pos': tf.to_float(synthImages), 'neg': tf.zeros_like(tf.to_float(synthImages)),'pos_segm': tf.to_float(segmImages), 'neg_segm': tf.zeros_like(tf.to_float(segmImages)) }

            iterator = tf.data.Dataset.from_tensor_slices(data)\
                       .batch(1).make_one_shot_iterator()
            handle = self.sess.run(iterator.string_handle())


            for k in range(realImages.shape[0]):
                output = self.sess.run(self.D, feed_dict={self.iter_handle: handle,
                                                          self.train_flag: True })

                if output.shape[-1] == 1:
                    output = np.squeeze(output, axis=-1)
                if k == 0:
                    print(output.shape)
                    output_matrix = output
                else:
                    output_matrix = np.concatenate((output_matrix, output),axis=0)

            return output_matrix

        def transformFC(self, realImages, synthImages, segmImages):
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

                # if output.ndim == 2 and output.shape[1]==1:
                #     output_temp = np.squeeze(output).reshape((self.ppd,self.ppd))
                # elif output.ndim == 3 and output.shape[1]==1 and output.shape[2]==1:
                #     output_temp = np.squeeze(output).reshape((self.ppd,self.ppd))
                # else:
                #     #  bs h  w  c
                #     # [64,32,32,1]
                #     # print(output.shape)
                #     if output.ndim == 4:
                #         output = np.squeeze(output, axis=-1)
                #     output_temp = np.concatenate((output[0:self.ppd]),axis=1)
                #     for l in range(1,self.ppd):
                #         temp_matrix = np.concatenate((output[self.ppd*l:self.ppd*(l+1)]),axis=1)
                #         output_temp = np.concatenate((output_temp,temp_matrix),axis=0)

                output_temp = np.squeeze(output).reshape((self.ppd,self.ppd))


                if k == 0:
                    output_matrix = np.expand_dims(output_temp, axis=0)
                else:
                    output_matrix = np.concatenate((output_matrix, np.expand_dims(output_temp, axis=0)),axis=0)

            return output_matrix

        if self.arch == 'arch9' or self.arch == 'arch10' or  self.arch[:4] == 'feat':
            return transformFC(self, realImages, synthImages, segmImages)
        else:
            return transformConv(self, realImages, synthImages, segmImages)

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
