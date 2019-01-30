from __future__ import division
import os,time,scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
from sys import stdout
from glob import glob
import numpy as np
import cv2
from six.moves import xrange
import sys

def lrelu(x):
    return tf.maximum(0.2*x,x)

def build_net(ntype,nin,nwb=None,name=None):
    if ntype=='conv':
        return tf.nn.relu(tf.nn.conv2d(nin,nwb[0],strides=[1,1,1,1],padding='SAME',name=name)+nwb[1])
    elif ntype=='pool':
        return tf.nn.avg_pool(nin,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def get_weight_bias(vgg_layers,i):
    weights=vgg_layers[i][0][0][2][0][0]
    weights=tf.constant(weights)
    bias=vgg_layers[i][0][0][2][0][1]
    bias=tf.constant(np.reshape(bias,(bias.size)))
    return weights,bias

def build_vgg19(input,reuse=False,path=None):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    net={}
    vgg_rawnet=scipy.io.loadmat(path)
    vgg_layers=vgg_rawnet['layers'][0]
    net['input']=input-np.array([123.6800, 116.7790, 103.9390]).reshape((1,1,1,3))
    net['conv1_1']=build_net('conv',net['input'],get_weight_bias(vgg_layers,0),name='vgg_conv1_1')
    net['conv1_2']=build_net('conv',net['conv1_1'],get_weight_bias(vgg_layers,2),name='vgg_conv1_2')
    net['pool1']=build_net('pool',net['conv1_2'])
    net['conv2_1']=build_net('conv',net['pool1'],get_weight_bias(vgg_layers,5),name='vgg_conv2_1')
    net['conv2_2']=build_net('conv',net['conv2_1'],get_weight_bias(vgg_layers,7),name='vgg_conv2_2')
    net['pool2']=build_net('pool',net['conv2_2'])
    net['conv3_1']=build_net('conv',net['pool2'],get_weight_bias(vgg_layers,10),name='vgg_conv3_1')
    net['conv3_2']=build_net('conv',net['conv3_1'],get_weight_bias(vgg_layers,12),name='vgg_conv3_2')
    net['conv3_3']=build_net('conv',net['conv3_2'],get_weight_bias(vgg_layers,14),name='vgg_conv3_3')
    net['conv3_4']=build_net('conv',net['conv3_3'],get_weight_bias(vgg_layers,16),name='vgg_conv3_4')
    net['pool3']=build_net('pool',net['conv3_4'])
    net['conv4_1']=build_net('conv',net['pool3'],get_weight_bias(vgg_layers,19),name='vgg_conv4_1')
    net['conv4_2']=build_net('conv',net['conv4_1'],get_weight_bias(vgg_layers,21),name='vgg_conv4_2')
    net['conv4_3']=build_net('conv',net['conv4_2'],get_weight_bias(vgg_layers,23),name='vgg_conv4_3')
    net['conv4_4']=build_net('conv',net['conv4_3'],get_weight_bias(vgg_layers,25),name='vgg_conv4_4')
    net['pool4']=build_net('pool',net['conv4_4'])
    net['conv5_1']=build_net('conv',net['pool4'],get_weight_bias(vgg_layers,28),name='vgg_conv5_1')
    net['conv5_2']=build_net('conv',net['conv5_1'],get_weight_bias(vgg_layers,30),name='vgg_conv5_2')
    return net

def recursive_generator(label,sp):
    dim=512 if sp>=128 else 1024
    if sp==4:
        input=label
    else:
        # downsampled=tf.image.resize_area(label,(sp//2,sp),align_corners=False) # If image is square, (sp//2,sp//2) if rectang(1:2) (sp//2,sp)
        # temp = tf.image.resize_bilinear(recursive_generator(downsampled,sp//2),(sp,sp*2),align_corners=True) # If image is square, (sp,sp) if rectang(1:2) (sp,sp*2)
        downsampled=tf.image.resize_area(label,(sp//2,sp//2),align_corners=False) # If image is square, (sp//2,sp//2) if rectang(1:2) (sp//2,sp)
        temp = tf.image.resize_bilinear(recursive_generator(downsampled,sp//2),(sp,sp),align_corners=True) # If image is square, (sp,sp) if rectang(1:2) (sp,sp*2)
        input=tf.concat([temp,label],3)
        # input=tf.concat([tf.image.resize_bilinear(recursive_generator(downsampled,sp//2),(sp,sp*2),align_corners=True),label],3)
    net=slim.conv2d(input,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv1')
    net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv2')
    if sp==256:
        net=slim.conv2d(net,3,[1,1],rate=1,activation_fn=None,scope='g_'+str(sp)+'_conv100')
        net = deprocess(net)
    return net

def compute_error(real,fake,label,size):
    # if size == 256:
    #     return tf.reduce_sum(tf.reduce_mean(label*tf.expand_dims(tf.reduce_mean(tf.abs(fake-real),reduction_indices=[3]),-1),reduction_indices=[1,2]))#diversity loss
    # else:
    return tf.reduce_mean(tf.abs(fake-real))#simple loss

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 255] => [-1, 1]
        return image/255 * 2 - 1
def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 255]
        return ((image + 1) / 2)*255

class cascRef(object):
    def __init__(self, sess, dataset_name='cityscapes_GAN',image_size=1024,
                 checkpoint_dir=None, data=None, data_desc=None,
                 is_training=True, checkpoint=None, vgg_checkpoint=None):
        """
        Args:
            sess: TensorFlow session
            dataset_name: Name of dataset to be loaded and trained on.
            checkpoint_dir: Path to directory where checkpoint will be saved
            checkpoint: Path to directory where a checkpoint will be loaded from
            data: Object that delivers the datasets.
            data_desc: Received the shape of data, to build the model/graph.
        """
        self.sess = sess
        self.image_size = image_size
        self.data = data
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint = checkpoint
        self.checkpoint_loaded = False
        # Get the data descriptors with the shape of data coming
        data_description = data_desc

        self.num_classes = data_desc[2]

        data_description = [data_description[0], {
            key: [None, *description]
            for key, description in data_description[1].items()}]

        # Create an iterator for the data
        self.iter_handle = tf.placeholder(tf.string, shape=[],
                                              name='training_placeholder')
        iterator = tf.data.Iterator.from_string_handle(
            self.iter_handle, *data_description)
        training_batch = iterator.get_next()
        # if is_training:
        self.vgg_checkpoint=vgg_checkpoint
        assert(self.vgg_checkpoint is not None)
        self.build_model(training_batch['labels'], training_batch['rgb'])


    def build_model(self, label, target):
        self.target = target
        self.input = label
        if self.image_size == 256:
            vgg_factors = [1.6, 2.3, 1.8,2.8,12.5]
        else:
            vgg_factors = [2.6, 4.8, 3.7,5.6,(20/3)]


        with tf.variable_scope(tf.get_variable_scope()):

            self.generator=recursive_generator(label,self.image_size)

            vgg_real=build_vgg19(target[...,::-1],path=self.vgg_checkpoint)
            vgg_fake=build_vgg19(self.generator,path=self.vgg_checkpoint)

            self.p0=compute_error(vgg_real['input'],vgg_fake['input'],label,self.image_size)
            self.p1=compute_error(vgg_real['conv1_2'],vgg_fake['conv1_2'],label,self.image_size)/vgg_factors[0]
            self.p2=compute_error(vgg_real['conv2_2'],vgg_fake['conv2_2'],tf.image.resize_area(label,(self.image_size//2,self.image_size)),self.image_size)/vgg_factors[1]
            self.p3=compute_error(vgg_real['conv3_2'],vgg_fake['conv3_2'],tf.image.resize_area(label,(self.image_size//4,self.image_size//2)),self.image_size)/vgg_factors[2]
            self.p4=compute_error(vgg_real['conv4_2'],vgg_fake['conv4_2'],tf.image.resize_area(label,(self.image_size//8,self.image_size//4)),self.image_size)/vgg_factors[3]
            self.p5=compute_error(vgg_real['conv5_2'],vgg_fake['conv5_2'],tf.image.resize_area(label,(self.image_size//16,self.image_size//8)),self.image_size)*vgg_factors[4]
            self.G_loss=self.p0+self.p1+self.p2+self.p3+self.p4+self.p5

        self.lr=tf.placeholder(tf.float32)

        self.G_opt=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.G_loss,var_list=[var for var in tf.trainable_variables() if var.name.startswith('g_')])

        self.sess.run(tf.global_variables_initializer())

        self.real_B_sum = tf.summary.image("Target", target[...,::-1])
        self.fake_B_sum = tf.summary.image("Generated", self.generator[...,::-1])

        self.g_loss_sum = tf.summary.scalar("g_loss", self.G_loss)

        self.saver = tf.train.Saver()

        if self.checkpoint is not None:
            self.checkpoint_loaded = True
            self.load(self.checkpoint)


    def train(self, args):
        """Train cascRef"""
        print("INFO: Beginning training")
        stdout.flush()

        self.sum = tf.summary.merge([self.real_B_sum, self.fake_B_sum, self.g_loss_sum])
        self.writer = tf.summary.FileWriter(self.checkpoint_dir, self.sess.graph)

        input_data = self.data.get_trainset()
        data_iterator = input_data.repeat(args.max_epochs).batch(1).make_one_shot_iterator()
        data_handle = self.sess.run(data_iterator.string_handle())
        counterTrain = 1
        start_time = time.time()
        while True:
            if np.mod(counterTrain, args.num_print) == 1:
                try:
                    _, summary_str, g_l,p0_l,p1_l,p2_l,p3_l,p4_l,p5_l = self.sess.run([self.G_opt, self.sum, self.G_loss, self.p0,self.p1,self.p2,self.p3,self.p4,self.p5],
                                                   feed_dict={ self.iter_handle: data_handle, self.lr:1e-4 })
                except tf.errors.OutOfRangeError:
                    print("INFO: Done with all steps")
                    self.save(self.checkpoint_dir, counterTrain)
                    break

                self.writer.add_summary(summary_str, counterTrain)
                print("Step: [%2d] rate: %4.4f steps/sec, g_loss: %.8f, p0_loss: %.8f, p1_loss: %.8f, p2_loss: %.8f, p3_loss: %.8f, p4_loss: %.8f, p5_loss: %.8f" \
                    % (counterTrain,counterTrain/(time.time() - start_time), g_l,p0_l,p1_l,p2_l,p3_l,p4_l,p5_l))
                stdout.flush()

            else:
                try:
                    self.sess.run(self.G_opt,feed_dict={ self.iter_handle: data_handle, self.lr:1e-4 })
                except tf.errors.OutOfRangeError:
                    print("INFO: Done with all training steps")
                    self.save(self.checkpoint_dir, counterTrain)
                    break
            counterTrain += 1

        # return vals

    def transform(self, args, labels):
        """Transforms dataset from labels [batch_size, img_h, img_w] or [batch_size, img_h, img_w, num_classes]
        This is used for pipeline.
        """
        # Check that a checkpoint was loaded at init
        assert(self.checkpoint_loaded is not False)

        # If labels not yet in one hot format, transform data.
        if labels.ndim == 3:
            labels = (np.arange(self.num_classes) == labels[...,None]).astype(int) # Make one hot for 12 classes.
            print(labels.shape)
            stdout.flush()

        # synth = np.zeros((labels.shape))
        synth = np.zeros((labels.shape[0],labels.shape[1],labels.shape[2], 3))

        data = {'rgb': tf.zeros(shape=[labels.shape[0],labels.shape[1],labels.shape[2], 3], dtype=tf.float32), 'labels': tf.to_float(labels)}

        iterator = tf.data.Dataset.from_tensor_slices(data)\
                   .batch(1).make_one_shot_iterator()
        handle = self.sess.run(iterator.string_handle())

        for i in range(labels.shape[0]):
            outImage = self.sess.run([self.generator],
                                           feed_dict={ self.iter_handle: handle })

            synth[i,:,:,:] = outImage[0]

        return synth

    def transformDatasets(self, args, data_desc):
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
                    outImage, inpt, target = self.sess.run([self.generator,self.input,self.target],
                                                   feed_dict={ self.iter_handle: set_handles[sets] })
                # When tf dataset is empty this error is thrown
                except tf.errors.OutOfRangeError:
                    print("INFO: Done with "+sets+" set")
                    break

                # Save the output of the generator
                filename = str(args.checkpoint)+"_"+ sets + str(counter) + ".png"
                tmp = outImage[...,::-1]
                cv2.imwrite(os.path.join(local_folder,filename), (tmp[0,:,:,:]), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                # This is wrong, need to transform one hot labels back to bgr image, but I dont need them, so not changing atm
                outputColor = data_desc.coloured_labels(labels=np.argmax(inpt,axis=3))
                outputColor = outputColor[0,:,:,:]
                filename = "input_" + sets + str(counter) + ".png"
                cv2.imwrite(os.path.join(local_folder,filename), outputColor[...,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                filename = "target_"+ sets + str(counter) + ".png"
                cv2.imwrite(os.path.join(local_folder,filename), (target[0,:,:,:]), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                counter +=1


    def save(self, checkpoint_dir, step):
        model_name = "cascRef.model"

        self.saver.save(self.sess,
                        os.path.join(self.checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        self.saver.restore(self.sess, checkpoint)
        self.graph = tf.get_default_graph()
        return True
