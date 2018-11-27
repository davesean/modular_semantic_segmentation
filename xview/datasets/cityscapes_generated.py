import numpy as np
from os import listdir, path, environ
import cv2
import tarfile
import glob
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import data as dt
from xview.settings import DATA_BASEPATH
from .data_baseclass import DataBaseclass
from .augmentation import augmentate, crop_multiple
from copy import deepcopy


CITYSCAPES_BASEPATH = path.join(DATA_BASEPATH, 'cityscapes')

class Cityscapes_generated(DataBaseclass):

    _data_shape_description = {
            'labels': (None, None, 3), 'pos': (None, None,3), 'neg': (None, None,3)}
    _num_default_classes = 2

    def __init__(self, base_path=CITYSCAPES_BASEPATH, batchsize=1):

        if not path.exists(base_path):
            message = 'ERROR: Path to CITYSCAPES dataset does not exist.'
            print(message)
            raise IOError(1, message, base_path)

        self.base_path = base_path
        self.id = self.base_path.split("/")[-1].split("_")[0]
        measure_path = self._create_Dictionary(base_path,'measure')
        training_path = self._create_Dictionary(base_path,'training')
        validation_path = self._create_Dictionary(base_path,'validation')

        # Intitialize Baseclass
        DataBaseclass.__init__(self, training_path, measure_path, testset=None,
                              validation_set=validation_path, labelinfo=None)

    def _create_Dictionary(self,path,prefix):
        target_list = glob.glob(os.path.join(path,prefix,"target_"+prefix+"*.png"))
        target_list.sort()
        fake_list = glob.glob(os.path.join(path,prefix,self.id+"_"+prefix+"*.png"))
        fake_list.sort()
        wrong_list = fake_list.copy()
        wrong_list.append(wrong_list.pop(0))
        # segm_list = glob.glob(os.path.join(path,"measure","input_measure*.png"))
        # segm_list.sort()
        # path_dics = {'target': target_list,'fake': fake_list,'segm': segm_list}
        path_dics = {'labels': target_list,'pos': fake_list,'neg': wrong_list}
        # print(path_dics['pos'][5],path_dics['neg'][5])
        return path_dics

    def _load_data(self, image_path):
        blob = {}
        blob['labels'] = cv2.imread(image_path['labels'])
        blob['pos'] = cv2.imread(image_path['pos'])
        blob['neg'] = cv2.imread(image_path['neg'])

        return blob

    def _get_data(self, image_path, training_format=False):
        """Returns data for one given image number from the specified sequence."""

        blob = self._load_data(image_path)

        # if training_format:
        #     blob = augmentate(blob, **self.config['augmentation'])

        # blob['rgb'] = cv2.resize(blob['rgb'], (256, 256),
        #                          interpolation=cv2.INTER_LINEAR)
        # blob['labels'] = cv2.resize(blob['labels'], (256, 256),
        #                          interpolation=cv2.INTER_NEAREST)

        return blob

    def _get_tf_dataset(self, setlist, training_format=False):
        def data_generator():
            for i in range(len(setlist['labels'])):
                item = {'labels': setlist['labels'][i],
                        'pos': setlist['pos'][i],
                        'neg': setlist['neg'][i]}
                data = self._get_data(item, training_format=training_format)
                yield data

        return tf.data.Dataset.from_generator(data_generator,
                                              *self.get_data_description()[:2])

    def get_ego_vehicle_mask(self, image_name, image_path):
        # save the old label mapping
        old_label_lookup = deepcopy(self.label_lookup)

        # now map everything on 0 except for ego-vehicle, which is mapped on 1
        self.label_lookup = [0 for _ in range(34)]
        self.label_lookup[1] = 1

        blob = self._load_data(image_name, image_path)

        # restore labellookup
        self.label_lookup = old_label_lookup
        return blob
