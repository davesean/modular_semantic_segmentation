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
            'labels': (None, None, 3), 'pos': (None, None,3), 'neg': (None, None,3), 'pos_segm': (None, None,3), 'neg_segm': (None, None,3)}
    _num_default_classes = 2

    def __init__(self, base_path=CITYSCAPES_BASEPATH, batchsize=1):

        if not path.exists(base_path):
            message = 'ERROR: Path to CITYSCAPES dataset does not exist.'
            print(message)
            raise IOError(1, message, base_path)

        self.base_path = base_path
        self.id = self.base_path.split("/")[-1].split("_")[0]
        measure_path, self.num_measure = self._create_Dictionary(base_path,'measure')
        training_path, self.num_training = self._create_Dictionary(base_path,'training')
        validation_path, self.num_validation = self._create_Dictionary(base_path,'validation',is_validation=True)

        # Intitialize Baseclass
        DataBaseclass.__init__(self, training_path, measure_path, testset=None,
                              validation_set=validation_path, labelinfo=None)

    def _create_Dictionary(self,path,prefix, is_validation=False):
        if is_validation:
            target_list = glob.glob(os.path.join(path,prefix,"target_*.png"))
            target_list.sort()
            fake_list = glob.glob(os.path.join(path,prefix,"synth_*.png"))
            fake_list.sort()
            segm_list = glob.glob(os.path.join(path,prefix,"segm_*.png"))
            segm_list.sort()

            wrong_list = fake_list.copy()
            wrong_list.append(wrong_list.pop(0))
            wrong_seg_list = segm_list.copy()
            wrong_seg_list.append(wrong_seg_list.pop(0))

            target_full_list = target_list.copy()
            fake_full_list = fake_list.copy()
            wrong_full_list = wrong_list.copy()
            segm_full_list = segm_list.copy()
            wrong_segm_full_list = wrong_seg_list.copy()

            num = len(target_list)-2

            for i in range(num):
                target_full_list.extend(target_list)
                fake_full_list.extend(fake_list)
                segm_full_list.extend(segm_list)

                wrong_list.append(wrong_list.pop(0))
                wrong_full_list.extend(wrong_list)

                wrong_seg_list.append(wrong_seg_list.pop(0))
                wrong_segm_full_list.extend(wrong_seg_list)


            for i in range(len(target_full_list)):
                if fake_full_list[i].split('/')[-1].split('.')[0].split('n')[-1] is wrong_full_list[i].split('/')[-1].split('.')[0].split('n')[-1]:
                    print(fake_full_list[i].split('/')[-1].split('.')[0].split('n')[-1],wrong_full_list[i].split('/')[-1].split('.')[0].split('n')[-1])
                    raise ValueError('A image path in validation was the same for the positive and negative example')


            # path_dics = {'labels': target_full_list,'pos': fake_full_list,'neg': wrong_full_list}
            path_dics = {'labels': target_full_list,'pos': fake_full_list,'neg': wrong_full_list, 'pos_segm': segm_full_list, 'neg_segm': wrong_segm_full_list}

            return path_dics, len(target_full_list)
        else:
            target_list = glob.glob(os.path.join(path,prefix,"target_*.png"))
            target_list.sort()
            fake_list = glob.glob(os.path.join(path,prefix,"synth_*.png"))
            fake_list.sort()
            segm_list = glob.glob(os.path.join(path,prefix,"segm_*.png"))
            segm_list.sort()


            wrong_list = fake_list.copy()
            wrong_list.append(wrong_list.pop(0))
            wrong_seg_list = segm_list.copy()
            wrong_seg_list.append(wrong_seg_list.pop(0))

            # path_dics = {'labels': target_list,'pos': fake_list,'neg': wrong_list}
            path_dics = {'labels': target_list,'pos': fake_list,'neg': wrong_list, 'pos_segm': segm_list, 'neg_segm': wrong_seg_list}

            return path_dics, len(target_list)

    def _load_data(self, image_path):
        blob = {}
        blob['labels'] = cv2.imread(image_path['labels'])
        blob['pos'] = cv2.imread(image_path['pos'])
        blob['neg'] = cv2.imread(image_path['neg'])
        blob['pos_segm'] = cv2.imread(image_path['pos_segm'])
        blob['neg_segm'] = cv2.imread(image_path['neg_segm'])
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
                        'neg': setlist['neg'][i],
                        'pos_segm': setlist['pos_segm'][i],
                        'neg_segm': setlist['neg_segm'][i]}
                data = self._get_data(item, training_format=training_format)
                yield data

        return tf.data.Dataset.from_generator(data_generator,
                                              *self.get_data_description()[:2])

    def get_validation_set(self, num_items=None, tf_dataset=True):
        """Return testset. By default as tf.data.dataset, otherwise as numpy array."""
        if num_items is None:
            num_items = len(self.validation_set)
        if not tf_dataset:
            return self._get_batch(self.validation_set[:num_items])
        return self._get_tf_dataset(self.validation_set), self.num_validation

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
