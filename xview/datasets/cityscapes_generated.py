import numpy as np
from os import listdir, path, environ
import cv2
import tarfile
import glob
import os
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import data as dt
from xview.settings import DATA_BASEPATH
from .data_baseclass import DataBaseclass
from .augmentation import augmentate, crop_multiple, add_gaussian_noise
from copy import deepcopy
from PIL import Image
# from sys import stdout

CITYSCAPES_BASEPATH = path.join(DATA_BASEPATH, 'cityscapes')

class Cityscapes_generated(DataBaseclass):

    _data_shape_description = {
            'labels': (None, None, 3), 'pos': (None, None,3), 'neg': (None, None,3), 'pos_segm': (None, None,3), 'neg_segm': (None, None,3)}
    _num_default_classes = 2

    def __init__(self, base_path=CITYSCAPES_BASEPATH, batchsize=1, ppd=8, **data_config):

        if not path.exists(base_path):
            message = 'ERROR: Path to CITYSCAPES dataset does not exist.'
            print(message)
            raise IOError(1, message, base_path)

        config = {
            'augmentation': {
                'crop': [1, 32],
                'scale': False,
                'vflip': 1,
                'hflip': 1,
                'gamma': False,
                'rotate': False,
                'shear': False,
                'contrast': False,
                'brightness': False,
                'noise': [0, 0.01]
            },
            'resize': False
        }
        config.update(data_config)
        self.config = config

        self.base_path = base_path
        # Patches per dimension
        self.ppd = ppd
        self.id = self.base_path.split("/")[-1].split("_")[0]
        measure_path, self.num_measure = self._create_Dictionary(base_path,'measure')
        training_path, self.num_training = self._create_Dictionary(base_path,'training')
        validation_path, self.num_validation = self._create_Dictionary(base_path,'validation',is_validation=True)

        self.num_validation *= self.ppd*self.ppd
        self.num_measure *= self.ppd*self.ppd
        self.num_training *= self.ppd*self.ppd

        # Intitialize Baseclass
        DataBaseclass.__init__(self, training_path, measure_path, testset=None,
                              validation_set=validation_path, labelinfo=None)

    def _create_Dictionary(self,path,prefix, is_validation=False):

        # target_list = glob.glob(os.path.join(path,prefix,"target_*.png"))
        target_list = glob.glob(os.path.join(path,prefix,"target_"+prefix+"*.png"))
        target_list.sort()
        # fake_list = glob.glob(os.path.join(path,prefix,"synth_*.png"))
        fake_list = glob.glob(os.path.join(path,prefix,str(self.id)+"_"+prefix+"*.png"))
        fake_list.sort()
        # segm_list = glob.glob(os.path.join(path,prefix,"segm_*.png"))
        segm_list = glob.glob(os.path.join(path,prefix,"input_"+prefix+"*.png"))
        segm_list.sort()

        false_list = glob.glob(os.path.join(path,prefix,str(self.id)+"_false*.png"))
        false_seg_list = glob.glob(os.path.join(path,prefix,"input_false*.png"))

        if len(false_list) == 0:
            wrong_list = fake_list.copy()
            wrong_list.append(wrong_list.pop(0))
            wrong_seg_list = segm_list.copy()
            wrong_seg_list.append(wrong_seg_list.pop(0))
        else:
            false_list.sort()
            false_seg_list.sort()
            wrong_list = false_list
            wrong_seg_list = false_seg_list

        path_dics = {'labels': target_list,'pos': fake_list,'neg': wrong_list, 'pos_segm': segm_list, 'neg_segm': wrong_seg_list}

        return path_dics, len(target_list)

    def _load_data(self, image_path):
        blob = {}
        blob['labels'] = cv2.imread(image_path['labels'])
        blob['pos'] = cv2.imread(image_path['pos'])
        blob['neg'] = cv2.imread(image_path['neg'])
        blob['pos_segm'] = cv2.imread(image_path['pos_segm'])
        blob['neg_segm'] = cv2.imread(image_path['neg_segm'])
        # stack = np.concatenate([blob['labels'][...,::-1], blob['pos'][...,::-1]],axis=1)
        # img = Image.fromarray(stack, 'RGB')
        # img.show()

        return blob

    def _get_data(self, image_path, training_format=False):
        """Returns data for one given image number from the specified sequence."""
        blob = self._load_data(image_path)

        return blob

    def _get_patch(self, blob, k, seed):
        modalities = list(blob.keys())
        tmp_blob = {}
        pos_blob = {}
        return_blob = {}
        img_h, img_w, _ = blob['labels'].shape
        dx_h = int(img_h/self.ppd)
        dx_w = int(img_w/self.ppd)

        p_h = int(k/self.ppd)
        p_w = k % self.ppd

        h_c = dx_h * p_h
        w_c = dx_w * p_w

        ref_patch = add_gaussian_noise(blob['labels'][h_c:h_c+dx_h, w_c:w_c+dx_w, ...],self.config['augmentation']['noise'][0],self.config['augmentation']['noise'][1])
        pos_blob['rgb'] = blob['pos'][h_c:h_c+dx_h, w_c:w_c+dx_w, ...]
        pos_segm = blob['pos_segm'][h_c:h_c+dx_h, w_c:w_c+dx_w, ...]

        # stack = np.concatenate([ref_patch[...,::-1], pos_blob['rgb'][...,::-1]],axis=1)
        # img = Image.fromarray(stack, 'RGB')
        # img.show()
        # input("Press Enter to continue...")
        # if k==1:
        #     assert(False)
        maxLoop = 20

        pos_blob = augmentate(pos_blob, brightness=self.config['augmentation']['brightness'], seed=seed)
        tmp_blob['rgb'] = blob['neg']
        tmp_blob['neg_segm'] = blob['neg_segm']

        counter = 0
        while(True):
            rng_seed = np.random.randint(maxLoop)
            return_blob = deepcopy(tmp_blob)
            return_blob = augmentate(return_blob,vflip=self.config['augmentation']['vflip'],
                                                 hflip=self.config['augmentation']['hflip'],
                                                 crop=[1, dx_h],
                                                 brightness=self.config['augmentation']['brightness'],
                                                 contrast=self.config['augmentation']['contrast'],
                                                 gamma=self.config['augmentation']['gamma'], seed=counter+1+seed)

            tmp = (return_blob['neg_segm'] == pos_segm)
            sameMask = np.logical_and(np.logical_and(tmp[:,:,0],tmp[:,:,1]),tmp[:,:,2])

            if(np.sum(sameMask)/sameMask.size < 0.5 or counter is maxLoop):
                if(counter == maxLoop):
                    print("Reached counter %d in finding neg patch loop" % maxLoop)
                break
            counter += 1

        return_blob['labels'] = ref_patch
        return_blob['pos'] = pos_blob['rgb']
        return_blob['pos_segm'] = pos_segm
        return_blob['neg'] = return_blob['rgb']
        return_blob.pop('rgb',None)

        return return_blob

    def _get_tf_dataset(self, setlist, training_format=False):
        def data_generator():
            for i in range(len(setlist['labels'])):
                item = {'labels': setlist['labels'][i],
                        'pos': setlist['pos'][i],
                        'neg': setlist['neg'][i],
                        'pos_segm': setlist['pos_segm'][i],
                        'neg_segm': setlist['neg_segm'][i]}

                data = self._get_data(item, training_format=training_format)
                for k in range(self.ppd*self.ppd):
                    patch_data = self._get_patch(data,k,(k*len(setlist['labels'])+i))

                    yield patch_data

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
