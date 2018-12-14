from xview.settings import DATA_BASEPATH
from os import path, environ
from xview.datasets import Cityscapes
import cv2
import numpy as np
from tqdm import tqdm
import tarfile
from .augmentation import augmentate
from .data_baseclass import DataBaseclass

def get_dataset(name):
    """have to reimplement as there is an import loop when using __init__"""
    if name == 'cityscapes':
        return Cityscapes


class AddRandomObjects(DataBaseclass):

    _data_shape_description = {'rgb': (None, None, 3), 'labels': (None, None)}
    _num_default_classes = 2

    def __init__(self, add_to_dataset='cityscapes', halfsize=False, animal='horse', augmentation=False,
                 in_memory=False, **data_config):
        self.base_path = path.join(DATA_BASEPATH, 'pascalvoc')
        if not path.exists(self.base_path):
            message = 'ERROR: Path to PascalVOC dataset does not exist.'
            print(message)
            raise IOError(1, message, self.base_path)

        config = {
            'in_memory': in_memory,
            'resize': False,
            'augmentation': {
                'obj_scale': [0.4, 0.7],
                'rotation': True,
                'crop': False,
                'scale': False,
                'vflip': False,
                'hflip': False,
                'gamma': False,
                'rotate': False,
                'shear': False,
                'contrast': False,
                'brightness': False
            }

        }

        config.update(data_config)
        self.config = config

        def _get_filenames(self):

            image_txt_path = path.join(self.base_path, 'ImageSets','Main',self.animal+'_trainval.txt')
            segm_txt_path = path.join(self.base_path, 'ImageSets','Segmentation','trainval.txt')

            image_txt = open(image_txt_path, "r")
            image_text = image_txt.readlines()

            segm_txt = open(segm_txt_path, "r")
            segm_text = segm_txt.readlines()

            str_list = []
            for lines in image_text:
                 tmp = int(lines.split(' ')[-1].split('\n')[0])
                 if tmp == 1:
                     str_list.append(lines.split(' ')[0])
            rtn_list = []
            for lines in segm_text:
                 for nums in str_list:
                     if lines.split('\n')[0] == nums:
                         rtn_list.append(nums)

            return rtn_list

        print('INFO: Loading Base Dataset')
        self.base_dataset = get_dataset(add_to_dataset)(**config)

        if in_memory:
            print('INFO loading dataset into memory')
            # first load the tarfile into a closer memory location, then load all the
            # images
            tar = tarfile.open(path.join(self.base_path, 'pascalvoc.tar.gz'))
            localtmp = environ['TMPDIR']
            tar.extractall(path=localtmp)
            tar.close()
            self.base_path = localtmp

        self.animal = animal
        self.file_list = _get_filenames(self)
        self.objects = {num: self._load_object(file_name)
                        for num, file_name in enumerate(self.file_list)}
        print('INFO loaded and preprocessed objects')


        DataBaseclass.__init__(self, self.base_dataset.trainset,
                               self.base_dataset.measureset,
                               self.base_dataset.testset,
                               {0: {'name': 'in-distribution'},
                                1: {'name': 'out-of-distribution'}},
                               validation_set=self.base_dataset.validation_set,
                               num_classes=self.base_dataset._num_default_classes)

    def _load_object(self, object_name):
        obj = cv2.imread(path.join(self.base_path, 'JPEGImages',object_name+'.jpg'))
        segm = cv2.imread(path.join(self.base_path, 'SegmentationClass',object_name+'.png'))

        target_color = [128,0,192]
        ones_mat = np.ones((segm.shape[0],segm.shape[1]))

        d_mat = np.dstack((ones_mat*target_color[0],
                           ones_mat*target_color[1],
                           ones_mat*target_color[2]))

        inv_mask = ~(np.sum(segm - d_mat,axis=2) == 0)
        inv_mask_3c = np.dstack((inv_mask,inv_mask,inv_mask))

        obj[inv_mask_3c] = 0

        # if self.config['halfsize']:
        #     h, w, _ = obj.shape
        #     cut_obj = cv2.resize(cut_obj, (h // 2, w // 2))
        #     mask = cv2.resize(mask, (h // 2, w // 2))
        return obj, (~inv_mask).astype(int)

    def _get_data(self, training_format=False, **kwargs):
        # load image from base dataset
        img = self.base_dataset._get_data(training_format=False, **kwargs)['rgb']

        labels = np.zeros((img.shape[0],img.shape[1]))

        # get a random object
        num = np.random.randint(0, len(self.file_list))
        obj, mask = self.objects[num]

        obj = cv2.resize(obj, (img.shape[0],img.shape[1]))
        mask = cv2.resize(mask, (img.shape[0],img.shape[1]),interpolation=cv2.INTER_NEAREST)

        if self.config['augmentation']['obj_scale'] is not None:
            scaling = np.random.uniform(self.config['augmentation']['obj_scale'][0],
                                        self.config['augmentation']['obj_scale'][1])
            h = int(obj.shape[0]*scaling)
            w = int(obj.shape[1]*scaling)

            obj = cv2.resize(obj, (h, w))
            mask = cv2.resize(mask, (h, w),interpolation=cv2.INTER_NEAREST)
        if self.config['augmentation']['rotation']:
            rotation = np.random.randint(0,359)
            h, w, _ = obj.shape
            M = cv2.getRotationMatrix2D((w/2,h/2),rotation,1)
            obj = cv2.warpAffine(obj,M,(w,h))
            mask = cv2.warpAffine(mask,M,(w,h),flags=cv2.INTER_NEAREST)
        h, w, _ = obj.shape

        # sample a random location where to put the object in the image
        img_h, img_w, _ = img.shape
        top = np.random.randint(img_h - h)
        left = np.random.randint(img_w - w)
        # create an overlay image with the object of the same size as the underlying
        # image
        obj = cv2.copyMakeBorder(obj, top, img_h - top - h, left, img_w - left - w,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))

        mask = cv2.copyMakeBorder(mask, top, img_h - top - h, left, img_w - left - w,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))

        img_mod = np.dstack((np.where(mask,obj[:,:,0],img[:,:,0]),
                          np.where(mask,obj[:,:,1],img[:,:,1]),
                          np.where(mask,obj[:,:,2],img[:,:,2])))

        blob = {
            'rgb': img_mod,
            'labels': mask
        }
        return blob
