import numpy as np
from os import listdir, path, environ
import cv2
import tarfile
from sklearn.model_selection import train_test_split

from xview.settings import DATA_BASEPATH
from .data_baseclass import DataBaseclass
from .augmentation import augmentate
from copy import deepcopy


POSNEG_BASEPATH = path.join(DATA_BASEPATH, 'posneg')

class POSNEG(DataBaseclass):
    _data_shape_description = {
            'rgb': (None, None, 3), 'labels': (None, None), 'mask': (None, None) }
    _num_default_classes = 2

    def __init__(self, base_path=POSNEG_BASEPATH, batchsize=1, **data_config):

        config = {
            'augmentation': {
                'crop': False, #[1, 240],
                'scale': False, #[.4, 1, 1.5],
                'vflip': False,
                'hflip': False,
                'gamma': False, #[.4, 0.3, 1.2],
                'rotate': False,
                'shear': False,
                'contrast': False, #[.3, 0.5, 1.5],
                'brightness': False, #[.2, -40, 40]
            },
            'resize': True
        }
        config.update(data_config)
        self.config = config

        if not path.exists(base_path):
            message = 'ERROR: Path to posneg dataset does not exist.'
            print(message)
            raise IOError(1, message, base_path)

        self.base_path = base_path
        self.image_prefixes = {
                'pos': 'pos',
                'neg': 'neg'
        }
        # get all the filenames with the two given prefixes
        def get_filenames(prefix1,prefix2):
            filenames = []
            filenames.extend(
                [{'image_path': prefix1+str(n)+'.png'}
                 for n in range(1,10)])
            filenames.extend(
                [{'image_path': prefix2+str(n)+'.png'}
                 for n in range(1,10)])
            return filenames

        validationset = get_filenames(prefix1=self.image_prefixes['pos'],
                                      prefix2=self.image_prefixes['neg'])

        # Intitialize Baseclass
        DataBaseclass.__init__(self, None, None, None, None, validation_set=validationset)

    def _load_data(self, image_path):
        blob = {}
        blob['rgb'] = cv2.imread(image_path)
        prefix = image_path.split('/')[-1]
        # Depending on if the image loaded is assumed positive (0, in-dist)
        # or negative (1, OoD), the mask generated needs to reflect that
        if prefix[0:len(self.image_prefixes['pos'])] == self.image_prefixes['pos']:
            blob['mask'] = np.zeros((blob['rgb'].shape[0],blob['rgb'].shape[1]))
        elif prefix[0:len(self.image_prefixes['neg'])] == self.image_prefixes['neg']:
            blob['mask'] = np.ones((blob['rgb'].shape[0],blob['rgb'].shape[1]))
        else:
            assert(False, prefix[0:2]+" was not an image prefix")
        # Resize the image to the necessary size, hard coded below!
        if self.config['resize']:
            blob['rgb'] = cv2.resize(blob['rgb'], (256, 256),
                                     interpolation=cv2.INTER_LINEAR)
            for m in ['mask']:
                blob[m] = cv2.resize(blob[m], (256, 256),
                                     interpolation=cv2.INTER_NEAREST)
        blob['labels'] = np.zeros((blob['mask'].shape[0],blob['mask'].shape[1]))
        return blob

    def _get_data(self, image_path):
        """Returns data for one given image number from the specified sequence."""
        blob = self._load_data(image_path)
        return blob
