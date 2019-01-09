import numpy as np
from os import listdir, path, environ
import cv2
import tarfile
from sklearn.model_selection import train_test_split

from xview.settings import DATA_BASEPATH
from .data_baseclass import DataBaseclass
from .augmentation import augmentate
from copy import deepcopy


WILDDASH_BASEPATH = path.join(DATA_BASEPATH, 'wilddash')

class Wilddash(DataBaseclass):

    _data_shape_description = {
            'rgb': (None, None, 3), 'labels': (None, None)}
    _num_default_classes = 12

    def __init__(self, base_path=WILDDASH_BASEPATH, batchsize=1, in_memory=False, **data_config):
        # This is a validation set, no augmentation should be applied
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
            message = 'ERROR: Path to WILDDASH dataset does not exist.'
            print(message)
            raise IOError(1, message, base_path)

        self.base_path = base_path
        self.modality_paths = {
                'rgb': 'wd_val_01/wd_val_01',
                'labels': 'wd_val_01/wd_val_01',
        }
        self.modality_suffixes = {
                'rgb': '_100000',
                'labels': '_100000_labelIds',
        }
        self.in_memory = in_memory
        # All labelinfo should be the same as in the cityscapes file
        original_labelinfo = {
                0: {'name': 'unlabeled', 'mapping': 'void'},
                1: {'name': 'ego vehicle', 'mapping': 'void'},
                2: {'name': 'rectification border', 'mapping': 'void'},
                3: {'name': 'out of roi', 'mapping': 'void'},
                4: {'name': 'static', 'mapping': 'void'},
                5: {'name': 'dynamic', 'mapping': 'void'},
                6: {'name': 'ground', 'mapping': 'void'},
                7: {'name': 'road', 'mapping': 'road'},
                8: {'name': 'sidewalk', 'mapping': 'sidewalk'},
                9: {'name': 'parking', 'mapping': 'road'},
                10: {'name': 'rail track', 'mapping': 'void'},
                11: {'name': 'building', 'mapping': 'building'},
                12: {'name': 'wall', 'mapping': 'building'},
                13: {'name': 'fence', 'mapping': 'fence'},
                14: {'name': 'guard rail', 'mapping': 'void'},
                15: {'name': 'bridge', 'mapping': 'void'},
                16: {'name': 'tunnel', 'mapping': 'void'},
                17: {'name': 'pole', 'mapping': 'pole'},
                18: {'name': 'polegroup', 'mapping': 'void'},
                19: {'name': 'traffic light', 'mapping': 'void'},
                20: {'name': 'traffic sign', 'mapping': 'traffic sign'},
                21: {'name': 'vegetation', 'mapping': 'vegetation'},
                22: {'name': 'terrain', 'mapping': 'vegetation'},
                23: {'name': 'sky', 'mapping': 'sky'},
                24: {'name': 'person', 'mapping': 'person'},
                25: {'name': 'rider', 'mapping': 'person'},
                26: {'name': 'car', 'mapping': 'vehicle'},
                27: {'name': 'truck', 'mapping': 'vehicle'},
                28: {'name': 'bus', 'mapping': 'vehicle'},
                29: {'name': 'caravan', 'mapping': 'vehicle'},
                30: {'name': 'trailer', 'mapping': 'vehicle'},
                31: {'name': 'train', 'mapping': 'vehicle'},
                32: {'name': 'motorcycle', 'mapping': 'vehicle'},
                33: {'name': 'bike', 'mapping': 'bicycle'}
        }

        labelinfo = {
            0: {'name': 'void', 'color': [0, 0, 0]},
            1: {'name': 'sky', 'color': [70, 130, 180]},
            2: {'name': 'building', 'color': [70, 70, 70]},
            3: {'name': 'road', 'color': [128, 64, 128]},
            4: {'name': 'sidewalk', 'color': [244, 35, 232]},
            5: {'name': 'fence', 'color': [190, 153, 153]},
            6: {'name': 'vegetation', 'color': [107, 142, 35]},
            7: {'name': 'pole', 'color': [153, 153, 153]},
            8: {'name': 'vehicle', 'color': [0,  0, 142]},
            9: {'name': 'traffic sign', 'color': [220, 220, 0]},
            10: {'name': 'person', 'color': [220, 20, 60]},
            11: {'name': 'bicycle', 'color': [119, 11, 32]}
        }

        self.label_lookup = [next(i for i in labelinfo
                                  if labelinfo[i]['name'] == k['mapping'])
                             for _, k in original_labelinfo.items()]

        # load training and test sets
        # Generate train/test splits
        def get_filenames():
            filenames = []
            base_dir = path.join(self.base_path, self.modality_paths['rgb'])

            filenames.extend(
                [{'image_path': path.join(
                    base_dir, path.splitext(n)[0].split('_')[0])}
                 for n in listdir(base_dir)])
            # As there are 4 files per sample, we need to only need the prefix once
            unique_filenames = list({v['image_path']:v for v in filenames}.values())
            return unique_filenames
        # in_memory=True not tested, as we are only using a subset
        if self.in_memory and 'TMPDIR' in environ:
            print('INFO loading dataset into machine ... ', end='')
            # first load the tarfile into a closer memory location, then load all the
            # images
            tar = tarfile.open(path.join(base_path, 'wd_val_01.zip'))
            localtmp = environ['TMPDIR']
            tar.extractall(path=localtmp)
            tar.close()
            self.base_path = localtmp
            self.images = {}
            print('DONE')
        elif self.in_memory:
            print('INFO Environment Variable TMPDIR not set, could not unpack data '
                  'and load into memory\n'
                  'Now trying to load every image seperately')

        validationset = get_filenames()

        # Intitialize Baseclass with only a validation set
        DataBaseclass.__init__(self, None, None, None, labelinfo, validation_set=validationset)

    def _load_data(self, image_path):
        rgb_filename, labels_filename = (
            image_path+self.modality_suffixes[str(m)]+'.png'
            for m in ['rgb', 'labels']
        )

        blob = {}
        blob['rgb'] = cv2.imread(rgb_filename)
        blob['labels'] = cv2.imread(labels_filename, cv2.IMREAD_ANYDEPTH)
        # apply label mapping
        blob['labels'] = np.asarray(self.label_lookup, dtype='int32')[blob['labels']]

        if self.config['resize']:
            blob['rgb'] = cv2.resize(blob['rgb'], (256, 256),
                                     interpolation=cv2.INTER_LINEAR)
            for m in ['labels']:
                blob[m] = cv2.resize(blob[m], (256, 256),
                                     interpolation=cv2.INTER_NEAREST)
        return blob

    def _get_data(self, image_path, training_format=False):
        """Returns data for one given image number from the specified sequence."""
        if self.in_memory and 'TMPDIR' in environ:
            if image_path not in self.images:
                self.images[image_path] = self._load_data(image_path)
            image = self.images[image_path]
            blob = {}
            for m in image:
                blob[m] = image[m].copy()
        else:
            blob = self._load_data(image_path)
        return blob
