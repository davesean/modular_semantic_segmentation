from .synthia import Synthia
from .raw_synthia import Synthia as SynthiaRaw
from .synthia_cityscapes import SynthiaCityscapes
from .cityscapes import Cityscapes
from .cityscapes_GAN import Cityscapes_GAN
from .cityscapes_cascGAN import Cityscapes_cascGAN
from .cityscapes_generated import Cityscapes_generated
from .synthia_rand import SynthiaRand
from .mixed_data import MixedData
from .not_cityscapes import AddRandomObjects
from .pos_neg import POSNEG
from .wilddash import Wilddash


def get_dataset(name):
    if name == 'synthia':
        return Synthia
    elif name == 'not_cityscapes':
        return AddRandomObjects
    elif name == 'raw_synthia':
        return SynthiaRaw
    elif name == 'synthia_cityscapes':
        return SynthiaCityscapes
    elif name in ['cityscapes', 'cityscapes_c']:
        return Cityscapes
    elif name in ['cityscapes_GAN']:
        return Cityscapes_GAN
    elif name in ['cityscapes_cascGAN']:
        return Cityscapes_cascGAN
    elif name in ['cityscapes_generated']:
        return Cityscapes_generated
    elif name == 'synthiarand':
        return SynthiaRand
    elif name == 'posneg':
        return POSNEG
    elif name == 'wilddash':
        return Wilddash
    if name == 'mixeddata':
        return MixedData
    else:
        raise UserWarning('ERROR: Dataset {} not found'.format(name))
