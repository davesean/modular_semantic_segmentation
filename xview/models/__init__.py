from .simple_fcn import SimpleFCN
from .bayes_mix import BayesFusion
from .dirichlet_mix import DirichletFusion
from .average_mix import AverageFusion
from .variance_mix import VarianceFusion
from .adapnet import Adapnet
from .fusion_fcn import FusionFCN
from .cGAN import pix2pix
from .cycleGAN import cycleGAN
from .cascGAN import cascRef
from .similarityDiscriminator import DiffDiscrim


def get_model(name):
    if name == 'fcn':
        return SimpleFCN
    elif name == 'fusion_fcn':
        return FusionFCN
    elif name in ['bayes_mix', 'bayes_fusion']:
        return BayesFusion
    elif name in ['dirichlet_mix', 'dirichlet_fusion']:
        return DirichletFusion
    elif name in ['average_fusion', 'average_mix']:
        return AverageFusion
    elif name in ['variance_mix', 'variance_fusion']:
        return VarianceFusion
    elif name == 'adapnet':
        return Adapnet
    elif name == 'cGAN':
        return pix2pix
    elif name == 'cascGAN':
        return cascRef
    elif name == 'cycleGAN':
        return cycleGAN
    elif name == 'simDisc':
        return DiffDiscrim
    else:
        raise UserWarning('ERROR: Model %s not found' % name)
