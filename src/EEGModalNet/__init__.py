import os
os.environ['KERAS_BACKEND'] = 'torch'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from .models.WGAN import WGAN_GP
from .utils import ProgressBarCallback, CustomModelCheckpoint
from .preprocessing.utils import get_averaged_data
from .models.common import SubjectLayers_v2, convBlock, ChannelMerger, ResidualBlock, build_eeg_transformer, Classifier
from .data.hpc_data_loader import load_data

__all__ = ['WGAN_GP',
           'SubjectLayers_v2', 'convBlock', 'ChannelMerger', 'ResidualBlock',
           'build_eeg_transformer', 'ProgressBarCallback', 'CustomModelCheckpoint',
           'get_averaged_data', 'Classifier', 'load_data']
