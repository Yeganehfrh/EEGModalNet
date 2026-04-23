import os
os.environ['KERAS_BACKEND'] = 'torch'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from .models.WGAN import WGAN_GP
from .models.WGAN_v0 import WGAN_GP_V0
from .models.TCN_WGAN import TCNWGAN
from .models.FiLMGAN import FiLMGAN
from .utils.utils import ProgressBarCallback, CustomModelCheckpoint, StepLossHistory, BalancedAccuracy
from .preprocessing.utils import get_averaged_data
from .models.common import SubjectLayers_v2, convBlock, ChannelMerger, ResidualBlock, build_eeg_transformer
from .data.hpc_data_loader import load_data, RandomCropEEGDataset
from .preprocessing.preprocessing import preprocess_data
from .utils.extractor import extract_features_batched, extract_features_batched_deterministic

__all__ = ['WGAN_GP', 'WGAN_GP_V0', 'TCNWGAN', 'FiLMGAN', 'RandomCropEEGDataset',
           'SubjectLayers_v2', 'convBlock', 'ChannelMerger', 'ResidualBlock',
           'build_eeg_transformer', 'ProgressBarCallback', 'CustomModelCheckpoint',
           'get_averaged_data', 'load_data', 'StepLossHistory', 'preprocess_data',
           'extract_features_batched', 'extract_features_batched_deterministic',
           'BalancedAccuracy']
