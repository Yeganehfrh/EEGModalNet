# =============================================================================
import pandas as pd
from pathlib import Path
from typing import List, Union
import torch
from mne.io import read_raw_eeglab
from scipy.signal import butter, sosfiltfilt
from sklearn.preprocessing import RobustScaler
from torch import nn


class EEGProcessing(nn.Module):
    def __init__(self,
                 baseline_correction: float = 0.5,
                 scaling: bool = True,
                 clamping: int = 20,
                 rereference: str | None = None,
                 filtering: List[float] | None = None,
                 **kwargs):
        super().__init__()

        self.baseline_correction = baseline_correction
        self.scaling = scaling
        self.clamping = clamping
        self.rereferencing = rereference
        self.filter = filtering
        self.kwargs = kwargs


def forward(self, X, y=None):
    if self.baseline_correction is not None:
        self.X = correct_baseline(self.X)
    if self.scaling:
        self.X = robust_scaling(self.X)
    if self.clamping is not None:
        self.X = clamp(self.X, clamp=clamp)
    if self.rereference is not None:
        self.X = rereferencing(self.X, rereferencing=self.rereferencing)
    if self.filtering is not None:
        self.X = bandpass_filter(self.X, bandpass=filter)

    return self.X


def bandpass_filter(x, bandpass, sf=128):
    """Bandpass filter input tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    bandpass : list
        Bandpass frequencies
    sf : int
        Sampling frequency, defaults to 128

    Returns
    -------
    torch.Tensor
        Filtered tensor
    """

    print(f'>>>>>> Filtering data with bandpass filter {bandpass} Hz')
    sos = butter(4, bandpass, 'bp', sf=sf, output='sos')
    x = torch.from_numpy(sosfiltfilt(sos, x, axis=1).copy()).float()
    return x


def correct_baseline(x, sf=128, duration=0.5):
    """Baseline correction of input tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    sf : int
        Sampling frequency, defaults to 128

    Returns
    -------
    torch.Tensor
        Baseline corrected tensor
    """

    print(f'>>>>>> Baseline correction with mean over first {duration} s')
    x = x - x[:, :, :int(sf * duration)].mean(axis=2, keepdims=True)
    return x


def robust_scaling(x):
    """Robust scaling of input tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor

    Returns
    -------
    torch.Tensor
        Scaled tensor
    """

    print('>>>>>> Robust scaling')
    x = RobustScaler().fit_transform(x)  # TODO: robust_scale(x, axis=0)
    return x


def clamp_EEG(x, dev=20):
    """Clamp input tensor proportional to the standard deviation in each
    channel and for each participant.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    clamp : int
        Clamping value, defaults to 20

    Returns
    -------
    torch.Tensor
        Clamped tensor
    """

    print('>>>>>> Clamping data')
    x = torch.clamp(x, -clamp, clamp)
    return x


def rereferencing(x, rereferencing='average'):
    """Rereferencing of input tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    rereferencing : str
        Rereferencing method, defaults to 'average'

    Returns
    -------
    torch.Tensor
        Rereferenced tensor
    """

    print('>>>>>> Rereferencing data')
    if rereferencing == 'average':
        x = x - x.mean(axis=1, keepdims=True)
    elif rereferencing == 'common':
        x = x - x.mean(axis=0, keepdims=True)
    return x


def find_excluded_channels(data_path: str, full_ch_list: List[str], pattern: str):
    exc_chs = {}
    for i in Path(data_path).glob(pattern):
        raw = read_raw_eeglab(i, verbose=False)
        sub = i.stem[:-3]
        excluded_chs = [i for i in full_ch_list if i not in raw.ch_names]
        exc_chs[sub] = [excluded_chs]
        df = pd.DataFrame.from_dict(exc_chs, orient='index', columns=['excluded_channels'])
        # replace the empty list with None
        df['bad_channels'] = df['bad_channels'].apply(lambda x: None if x == [] else x)
    return df
