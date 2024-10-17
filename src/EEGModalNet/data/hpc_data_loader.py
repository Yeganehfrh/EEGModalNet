import os
import numpy as np
import xarray as xr
import pandas as pd
os.environ['KERAS_BACKEND'] = 'torch'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU usage
from sklearn.model_selection import StratifiedGroupKFold
from scipy.signal import butter, sosfilt

from ...EEGModalNet import get_averaged_data

import torch
import xarray as xr
import pandas as pd


def load_data(eeg_data_path,
              session_data_path,
              channels,
              time_dim=512,
              n_subject=51,
              n_splits=5,
              highpass_filter=True,
              average_over_channels=True,
              only_labelled_hypnosis=True):

    EEG_data = xr.open_dataarray(eeg_data_path, engine='h5netcdf')

    # open session data
    session_data = pd.read_csv(session_data_path)

    if average_over_channels:
        EEG_data = EEG_data.sel(subject=EEG_data.subject[:n_subject])
        X_input, channels = get_averaged_data(EEG_data)

    else:
        X_input = EEG_data.sel(subject=EEG_data.subject[:n_subject], channel=channels).to_numpy()

    # including only hypnosis sessions
    if only_labelled_hypnosis:
        print('>>>>> Including only labelled hypnosis sessions')
        session_data = session_data.query('description == "hypnosis"')[['bids_id', 'session',
                                                                        'cluster_small']].set_index('bids_id')
        X_input_hyp = np.zeros([52, 2, len(channels), X_input.shape[-1]])
        for i in range(X_input.shape[0]):
            ses = session_data.loc[i+1, 'session'].values - 1
            X_input_hyp[i] = X_input[i, ses, :, :]
        X_input = X_input_hyp
        del X_input_hyp

    # preparing x
    if highpass_filter:
        sos = butter(4, 1, btype='high', fs=128, output='sos')
        X_input = sosfilt(sos, X_input, axis=-1)

    X_input = torch.tensor(X_input.squeeze()).unfold(-1, time_dim, time_dim).permute(0, 1, 3, 4, 2).flatten(0, 1)

    # remove missing sessions for sub-52 depending on which sessions are included
    remove_sess = -2 if only_labelled_hypnosis else [-2, -3]
    X_input = np.delete(X_input, remove_sess, axis=0)
    remove_ind = 102 if only_labelled_hypnosis else [205, 206]
    session_data = session_data.reset_index().drop(index=remove_ind)

    # prepare y
    y = session_data['cluster_small']
    groups = session_data['bids_id'].values
    y = y.values

    # prepare Kfold cross validation
    group_kfold = StratifiedGroupKFold(n_splits=n_splits)
    train_val_splits = []
    for train_idx, val_idx in group_kfold.split(X_input, y, groups=groups):
        train_val_splits.append((train_idx, val_idx))

    # make sure that the splits are stratified and ther is a balance between the classes
    for train_idx, val_idx in train_val_splits:
        print(np.unique(y[train_idx], return_counts=True)[1] / len(y[train_idx]), np.unique(y[val_idx], return_counts=True)[1] / len(y[val_idx]))

    y = torch.tensor(y).reshape(-1, 1).repeat(1, X_input.shape[1])

    return X_input, y, train_val_splits, channels
