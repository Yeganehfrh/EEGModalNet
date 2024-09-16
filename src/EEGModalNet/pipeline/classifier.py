import os
os.environ['KERAS_BACKEND'] = 'torch'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import keras
from ...EEGModalNet import CustomModelCheckpoint
from ...EEGModalNet import build_eeg_transformer
from tqdm.auto import tqdm
from typing import List
import numpy as np
import xarray as xr
from scipy.signal import butter, sosfilt
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


def find_channel_ids(dataarray, ch_names):
    return [i for i, ch in enumerate(dataarray.channel.to_numpy()) if ch in ch_names]


def load_data(data_path: str,
              n_subjects: int,
              channels: List[str] = ['all'],
              highpass_filter: float = 1,
              time_dim: int = 1024,
              stratified=True) -> tuple:

    xarray = xr.open_dataarray(data_path, engine='h5netcdf')
    xarray = xarray.sel(subject=xarray.subject[:n_subjects])
    x = xarray.to_numpy()

    if channels[0] != 'all':
        ch_ind = find_channel_ids(xarray, channels)
        x = x[:, ch_ind, 440:]

    if highpass_filter is not None:
        sos = butter(4, highpass_filter, btype='high', fs=128, output='sos')
        x = sosfilt(sos, x, axis=-1)

    x = torch.tensor(x).unfold(2, time_dim, time_dim).permute(0, 2, 3, 1)

    # output
    y = xarray.gender[:n_subjects]
    y -= 1
    y = torch.tensor(y).reshape(-1, 1).repeat(1, x.shape[1])
    y = F.one_hot(y, num_classes=2).float()

    train_ids, val_ids = train_test_split(np.arange(x.shape[0]), test_size=0.3, stratify=y[:, 0].numpy() if stratified else None)
    x_train, x_val = x[train_ids].flatten(0, 1), x[val_ids].flatten(0, 1)
    y_train, y_val = y[train_ids].flatten(0, 1), y[val_ids].flatten(0, 1)

    return x_train, x_val, y_train, y_val


def run(data,
        sequence_length,
        embed_dim,
        num_heads,
        ff_dim,
        n_layers,
        n_channels,
        max_epochs,
        cvloger_path,
        model_path,
        reuse_model=False,
        reuse_model_path=None):

    x_train, x_val, y_train, y_val = data

    model = build_eeg_transformer(sequence_length=sequence_length,
                                  embed_dim=embed_dim,
                                  num_heads=num_heads,
                                  ff_dim=ff_dim,
                                  num_layers=n_layers,
                                  n_channels=n_channels)

    if reuse_model:
        print(reuse_model_path)
        model.load_weights(reuse_model_path)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train,
                        y_train,
                        batch_size=64,
                        epochs=max_epochs,
                        shuffle=True,
                        callbacks=[CustomModelCheckpoint(model_path, save_freq=100),
                                   keras.callbacks.ModelCheckpoint(f'{model_path}_best_val_loss.model.keras', save_best_only=True),
                                   keras.callbacks.ModelCheckpoint(f'{model_path}_best_val_acc.model.keras', monitor='val_accuracy',
                                                                   save_best_only=True),
                                   keras.callbacks.CSVLogger(cvloger_path)],
                        validation_data=(x_val, y_val))
    return model, history


if __name__ == '__main__':

    channels = ['O1', 'F1', 'O2', 'F2', 'Fp1', 'Fp2', 'F3', 'Cz', 'P1', 'P2', 'Pz', 'C1', 'C2', 'Fz']
    input_path = 'data/LEMON_DATA/eeg_EC_BaseCorr_Norm_Clamp_with_pos.nc5'
    output_path = 'logs/models/classifier/transformer_16.09.2024'

    data = load_data(input_path,
                     n_subjects=202,
                     channels=channels,
                     highpass_filter=1,
                     time_dim=512,
                     stratified=False)

    model, _ = run(data,
                   sequence_length=512,
                   embed_dim=64,
                   num_heads=4,
                   ff_dim=128,
                   n_layers=4,
                   n_channels=len(channels),
                   max_epochs=1000,
                   cvloger_path=f'{output_path}.csv',
                   model_path=output_path,
                   reuse_model=False,
                   reuse_model_path=None)
    # backup
    model.save(f'{output_path}_final.model.keras')
