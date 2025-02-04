import os
os.environ['KERAS_BACKEND'] = 'torch'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import keras
from ...EEGModalNet import WGAN_GP
from ...EEGModalNet import CustomModelCheckpoint
from typing import List
import numpy as np
import xarray as xr
from scipy.signal import butter, sosfiltfilt


def find_channel_ids(dataarray, ch_names):
    return [i for i, ch in enumerate(dataarray.channel.to_numpy()) if ch in ch_names]


def load_data(data_path: str,
              n_subjects: int = 202,
              channels: List[str] = ['all'],
              highpass_filter: float = 0.5,
              time_dim: int = 1024,
              exclude_sub_ids=None) -> tuple:

    xarray = xr.open_dataarray(data_path, engine='h5netcdf')
    x = xarray.sel(subject=xarray.subject[:n_subjects], channel=channels)

    if exclude_sub_ids is not None:
        x = x.sel(subject=~x.subject.isin(exclude_sub_ids))

    x = x.to_numpy()
    n_subjects = x.shape[0]

    if highpass_filter is not None:
        sos = butter(4, highpass_filter, btype='high', fs=128, output='sos')
        x = sosfiltfilt(sos, x, axis=-1)

    x = torch.tensor(x.copy()).unfold(2, time_dim, time_dim).permute(0, 2, 3, 1).flatten(0, 1)  # TODO: copy was added because of an error, look into this

    sub = np.arange(0, n_subjects).repeat(x.shape[0] // n_subjects)[:, np.newaxis]

    # ch_ind = find_channel_ids(xarray, channels)
    # pos = xarray.ch_positions[ch_ind][None].repeat(x.shape[0], 0)
    pos = torch.tensor(xarray.ch_positions[None].repeat(x.shape[0], 0))

    return {'x': x, 'sub': torch.tensor(sub), 'pos': pos}, n_subjects


def run(data,
        n_subjects,
        max_epochs=100_000,
        latent_dim=64,
        batch_size=64,
        cvloger_path='tmp/simple_gan_v1.csv',
        model_path='tmp/wgan_v2',
        reuse_model=False,
        reuse_model_path=None):

    model = WGAN_GP(time_dim=512, feature_dim=data['x'].shape[-1],
                    latent_dim=latent_dim, n_subjects=n_subjects,
                    use_sublayer_generator=True,
                    use_sublayer_critic=True,
                    use_channel_merger_c=False,
                    use_channel_merger_g=False,
                    interpolation='bilinear')

    model.compile(d_optimizer=keras.optimizers.Adam(0.0000940, beta_1=0.5, beta_2=0.9),
                  g_optimizer=keras.optimizers.Adam(0.0000940, beta_1=0.5, beta_2=0.9),
                  gradient_penalty_weight=10.0)

    callbacks = [CustomModelCheckpoint(model_path, save_freq=50),
                 keras.callbacks.CSVLogger(cvloger_path, append=True),
                 keras.callbacks.ModelCheckpoint(model_path + 'best_gloss.model.keras', monitor='g_loss', save_best_only=True),
                 keras.callbacks.TerminateOnNaN()]

    _ = model.fit(data,
                  epochs=max_epochs,
                  batch_size=batch_size,
                  callbacks=callbacks)

    return model


if __name__ == '__main__':
    data, n_subs = load_data('data/LEMON_DATA/EC_8_channels_processed.nc5',
                             n_subjects=202, channels=['O1', 'O2', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2'],
                             highpass_filter=0.5, time_dim=512,
                             )

    output_path = 'logs/multichannel_test_08-01-2025'

    max_epochs = 2000
    latent_dim = 128
    run(data,
        n_subs,
        max_epochs=max_epochs,
        latent_dim=latent_dim,
        batch_size=128,
        cvloger_path=f'{output_path}.csv',
        model_path=output_path,
        reuse_model=False,
        reuse_model_path=None)
