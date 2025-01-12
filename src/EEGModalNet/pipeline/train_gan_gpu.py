import os
os.environ['KERAS_BACKEND'] = 'torch'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import keras
from keras.optimizers.schedules import ExponentialDecay
from torch.utils.data import DataLoader, TensorDataset
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
              bandpass_filter: List[int] = [1, 42],
              time_dim: int = 1024,
              exclude_sub_ids=None) -> tuple:

    xarray = xr.open_dataarray(data_path, engine='h5netcdf')
    x = xarray.sel(subject=xarray.subject[:n_subjects], channel=channels)

    if exclude_sub_ids is not None:
        x = x.sel(subject=~x.subject.isin(exclude_sub_ids))

    x = x.to_numpy()
    n_subjects = x.shape[0]

    if bandpass_filter is not None:
        sos = butter(4, bandpass_filter, btype='band', fs=128, output='sos')
        x = sosfiltfilt(sos, x, axis=-1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.tensor(x.copy(), device=device).unfold(2, time_dim, time_dim).permute(0, 2, 3, 1).flatten(0, 1)  # TODO: copy was added because of an error, look into this
    sub = torch.tensor(np.arange(0, n_subjects).repeat(x.shape[0] // n_subjects)[:, np.newaxis], device=device)

    ch_ind = find_channel_ids(xarray, channels)
    pos = torch.tensor(xarray.ch_positions[ch_ind][None].repeat(x.shape[0], 0), device=device)

    data = {'x': x, 'sub': sub, 'pos': pos}

    return data, n_subjects


def run(data,
        n_subjects,
        max_epochs=100_000,
        latent_dim=64,
        batch_size=64,
        cvloger_path='tmp/tmp/simple_gan_v1.csv',
        model_path='tmp/tmp/wgan_v2.model.keras',
        reuse_model=False,
        reuse_model_path=None):

    model = WGAN_GP(time_dim=512, feature_dim=data['x'].shape[-1],
                    latent_dim=latent_dim, n_subjects=n_subjects,
                    use_sublayer_generator=False,
                    use_sublayer_critic=False,
                    use_channel_merger_g=True,
                    use_channel_merger_c=True,
                    kerner_initializer='random_normal',
                    interpolation='bilinear')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f'>>>> Model is on {device}')

    if reuse_model:
        print(reuse_model_path)
        model.load_weights(reuse_model_path)

    lr_schedule_g = ExponentialDecay(0.00001, decay_steps=10000, decay_rate=0.96, staircase=True)
    lr_schedule_d = ExponentialDecay(0.00001, decay_steps=1000, decay_rate=0.96, staircase=True)

    model.compile(d_optimizer=keras.optimizers.Adam(lr_schedule_d, beta_1=0.5, beta_2=0.9),
                  g_optimizer=keras.optimizers.Adam(0.00005, beta_1=0.5, beta_2=0.9),
                  gradient_penalty_weight=0)

    torch.cuda.synchronize()  # wait for model to be loaded

    history = model.fit(data,
                        batch_size=batch_size,
                        epochs=max_epochs,
                        shuffle=True,
                        callbacks=[
                            CustomModelCheckpoint(model_path, save_freq=5),
                            keras.callbacks.ModelCheckpoint(f'{model_path}_best_gloss.model.keras', monitor='g_loss', save_best_only=True),
                            keras.callbacks.ModelCheckpoint(f'{model_path}_best_dloss.model.keras', monitor='d_loss', save_best_only=True),
                            keras.callbacks.CSVLogger(cvloger_path),
                        ])

    return model, history


if __name__ == '__main__':
    data, n_subs = load_data('data/LEMON_DATA/eeg_EC_BaseCorr_Norm_Clamp_with_pos.nc5',
                             n_subjects=202,
                             channels=['O1', 'O2', 'Fp1', 'Fp2', 'C1', 'C2', 'P1', 'P2'],
                             bandpass_filter=[1, 42],
                             time_dim=1024,
                             exclude_sub_ids=None)

    if torch.cuda.is_available():
        print('GPU is available')
        torch.cuda.current_device()
    else:
        print('GPU is not available!!')
        exit()

    print(f'Running on {torch.cuda.device_count()} GPUs')
    print(f'Using CUDA device: {torch.cuda.get_device_name(0)}')

    # Explicitly set the CUDA device
    torch.cuda.set_device(0)

    # preload CUDA libraries with a dummy tensor
    _ = torch.randn(1, device="cuda")

    # Apply mixed precision policy
    keras.mixed_precision.set_global_policy('mixed_float16')
    print(f'Global policy is {keras.mixed_precision.global_policy().name}')

    output_path = 'logs/12.01.2025'

    model, _ = run(data,
                   n_subjects=n_subs,
                   max_epochs=280,
                   latent_dim=128,
                   batch_size=128,
                   cvloger_path=f'{output_path}.csv',
                   model_path=output_path,
                   reuse_model=False,
                   reuse_model_path=None)

    # # backup
    # model.save(f'{output_path}_final.model.keras')
