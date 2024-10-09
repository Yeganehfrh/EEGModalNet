import os
os.environ['KERAS_BACKEND'] = 'torch'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import keras
from ...EEGModalNet import WGAN_GP
from ...EEGModalNet import ProgressBarCallback, CustomModelCheckpoint
from tqdm.auto import tqdm
from typing import List
import numpy as np
import xarray as xr
from scipy.signal import butter, sosfilt


def find_channel_ids(dataarray, ch_names):
    return [i for i, ch in enumerate(dataarray.channel.to_numpy()) if ch in ch_names]


def load_data(data_path: str,
              n_subjects: int = 202,
              channels: List[str] = ['all'],
              highpass_filter: float = 1,
              time_dim: int = 1024,
              exclude_sub_ids=None) -> tuple:

    xarray = xr.open_dataarray(data_path, engine='h5netcdf')
    x = xarray.sel(subject=xarray.subject[:n_subjects], channels=channels)

    if exclude_sub_ids is not None:
        x = x.sel(subject=~x.subject.isin(exclude_sub_ids))

    x = x.to_numpy()
    n_subjects = x.shape[0]

    if highpass_filter is not None:
        sos = butter(4, highpass_filter, btype='high', fs=128, output='sos')
        x = sosfilt(sos, x, axis=-1)

    x = torch.tensor(x).unfold(2, time_dim, time_dim).permute(0, 2, 3, 1).flatten(0, 1)

    sub = np.arange(0, n_subjects).repeat(x.shape[0] // n_subjects)[:, np.newaxis]

    ch_ind = find_channel_ids(xarray, channels)
    pos = xarray.ch_positions[ch_ind][None].repeat(x.shape[0], 0)

    return {'x': x, 'sub': sub, 'pos': pos}, n_subjects


def run(data,
        n_subjects,
        max_epochs=100_000,
        latent_dim=64,
        cvloger_path='tmp/tmp/simple_gan_v1.csv',
        model_path='tmp/tmp/wgan_v2.model.keras',
        reuse_model=False,
        reuse_model_path=None):

    reusable_pbar = tqdm(total=max_epochs, unit='epoch', leave=False, dynamic_ncols=True)

    model = WGAN_GP(time_dim=1024, feature_dim=1,
                    latent_dim=latent_dim, n_subjects=n_subjects,
                    use_sublayer_generator=True,
                    use_sublayer_critic=False,
                    use_channel_merger=False)

    if reuse_model:
        print(reuse_model_path)
        model.load_weights(reuse_model_path)

    model.compile(d_optimizer=keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.9),
                  g_optimizer=keras.optimizers.Adam(0.001, beta_1=0.5, beta_2=0.9),
                  gradient_penalty_weight=1.0)

    history = model.fit(data,
                        batch_size=64,
                        epochs=max_epochs,
                        shuffle=True,
                        callbacks=[
                            CustomModelCheckpoint(model_path, save_freq=10),
                            keras.callbacks.ModelCheckpoint(f'{model_path}_best_gloss.model.keras', monitor='g_loss', save_best_only=True),
                            keras.callbacks.ModelCheckpoint(f'{model_path}_best_dloss.model.keras', monitor='d_loss', save_best_only=True),
                            keras.callbacks.CSVLogger(cvloger_path),
                            ProgressBarCallback(n_epochs=max_epochs, n_runs=1, run_index=0, reusable_pbar=reusable_pbar),
                        ])
    return model, history


if __name__ == '__main__':
    data, n_subs = load_data('data/LEMON_DATA/eeg_EC_BaseCorr_Norm_Clamp_with_pos.nc5',
                             n_subjects=202, channels=['O1'], highpass_filter=1,
                             exclude_sub_ids=['sub-010257', 'sub-010044', 'sub-010266'])

    output_path = 'logs/outputs/F1/F1_08.10.2024'

    model, _ = run(data,
                   n_subjects=n_subs,
                   max_epochs=1500,
                   latent_dim=64,
                   cvloger_path=f'{output_path}.csv',
                   model_path=output_path,
                   reuse_model=False,
                   reuse_model_path=None)

    # backup
    model.save(f'{output_path}_final.model.keras')
