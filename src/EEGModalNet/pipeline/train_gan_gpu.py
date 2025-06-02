import os
os.environ['KERAS_BACKEND'] = 'torch'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import keras
from keras.optimizers.schedules import ExponentialDecay
from ...EEGModalNet import WGAN_GP
from ...EEGModalNet import CustomModelCheckpoint, StepLossHistory
from typing import List
import numpy as np
import pandas as pd
import xarray as xr


def load_data(data_path: str,
              channels: List[str] = ['O1', 'O2', 'P1', 'P2', 'C1', 'C2', 'F1', 'F2'],
              n_subjects: int = 202,
              time_dim: int = 512,
              exclude_sub_ids=None) -> tuple:

    xarray = xr.open_dataarray(data_path, engine='h5netcdf')
    x = xarray.sel(subject=xarray.subject[:n_subjects], channel=channels)

    if exclude_sub_ids is not None:
        x = x.sel(subject=~x.subject.isin(exclude_sub_ids))

    x = x.to_numpy()
    n_subjects = x.shape[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.tensor(x.copy(), device=device).flatten(0, 1)  # TODO: merge condition and participants' axes
    x = x.unfold(2, time_dim, time_dim).permute(0, 2, 3, 1).flatten(0, 1)

    sub = torch.tensor(np.arange(0, n_subjects).repeat(x.shape[0] // n_subjects // 2)[:, np.newaxis], device=device)
    sub = torch.concat([sub, sub])  # the first half: eye open, and the second half: eye close

    pos = torch.tensor(xarray.ch_positions[None].repeat(x.shape[0], 0), device=device)

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

    model = WGAN_GP(time_dim=512,
                    feature_dim=data['x'].shape[-1],
                    latent_dim=latent_dim,
                    n_subjects=n_subjects,
                    use_sublayer_generator=True,
                    use_sublayer_critic=True,
                    use_channel_merger_g=False,
                    use_channel_merger_c=False,
                    interpolation='bilinear')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f'>>>> Model is on {device}')

    if reuse_model:
        print(reuse_model_path)
        model.load_weights(reuse_model_path)

    lr_schedule_g = ExponentialDecay(0.000188//2, decay_steps=100000, decay_rate=0.90, staircase=True)
    lr_schedule_d = ExponentialDecay(0.000282//2, decay_steps=100000, decay_rate=0.90, staircase=True)

    model.compile(d_optimizer=keras.optimizers.Adam(lr_schedule_d, beta_1=0.5, beta_2=0.9),
                  g_optimizer=keras.optimizers.Adam(lr_schedule_g, beta_1=0.5, beta_2=0.9),
                  gradient_penalty_weight=1.0)

    torch.cuda.synchronize()  # wait for model to be loaded

    # step_loss_history = StepLossHistory()

    _ = model.fit(data,
                  batch_size=batch_size,
                  epochs=max_epochs,
                  shuffle=True,
                  callbacks=[
                      CustomModelCheckpoint(model_path, save_freq=20),
                      keras.callbacks.ModelCheckpoint(f'{model_path}_best_gloss.model.keras', monitor='2 g_loss', save_best_only=True),
                      keras.callbacks.ModelCheckpoint(f'{model_path}_best_dloss.model.keras', monitor='1 d_loss', save_best_only=True),
                      keras.callbacks.CSVLogger(cvloger_path),
                      keras.callbacks.TerminateOnNaN()
                      # step_loss_history
                  ])

    return model


if __name__ == '__main__':
    electrodes_choices = {
        8: ['O1', 'O2', 'P1', 'P2', 'C1', 'C2', 'F1', 'F2'],
        16: ['O1', 'O2', 'P3', 'P1', 'Pz', 'P2', 'P4',
             'C3', 'C1', 'C2', 'C4', 'F1', 'F2', 'AF3', 'AFz', 'AF4'],
        32: ['O1', 'O2', 'P3', 'P1', 'Pz', 'P2', 'P4',
             'C3', 'C1', 'C2', 'C4', 'F1', 'F2', 'AF3', 'AFz', 'AF4',
             'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2',
             'FC6', 'CP5', 'CP1', 'CP2', 'CP6', 'POz'],
        56: ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
             'C3', 'C4', 'CP5', 'CP1', 'CP2', 'CP6', 'AFz', 'P7',
             'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2', 'AF7', 'AF3',
             'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT7', 'FC3', 'FC4', 'FT8', 'C5',
             'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2',
             'P6', 'PO7', 'PO3', 'POz', 'PO4', 'PO8']
    }

    data, n_subs = load_data('data/LEMON_DATA/EO-EC_processed_ch-16_sf-128.nc5',
                             channels=electrodes_choices[16],
                             n_subjects=202,
                             time_dim=512,
                             exclude_sub_ids=None)

    if torch.cuda.is_available():
        print('GPU is available')
        # torch.cuda.current_device()
    else:
        print('GPU is not available!!')
        exit()

    print(f'Running on {torch.cuda.device_count()} GPUs')
    # print(f'Using CUDA device: {torch.cuda.get_device_name(0)}')

    # Explicitly set the CUDA device
    torch.cuda.set_device(0)

    # preload CUDA libraries with a dummy tensor
    _ = torch.randn(1, device="cuda")

    # Apply mixed precision policy
    keras.mixed_precision.set_global_policy('mixed_float16')
    print(f'Global policy is {keras.mixed_precision.global_policy().name}')

    output_path = 'logs/20250602'

    model = run(data,
                n_subjects=n_subs,
                max_epochs=5000,
                latent_dim=128 * 2,
                batch_size=128,
                cvloger_path=f'{output_path}.csv',
                model_path=output_path,
                reuse_model=False,
                reuse_model_path=None)
