import os
os.environ['KERAS_BACKEND'] = 'torch'

import torch
import keras
from keras.optimizers.schedules import ExponentialDecay
from src.EEGModalNet import WGAN_GP
from src.EEGModalNet import CustomModelCheckpoint, StepLossHistory
from typing import Dict
import numpy as np
import xarray as xr
from scipy.signal import butter, sosfiltfilt


def load_data(data_path: str,
              n_subjects: int = 202,
              channels = ['O1', 'O2', 'P1', 'P2', 'C1', 'C2', 'F1', 'F2'],
              bandpass_filter: float = 0.5,
              time_dim: int = 512,
              exclude_sub_ids=None,
              device='cpu') -> Dict:

    db = xr.open_dataarray(data_path, engine='h5netcdf')
    x = db.sel(subject=db.subject[:n_subjects], channel=channels)

    if exclude_sub_ids is not None:
        x = x.sel(subject=~x.subject.isin(exclude_sub_ids))

    x = x.to_numpy()
    n_subjects = x.shape[0]

    if bandpass_filter is not None:
        sos = butter(4, bandpass_filter, btype='high', fs=98, output='sos')
        x = sosfiltfilt(sos, x, axis=-1)
    
    # HACK MPS does not support float64
    x = x.astype(np.float32)

    x = torch.tensor(x.copy(), device=device).unfold(2, time_dim, time_dim).permute(0, 2, 3, 1).flatten(0, 1)  # TODO: copy was added because of an error, look into this

    sub = torch.tensor(np.arange(0, n_subjects).repeat(x.shape[0] // n_subjects)[:, np.newaxis], device=device)

    pos = torch.tensor(db.ch_positions[None].repeat(x.shape[0], 0), device=device)

    data = {'x': x, 'sub': sub, 'pos': pos}

    return data

def run(data,
        n_subjects,
        max_epochs=100_000,
        latent_dim=64,
        batch_size=64,
        cvloger_path='tmp/tmp/simple_gan_v1.csv',
        model_path='tmp/tmp/wgan_v2.model.keras',
        reuse_model=False,
        reuse_model_path=None,
        device='cpu'):

    model = WGAN_GP(time_dim=512, feature_dim=data['x'].shape[-1],
                    latent_dim=latent_dim, n_subjects=n_subjects,
                    use_sublayer_generator=True,
                    use_sublayer_critic=True,
                    use_channel_merger_g=False,
                    use_channel_merger_c=False,
                    interpolation='bilinear')

    model = model.to(device)
    print(f'>>>> Model is on {device}')
    print(f">>>> data.x is on {data['x'].device}")
    print(f">>>> data.sub is on {data['sub'].device}")
    print(f">>>> data.pos is on {data['pos'].device}")

    if reuse_model:
        print(reuse_model_path)
        model.load_weights(reuse_model_path)

    lr_schedule_g = ExponentialDecay(0.000188, decay_steps=100000, decay_rate=0.90, staircase=True)
    lr_schedule_d = ExponentialDecay(0.000282, decay_steps=100000, decay_rate=0.90, staircase=True)

    model.compile(d_optimizer=keras.optimizers.Adam(lr_schedule_d, beta_1=0.5, beta_2=0.9),
                  g_optimizer=keras.optimizers.Adam(lr_schedule_g, beta_1=0.5, beta_2=0.9),
                  gradient_penalty_weight=10.0)

    if device == 'cuda':
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


def main(data_path: str, channels: list):

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        print('CUDA is available')
        torch.cuda.set_device(0)
        print(f'Running on {torch.cuda.device_count()} CUDA devices')
        # Explicitly set the CUDA device
        # preload CUDA libraries with a dummy tensor
        _ = torch.randn(1, device="cuda")
    elif torch.backends.mps.is_available():
        print('MPS is available')
        device = 'mps'
    else:
        print('GPU is not available!!')
        exit()

    data = load_data(data_path, channels=channels, device=device)
    n_subjects = len(torch.unique(data['sub']))
    n_channels = len(channels)

    # Apply mixed precision policy
    keras.mixed_precision.set_global_policy('mixed_float16')
    print(f'Global policy is {keras.mixed_precision.global_policy().name}')

    output_path = f'logs/benchmarks/20250413_{n_channels}_electrodes'

    if not os.path.exists('logs/benchmarks'):
        os.makedirs('logs/benchmarks')

    model = run(data,
                n_subjects=n_subjects,
                max_epochs=5000,
                latent_dim=128,
                batch_size=128,
                cvloger_path=f'{output_path}.csv',
                model_path=output_path,
                reuse_model=False,
                reuse_model_path=None,
                device=device)


# Entry point
if __name__ == '__main__':
    data_path = 'data/LEMON_DATA/EC_all_channels_processed_downsampled.nc5'
    channels = ['O1']
    print(f'Running with channels: {channels}')
    main(data_path=data_path, channels=channels)
