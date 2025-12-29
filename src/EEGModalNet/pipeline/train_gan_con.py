## Fix the the seed for ablation experiment
SEED = 42
import os
import random
import numpy as np

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

## Setting Keras backend to PyTorch and configuring CUDA devices
os.environ['KERAS_BACKEND'] = 'torch'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

print("SDP backends:",
      "flash =", torch.backends.cuda.flash_sdp_enabled(),
      "mem_efficient =", torch.backends.cuda.mem_efficient_sdp_enabled(),
      "math =", torch.backends.cuda.math_sdp_enabled())

import keras
keras.utils.set_random_seed(SEED)

from keras.optimizers.schedules import ExponentialDecay
from ...EEGModalNet import TCNWGAN, CustomModelCheckpoint, preprocess_data, WGAN_GP_V0, FiLMGAN, RandomCropEEGDataset
from typing import List, Dict
import numpy as np
import xarray as xr
from meegkit import dss
from scipy.signal import butter, sosfiltfilt


def load_data(data_path: str,
              channels: List[str] | str = ['O1', 'O2', 'P1', 'P2', 'C1', 'C2', 'F1', 'F2'],
              n_subjects: int = 202,
              condition: str | None = None,
              exclude_sub_ids=None,
              preprocess=False,
              highpass=False,
              remove_line_noise=True) -> Dict:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    xarray = xr.open_dataarray(data_path, engine='h5netcdf')

    if n_subjects < xarray.sizes['subject']:
        xarray = xarray.sel(subject=xarray.subject[:n_subjects])

    if condition == 'both':
        data_path_2 = data_path.replace('EC', 'EO') if 'EC' in data_path else data_path.replace('EO', 'EC')
        xarray_2 = xr.open_dataarray(data_path_2, engine='h5netcdf')
        if n_subjects < xarray.sizes['subject']:
            xarray_2 = xarray_2.sel(subject=xarray.subject[:n_subjects])
        xarray_2  = xarray_2.rename({"time": "timestep"}) # we know that the naming of the time dimensions are not the same
        xarray = xr.concat([xarray, xarray_2], dim='subject')

    if channels != 'all':
        xarray = xarray.sel(channel=channels)

    if exclude_sub_ids is not None:
        xarray = xarray.sel(subject=~xarray.subject.isin(exclude_sub_ids))

    x = xarray.to_numpy()

    if preprocess:
        x = preprocess_data(x, sampling_rate=128)

    if highpass:
        sos = butter(4, 0.5, btype='high', fs=128, output='sos')
        x = sosfiltfilt(sos, x, axis=-1)

    if remove_line_noise:
        x, _ = dss.dss_line(x.T, fline=50, sfreq=128, nremove=1)
        x = x.T

    x = torch.tensor(x.copy(), dtype=torch.float32)
    sub = torch.arange(n_subjects)[:, None]
    pos = torch.tensor(xarray.ch_positions[None].repeat(x.shape[0], axis=0), dtype=torch.float32)

    if condition == 'both':
        sub = sub.repeat(2, 1)
        state_ids = torch.cat([
            torch.zeros(n_subjects, dtype=torch.long),
            torch.ones(n_subjects, dtype=torch.long)
        ], dim=0)[:, None]

    return {'x': x, 'sub': sub, 'pos': state_ids if condition == 'both' else pos}


def run(train_loader,
        n_subjects,
        channels,
        max_epochs=100_000,
        latent_dim=64,
        batch_size=64,
        cvloger_path='tmp/tmp/simple_gan_v1.csv',
        model_path='tmp/tmp/wgan_v2.model.keras',
        reuse_model=False,
        reuse_model_path=None,
        shuffle=False,
        steps_per_epoch=500):

    model = FiLMGAN(time_dim=512,
                    feature_dim=len(channels),
                    latent_dim=latent_dim,
                    n_subjects=n_subjects,
                    use_sublayer_generator=True,
                    use_sublayer_critic=True,
                    use_channel_merger_g=False,
                    use_channel_merger_c=False,
                    interpolation='bilinear',
                    steps_per_epoch=steps_per_epoch)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f'>>>> Model is on {device}')

    if reuse_model:
        print(reuse_model_path)
        model.load_weights(reuse_model_path)

    lr_schedule_g = ExponentialDecay(0.0002, decay_steps=100000, decay_rate=0.90, staircase=True)
    lr_schedule_d = ExponentialDecay(0.0006, decay_steps=100000, decay_rate=0.90, staircase=True)

    model.compile(d_optimizer=keras.optimizers.Adam(lr_schedule_d, beta_1=0.0, beta_2=0.9),
                  g_optimizer=keras.optimizers.Adam(lr_schedule_g, beta_1=0.0, beta_2=0.9),
                  gradient_penalty_weight=5)

    torch.cuda.synchronize()  # wait for model to be loaded

    # step_loss_history = StepLossHistory()
    def infinite_loader(loader):
        while True:
            for batch in loader:
                yield (batch,)

    _ = model.fit(infinite_loader(train_loader),
                  batch_size=batch_size,
                  epochs=max_epochs,
                  shuffle=shuffle,
                  steps_per_epoch=steps_per_epoch,
                  callbacks=[
                      CustomModelCheckpoint(model_path, save_freq=20),
                      keras.callbacks.ModelCheckpoint(f'{model_path}_best_gloss.model.keras', monitor='2 g_loss', save_best_only=True, mode='min'),
                      keras.callbacks.ModelCheckpoint(f'{model_path}_best_dloss.model.keras', monitor='1 d_loss', save_best_only=True, mode='min'),
                      keras.callbacks.CSVLogger(cvloger_path),
                      keras.callbacks.TerminateOnNaN()
                      # step_loss_history
                  ])

    return model


if __name__ == '__main__':
    CHANNELS = {
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
    N_SUBJECTS = 202
    LATENT_DIM = 128
    BATCH_SIZE = 128
    OUTPUT_PATH = 'logs/20251229_v3'
    CONDITION = 'both' #FIX currently it only work with two conditions

    data = load_data('data/LEMON_DATA/EC_ch-8_sf-128.nc5',
                     channels='all',
                     n_subjects=N_SUBJECTS,
                     condition=CONDITION,
                     exclude_sub_ids=None,
                     preprocess=True,
                     highpass=True,
                     remove_line_noise=True)
    

    # Set random generator for DataLoader
    g = torch.Generator()
    g.manual_seed(SEED)
    
    train_loader = torch.utils.data.DataLoader(
    RandomCropEEGDataset(
        data['x'], data['sub'], data['pos'],
        seg_len=512,
        n_samples=100_000   # 100k random crops per epoch
    ),
    batch_size=128,
    shuffle=False,
    num_workers=0,
    drop_last=True,
    generator=g
    )

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

    model = run(train_loader,
                n_subjects=N_SUBJECTS,
                channels=CHANNELS[8],
                max_epochs=1000,
                latent_dim=LATENT_DIM,
                batch_size=BATCH_SIZE,
                cvloger_path=f'{OUTPUT_PATH}.csv',
                model_path=OUTPUT_PATH,
                reuse_model=False,
                reuse_model_path=None,
                shuffle=False)
