import torch
import keras
from ...EEGModalNet import WGAN_GP
from ...EEGModalNet import ProgressBarCallback
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
              time_dim: int = 1024) -> dict:

    xarray = xr.open_dataarray(data_path, engine='h5netcdf')
    x = xarray.sel(subject=xarray.subject[:n_subjects]).to_numpy()

    if channels[0] != 'all':
        ch_ind = find_channel_ids(xarray, channels)
        x = x[:, ch_ind, 440:]

    if highpass_filter is not None:
        sos = butter(4, highpass_filter, btype='high', fs=128, output='sos')
        x = sosfilt(sos, x, axis=-1)

    x = torch.tensor(x).unfold(2, time_dim, time_dim).permute(0, 2, 3, 1).flatten(0, 1)
    sub = np.arange(0, n_subjects).repeat(x.shape[0] // n_subjects)[:, np.newaxis]
    pos = xarray.ch_positions[ch_ind][None].repeat(x.shape[0], 0)
    return {'x': x, 'sub': sub, 'pos': pos}


def run(data, n_subjects, max_epochs=100_000, latent_dim=64):

    reusable_pbar = tqdm(total=max_epochs, unit='epoch', leave=False, dynamic_ncols=True)

    model = WGAN_GP(time_dim=1024, feature_dim=1,
                    latent_dim=latent_dim, n_subjects=n_subjects,
                    use_sublayer_generator=True,
                    use_sublayer_critic=False,
                    use_channel_merger=True,
                    kerner_initializer='random_normal')

    model.compile(d_optimizer=keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.9),
                  g_optimizer=keras.optimizers.Adam(0.001, beta_1=0.5, beta_2=0.9),
                  gradient_penalty_weight=1.0)


    history = model.fit(data,
                        batch_size=64,
                        epochs=max_epochs,
                        shuffle=True,
                        callbacks=[
                            keras.callbacks.ModelCheckpoint('tmp/keras_models/wgan_v1.model.keras', monitor='d_loss', mode='min'),
                            keras.callbacks.EarlyStopping(monitor='g_loss', mode='min', patience=500),
                            keras.callbacks.CSVLogger('tmp/keras_logs/simple_gan_v1.csv'),
                            ProgressBarCallback(n_epochs=max_epochs, n_runs=1, run_index=0, reusable_pbar=reusable_pbar),
                        ])


if __name__ == '__main__':
    data = load_data('data/EEGData.h5', n_subjects=20, channels=['F1'], highpass_filter=1)
    run(data, n_subjects=20, max_epochs=100, latent_dim=64)
