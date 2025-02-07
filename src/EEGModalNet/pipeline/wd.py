import os
os.environ['KERAS_BACKEND'] = 'torch'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import keras
from ...EEGModalNet import WGAN_GP
from typing import List
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import butter, sosfiltfilt
from pathlib import Path
from scipy.signal import welch
from scipy.stats import entropy, skew, kurtosis
from scipy.stats import wasserstein_distance


def time_domain_features_multi_channel(x, axis=-1):  # x: (n_samples, n_channels, n_timepoints)
    # mean = np.mean(x, axis=axis)
    # std = np.std(x, axis=axis)
    skewness = skew(x, axis=axis)
    kurtosis_ = kurtosis(x, axis=axis)
    # rms = np.sqrt(np.mean(x**2, axis=axis))
    return np.stack([skewness, kurtosis_], axis=-1)  # (n_samples, n_channels, n_features)


def hjorth_parameters_multi_channel(signal, axis=-1): 
    activity = np.var(signal, axis=axis)
    first_derivative = np.diff(signal, axis=axis)
    mobility = np.sqrt(np.var(first_derivative, axis=axis) / activity)
    second_derivative = np.diff(first_derivative, axis=axis)
    complexity = np.sqrt(np.var(second_derivative, axis=axis) / np.var(first_derivative, axis=axis)) / mobility
    return np.stack([activity, mobility, complexity], axis=-1)


def spectral_features_multi_channel(signal, fs=128, nperseg=512):
    f, Pxx = welch(signal, fs=fs, nperseg=nperseg, axis=-1)
    total_power = np.sum(Pxx, axis=-1)
    delta = np.sum(Pxx[:, :, (f >= 0.25) & (f < 4)], axis=-1) / total_power
    theta = np.sum(Pxx[:, :, (f >= 4) & (f < 8)], axis=-1) / total_power
    alpha = np.sum(Pxx[:, :, (f >= 8) & (f < 13)], axis=-1) / total_power
    beta = np.sum(Pxx[:, :, (f >= 13) & (f <= 30)], axis=-1) / total_power
    gamma = np.sum(Pxx[:, :, (f > 30) & (f <= 50)], axis=-1) / total_power
    pxx_entropy = entropy(Pxx, axis=-1)
    return np.stack([delta, theta, alpha, beta, gamma, pxx_entropy], axis=-1)


def aggregate_features(real_signal, gen_signal):
    # Extract features
    real_time_features = time_domain_features_multi_channel(real_signal)
    real_hjorth_features = hjorth_parameters_multi_channel(real_signal)
    real_spectral_features = spectral_features_multi_channel(real_signal)

    gen_time_features = time_domain_features_multi_channel(gen_signal)
    gen_hjorth_features = hjorth_parameters_multi_channel(gen_signal)
    gen_spectral_features = spectral_features_multi_channel(gen_signal)

    # Concatenate features
    real_features = np.concatenate([real_time_features, real_hjorth_features, real_spectral_features], axis=-1)
    gen_features = np.concatenate([gen_time_features, gen_hjorth_features, gen_spectral_features], axis=-1)

    return real_features, gen_features


def load_data(data_path: str,
              n_subjects: int = 202,
              channels: List[str] = ['all'],
              bandpass_filter: float = 1.0,
              time_dim: int = 1024,
              exclude_sub_ids=None) -> tuple:

    xarray = xr.open_dataarray(data_path, engine='h5netcdf')
    x = xarray.sel(subject=xarray.subject[:n_subjects])

    if exclude_sub_ids is not None:
        x = x.sel(subject=~x.subject.isin(exclude_sub_ids))

    x = x.to_numpy()
    n_subjects = x.shape[0]

    if bandpass_filter is not None:
        sos = butter(4, bandpass_filter, btype='high', fs=128, output='sos')
        x = sosfiltfilt(sos, x, axis=-1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.tensor(x.copy(), device=device).unfold(2, time_dim, time_dim).permute(0, 2, 3, 1).flatten(0, 1)  # TODO: copy was added because of an error, look into this
    sub = torch.tensor(np.arange(0, n_subjects).repeat(x.shape[0] // n_subjects)[:, np.newaxis], device=device)

    pos = torch.tensor(xarray.ch_positions[None].repeat(x.shape[0], 0), device=device)

    data = {'x': x, 'sub': sub, 'pos': pos}

    return data, n_subjects


if __name__ == '__main__':
    latent_dim = 128
    channels = ['O1', 'O2', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2']
    data, n_subs = load_data('data/LEMON_DATA/EC_8_channels_processed_downsampled.nc5',
                             n_subjects=202,
                             channels=channels,
                             bandpass_filter=0.5,
                             time_dim=512,
                             exclude_sub_ids=None)
    sub = data['sub']
    pos = data['pos']
    x = data['x']
    l = len(x)
    x_mean = x.mean()
    x_std = x.std()
    f_names = f_names = ['skewness', 'kurtosis', 'activity', 'mobility', 'complexity',
                         'delta', 'theta', 'alpha', 'beta', 'gamma', 'entropy']

    model = WGAN_GP(time_dim=512, feature_dim=len(channels),
                    latent_dim=latent_dim, n_subjects=202,
                    use_sublayer_generator=True,
                    use_sublayer_critic=True,
                    use_channel_merger_g=False,
                    use_channel_merger_c=False,
                    interpolation='bilinear')
    all_wd = {}

    for path in sorted(Path('logs/').glob('06*.model.keras'))[:2]:
        model_name = path.stem.split('.')[2][11:]
        model.load_weights(path)
        print(f'Calculating WD for {model_name}...')
        # generated data
        x_gen = model.generator((keras.random.normal((l, latent_dim), mean=x_mean, stddev=x_std), sub[:l].to('mps'), pos[:l].to('mps'))).cpu().detach()
        real_f, fake_f = aggregate_features(x.permute(0, 2, 1).cpu().detach().numpy(), x_gen.permute(0, 2, 1).cpu().detach().numpy())

        for ch in range(len(channels)):
            for fn in range(len(f_names)):
                all_wd[f'{channels[ch]}_{f_names[fn]}_{model_name}'] = wasserstein_distance(real_f[:, ch, fn], fake_f[:, ch, fn])

    all_wd_df = pd.DataFrame(all_wd, index=[0])
    all_wd_df = all_wd_df.melt(var_name='name', value_name='WD')
    all_wd_df[['channel', 'feature', 'epoch']] = all_wd_df['name'].apply(lambda x: x.split('_')).apply(pd.Series)
    all_wd_df.drop('name', axis=1, inplace=True)
    all_wd_df = all_wd_df.pivot(columns='feature', values='WD', index=['channel', 'epoch'])

    all_wd_df.to_csv('logs/wd.csv')
