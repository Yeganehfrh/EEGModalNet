import os
os.environ['KERAS_BACKEND'] = 'torch'
from typing import List
import numpy as np
import torch
import keras
import xarray as xr
from matplotlib import pyplot as plt
from scipy.signal import butter, sosfilt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from ...EEGModalNet import WGAN_GP


def find_channel_ids(dataarray, ch_names):
    return [i for i, ch in enumerate(dataarray.channel.to_numpy()) if ch in ch_names]


def load_data(data_path: str,
              n_subjects: int = 202,
              channels: List[str] = ['all'],
              highpass_filter: float = 1,
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
        x = sosfilt(sos, x, axis=-1)

    x = torch.tensor(x).unfold(2, time_dim, time_dim).permute(0, 2, 3, 1).flatten(0, 1)

    sub = np.arange(0, n_subjects).repeat(x.shape[0] // n_subjects)[:, np.newaxis]

    ch_ind = find_channel_ids(xarray, channels)
    pos = xarray.ch_positions[ch_ind][None].repeat(x.shape[0], 0)

    return {'x': x, 'sub': sub, 'pos': pos}, n_subjects


def plot_2d_components(x, x_gen, method='pca', plot_name='test'):
    sample_len = x.shape[0]
    x_flat = x.mean(axis=2)
    x_flat_hat = x_gen.mean(axis=2)

    x_flat_final = np.concatenate((x_flat, x_flat_hat), axis=0)
    if method == 'tsne':
        tsne = TSNE(n_components=2, verbose=1, perplexity=20)
    if method == 'pca':
        tsne = PCA(n_components=2)
    tsne_results = tsne.fit_transform(x_flat_final)

    # save the results
    np.save(f'tsne_results_{plot_name}.npy', tsne_results)

    # Plotting
    f, ax = plt.subplots(1)
    colors = ["red" for i in range(sample_len)] + ["blue" for i in range(sample_len)]

    plt.scatter(tsne_results[:sample_len, 0], tsne_results[:sample_len, 1],
                c=colors[:sample_len], alpha=0.2, label="Original")
    plt.scatter(tsne_results[sample_len:, 0], tsne_results[sample_len:, 1],
                c=colors[sample_len:], alpha=0.2, label="Synthetic")

    ax.legend()
    plt.title(f'{plot_name}')
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')
    f.savefig(f'{plot_name}.png')
    plt.close(f)


if __name__ == '__main__':
    data, n_subs = load_data('data/LEMON_DATA/eeg_EC_BaseCorr_Norm_Clamp_with_pos.nc5',
                             n_subjects=202, channels=['O1'], highpass_filter=1,
                             exclude_sub_ids=['sub-010257', 'sub-010044', 'sub-010266'])
    x = data['x']
    pos = data['pos']
    sub = data['sub']
    mean_x, std_x = x.mean(), x.std()
    wgan_gp = WGAN_GP(time_dim=1024, feature_dim=1,
                      latent_dim=64, n_subjects=199,
                      use_sublayer_generator=True,
                      use_sublayer_critic=False,
                      use_channel_merger=False,)

    # load the model weights
    for i in range(20, 100, 20):
        wgan_gp.load_weights(f'data/logs/O1/O1_09.10.2024_epoch_{i}.model.keras')

        x_gen = wgan_gp.generator(keras.random.normal((len(x), 64), mean_x, std_x), torch.tensor(sub).to('mps'),
                                  pos).cpu().detach()
        plot_2d_components(x, x_gen, 'tsne', f"Epoch_{i}")
