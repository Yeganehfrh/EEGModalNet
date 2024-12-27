import os
os.environ['KERAS_BACKEND'] = 'torch'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.multiprocessing as mp
import keras
from ...EEGModalNet import WGAN_GP
from ...EEGModalNet import CustomModelCheckpoint
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
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

    return {'x': x[:16384], 'sub': torch.tensor(sub)[:16384], 'pos': torch.tensor(pos)[:16384]}, n_subjects  # TODO: remove hard-coded value


def run(rank,
        world_size,
        data,
        n_subjects,
        max_epochs=100_000,
        latent_dim=64,
        cvloger_path='tmp/tmp/simple_gan_v1.csv',
        model_path='tmp/tmp/wgan_v2.model.keras',
        reuse_model=False,
        reuse_model_path=None):

    # Set up the environment variables for master address and port
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    device = torch.device("cuda:{}".format(rank) if torch.cuda.is_available() else "cpu")

    # Initialize the process group
    dist.init_process_group(rank=rank, world_size=world_size)

    torch.cuda.set_device(device)

    model = WGAN_GP(time_dim=512, feature_dim=1,
                    latent_dim=latent_dim, n_subjects=n_subjects,
                    use_sublayer_generator=True,
                    use_sublayer_critic=False,
                    use_channel_merger=False,
                    kerner_initializer='random_normal',
                    interpolation='bilinear')  # TODO model.to(rank)

    model.to(rank)

    model = DDP(model, device_ids=[rank], output_device=rank)  # device_ids=[rank] # TODO Create the model and move it to the appropriate device

    # Create the dataset and dataloader
    sampler = DistributedSampler(data, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(data, batch_size=128, sampler=sampler, shuffle=False)

    print('>>>>>>> Model and DataLoader created')

    model.compile(d_optimizer=keras.optimizers.Adam(0.00005, beta_1=0.5, beta_2=0.9),
                  g_optimizer=keras.optimizers.Adam(0.0005, beta_1=0.5, beta_2=0.9),
                  gradient_penalty_weight=5.0)

    for epoch in range(max_epochs):
        for batch in dataloader:
            x, sub, pos = batch
            logs = model.train_step((x, sub, pos))
            print(f"Rank {rank}, Epoch {epoch}, Batch {batch}, Logs {logs}")

    # Clean up
    dist.destroy_process_group()

    return model, None


def train_worker(rank, world_size, data, n_subjects, max_epochs, latent_dim, cvloger_path, model_path, reuse_model, reuse_model_path):
    torch.set_num_threads(1)  # Limit each worker to a single thread
    run(rank, world_size, data, n_subjects, max_epochs, latent_dim, cvloger_path, model_path, reuse_model, reuse_model_path)


if __name__ == '__main__':
    data, n_subs = load_data('data/LEMON_DATA/eeg_EC_BaseCorr_Norm_Clamp_with_pos.nc5',
                             n_subjects=34, channels=['O1'], highpass_filter=1, time_dim=512,  # TODO: n_subjects=202
                             exclude_sub_ids=['sub-010257', 'sub-010044', 'sub-010266'])

    dataset = TensorDataset(data['x'], data['sub'], data['pos'])

    output_path = 'logs/outputs/multiprocessing_test'

    n_cpus = 1
    max_epochs = 1  # 100_000
    mp.spawn(train_worker,
             args=(n_cpus, dataset, n_subs, max_epochs, 64, f'{output_path}.csv', output_path, False, None),
             nprocs=n_cpus,
             join=True)
