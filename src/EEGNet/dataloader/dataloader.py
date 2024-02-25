from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader
import mne


class TimeDimSplit(pl.LightningDataModule):
    """Data module to upload input data and split it into train and validation sets on
    time dimension
    """
    def __init__(self,
                 data_dir: Path = Path('data/processed/normalized_clamped/eeg_EC_BaseCorr_Norm_Clamp.nc5'),
                 info_dir: Path = Path('data/info/sub-info.fif'),
                 train_ratio: float = 0.7,
                 segment_size: int = 128,
                 batch_size: int = 32,
                 preprocess: bool = False,
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.train_ratio = train_ratio
        self.segment_size = segment_size
        self.batch_size = batch_size
        self.preprocess = preprocess

    def prepare_data(self):
        # read data from file
        da = xr.open_dataarray(self.data_dir)
        da = da.sel(subject=da.subject.values[:50])  # TODO: remove this line
        X_input = torch.from_numpy(da.values).float().permute(0, 2, 1)

        # read subject info
        mne_info = mne.read_info(self.info_dir)

        # segment
        X_input = X_input.unfold(1, self.segment_size, self.segment_size).permute(0, 1, 3, 2)

        # create subject ids
        subject_ids = torch.arange(0, X_input.shape[0]).reshape(-1, 1, 1).repeat(1, X_input.shape[1], 1)

        # train/test split
        cut_point = int(X_input.shape[1] * self.train_ratio)  # cutoff
        X_train = X_input[:, :cut_point, :, :].flatten(0, 1)
        X_test = X_input[:, cut_point:, :, :].flatten(0, 1)

        if self.preprocess:
            # first check if the preprocessed data with the same cutoff already exists
            if Path(f'tmp/train_{self.train_ratio}.pt').exists():
                self.train_dataset = torch.load(f'tmp/train_{self.train_ratio}.pt')
                self.val_dataset = torch.load(f'tmp/val_{self.train_ratio}.pt')
                return
            # Pre_Process: robust scaling
            print('Scaling data...')
            print('X_train shape:', X_train.shape)
            X_train = torch.tensor(np.array(
                [RobustScaler().fit_transform(X_train[i, :, :]) for i in range(X_train.shape[0])]
                )).float()
            X_test = torch.tensor(np.array(
                [RobustScaler().fit_transform(X_test[i, :, :]) for i in range(X_test.shape[0])]
                )).float()

            # Pre_Process: baseline correction
            print('Clamping data...')
            X_train = torch.clamp(X_train, min=-20, max=20)   # TODO The way we do clamping needs to be updated
            X_test = torch.clamp(X_test, min=-20, max=20)

        self.train_dataset = torch.utils.data.TensorDataset(
            X_train,
            subject_ids[:, :cut_point, :].flatten(0, 1),
            mne_info
        )

        self.val_dataset = torch.utils.data.TensorDataset(
            X_test,
            subject_ids[:, cut_point:, :].flatten(0, 1),
            mne_info
        )

        # torch.save(self.train_dataset, f'tmp/train_{self.train_ratio}.pt')
        # torch.save(self.val_dataset, f'tmp/val_{self.train_ratio}.pt')

    def train_dataloader(self):
        rnd_g = torch.Generator()
        rnd_g.manual_seed(42)   # TODO: remove manual seed
        return DataLoader(self.train_dataset, batch_size=self.batch_size, generator=rnd_g, num_workers=7,
                          persistent_workers=True)

    def val_dataloader(self):
        rnd_g = torch.Generator()
        rnd_g.manual_seed(42)  # TODO: remove manual seed
        return DataLoader(self.val_dataset, batch_size=self.batch_size, generator=rnd_g, num_workers=7,
                          persistent_workers=True)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...
