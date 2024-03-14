from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.utils.data
from src.EEGNet.preprocessing.utils import split_data


class TimeDimSplit(pl.LightningDataModule):
    """Data module to upload input data and split it into train and validation sets on
    time dimension
    """
    def __init__(self,
                 data_dir: Path = Path('data/processed/normalized_clamped/eeg_EC_BaseCorr_Norm_Clamp_with_pos.nc5'),
                 train_ratio: float = 0.7,
                 segment_size: int = 128,
                 batch_size: int = 32,
                 preprocess: bool = False,
                 n_subjects: int = 10,
                 split_type: str = 'time',
                 shuffling: str = 'no_shuffle',
                 stratified: bool = True
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.train_ratio = train_ratio
        self.segment_size = segment_size
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.n_subjects = n_subjects
        self.shuffling = shuffling
        self.split_type = split_type
        self.stratified = stratified

    def prepare_data(self):
        # read data from file
        ds = xr.open_dataset(self.data_dir)
        if self.n_subjects is not None:
            ds = ds.sel(subject=ds.subject.values[:self.n_subjects])
        X_input = torch.from_numpy(ds['__xarray_dataarray_variable__'].values).float().permute(0, 2, 1)

        # segment
        X_input = X_input.unfold(1, self.segment_size, self.segment_size).permute(0, 1, 3, 2)

        # create positions
        positions = torch.from_numpy(ds.ch_positions).float()
        # repeat positions for each subject
        positions = positions.repeat(self.n_subjects, X_input.shape[1], 1, 1)

        # y labels
        gender = torch.from_numpy(ds.gender)
        if self.n_subjects is not None:
            gender = gender[:self.n_subjects]
        gender -= 1   # 0 for female and 2 for male
        gender = gender.reshape(-1, 1).repeat(1, X_input.shape[1])
        gender = F.one_hot(gender, num_classes=2).float()

        # create subject ids
        subject_ids = torch.arange(0, X_input.shape[0]).reshape(-1, 1, 1).repeat(1, X_input.shape[1], 1)

        # train/test split
        all_data = split_data(X_input, subject_ids, positions, gender,
                              self.shuffling, self.split_type, self.train_ratio,
                              self.stratified)
        X_train, X_test = all_data[0], all_data[1]
        subject_ids_train, subject_ids_test = all_data[2], all_data[3]
        positions_train, positions_test = all_data[4], all_data[5]
        gender_train, gender_test = all_data[6], all_data[7]

        self.train_dataset = torch.utils.data.TensorDataset(
            X_train,
            subject_ids_train,
            positions_train,
            gender_train
        )

        self.val_dataset = torch.utils.data.TensorDataset(
            X_test,
            subject_ids_test,
            positions_test,
            gender_test
        )

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
