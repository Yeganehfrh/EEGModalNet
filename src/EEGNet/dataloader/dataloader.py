from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.utils.data
from src.EEGNet.preprocessing.utils import split_data, get_averaged_data


class EEGNetDataModule(pl.LightningDataModule):
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
                 target: str = 'gender',
                 split_type: str = 'time',
                 shuffling: str = 'no_shuffle',
                 stratified: bool = True,
                 patching: bool = False,
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
        self.target = target
        self.patching = patching
        assert target in ['gender', 'calmness', 'alertness', 'wellbeing'], "The target variable is unknown"
        assert split_type in ['time', 'subject'], "The split type is unknown"

    def prepare_data(self):
        # read data from file
        da = xr.open_dataarray(self.data_dir)

        if self.n_subjects != 'all':
            da = da.sel(subject=da.subject.values[:self.n_subjects])

        if self.patching:
            da = get_averaged_data(da)
            X_input = torch.from_numpy(da.values).float().permute(0, 2, 1)
        else:
            X_input = torch.from_numpy(da.values).float().permute(0, 2, 1)

        # segment
        X_input = X_input.unfold(1, self.segment_size, self.segment_size).permute(0, 1, 3, 2)

        # create positions
        positions = torch.from_numpy(da.ch_positions).float()
        # repeat positions for each subject
        positions = positions.repeat(self.n_subjects, X_input.shape[1], 1, 1)

        # y labels
        target = torch.from_numpy(da.attrs[self.target])
        if self.n_subjects is not None:
            target = target[:self.n_subjects]

        if self.target == 'gender':
            target -= 1   # 0 for female and 1 for male
            target = target.reshape(-1, 1).repeat(1, X_input.shape[1])
            target = F.one_hot(target, num_classes=2).float()
        else:
            # min-max scaling
            target = (target - target.min()) / (target.max() - target.min())
            # TODO: mdeian split
            target = (target > target.median()).float().reshape(-1, 1).repeat(1, X_input.shape[1])

        # create subject ids
        subject_ids = torch.arange(0, X_input.shape[0]).reshape(-1, 1, 1).repeat(1, X_input.shape[1], 1)

        # train/test split
        all_data = split_data(X_input, subject_ids, positions, target,
                              self.shuffling, self.split_type, self.train_ratio,
                              self.stratified)
        X_train, X_test = all_data[0], all_data[1]
        subject_ids_train, subject_ids_test = all_data[2], all_data[3]
        positions_train, positions_test = all_data[4], all_data[5]
        target_train, target_test = all_data[6], all_data[7]

        self.train_dataset = torch.utils.data.TensorDataset(
            X_train,
            subject_ids_train,
            positions_train,
            target_train
        )

        self.val_dataset = torch.utils.data.TensorDataset(
            X_test,
            subject_ids_test,
            positions_test,
            target_test
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


class EEGNetDataModuleKFold(pl.LightningDataModule):  # TODO: REFACTOR: implement kfold cross validation in utils and merge with EEGNetDataModule
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
                 n_folds: int = 5,
                 ):
        assert n_subjects >= n_folds, "n_subjects should be equal to or greater than n_folds"
        super().__init__()
        self.data_dir = data_dir
        self.train_ratio = train_ratio
        self.segment_size = segment_size
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.n_subjects = n_subjects
        self.n_folds = n_folds

    def prepare_data(self):
        # read data from file
        ds = xr.open_dataarray(self.data_dir)
        if self.n_subjects is not None:
            ds = ds.sel(subject=ds.subject.values[:self.n_subjects])
        X_input = torch.from_numpy(ds.values).float().permute(0, 2, 1)

        # segment
        X_input = X_input.unfold(1, self.segment_size, self.segment_size).permute(0, 1, 3, 2)

        # create positions
        positions = torch.from_numpy(ds.ch_positions).float()
        # repeat positions for each subject
        positions = positions.repeat(self.n_subjects, X_input.shape[1], 1, 1)

        # y labels
        target = torch.from_numpy(ds.target)
        if self.n_subjects is not None:
            target = target[:self.n_subjects]
        target -= 1   # 0 for female and 1 for male
        target = target.reshape(-1, 1).repeat(1, X_input.shape[1])
        target = F.one_hot(target, num_classes=2).float()

        # create subject ids
        subject_ids = torch.arange(0, X_input.shape[0]).reshape(-1, 1, 1).repeat(1, X_input.shape[1], 1)

        # k-fold cross validation
        k = self.n_folds
        fold_size = len(X_input) // k
        train_datasets = []
        val_datasets = []

        for i in range(k):
            start = i * fold_size
            end = (i + 1) * fold_size

            X_val = X_input[start:end]
            subject_ids_val = subject_ids[start:end]
            positions_val = positions[start:end]
            target_val = target[start:end]

            X_train_fold = torch.cat([X_input[:start], X_input[end:]], dim=0)
            subject_ids_train_fold = torch.cat([subject_ids[:start], subject_ids[end:]], dim=0)
            positions_train_fold = torch.cat([positions[:start], positions[end:]], dim=0)
            target_train_fold = torch.cat([target[:start], target[end:]], dim=0)

            train_datasets.append(torch.utils.data.TensorDataset(
                X_train_fold.flatten(0, 1),
                subject_ids_train_fold.flatten(0, 1),
                positions_train_fold.flatten(0, 1),
                target_train_fold.flatten(0, 1),
            ))

            val_datasets.append(torch.utils.data.TensorDataset(
                X_val.flatten(0, 1),
                subject_ids_val.flatten(0, 1),
                positions_val.flatten(0, 1),
                target_val.flatten(0, 1),
            ))

        self.train_datasets = train_datasets
        self.val_datasets = val_datasets

    def train_dataloader(self):
        rnd_g = torch.Generator()
        rnd_g.manual_seed(42)   # TODO: remove manual seed
        return [
            DataLoader(dataset, batch_size=self.batch_size, generator=rnd_g, num_workers=7, persistent_workers=True)
            for dataset in self.train_datasets
        ]

    def val_dataloader(self):
        rnd_g = torch.Generator()
        rnd_g.manual_seed(42)  # TODO: remove manual seed
        return [
            DataLoader(dataset, batch_size=self.batch_size, generator=rnd_g, num_workers=7, persistent_workers=True)
            for dataset in self.val_datasets
        ]

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...
