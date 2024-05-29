from pathlib import Path
from typing import Literal

import torch
import xarray as xr
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.utils.data
from src.EEGModalNet.preprocessing.utils import split_data, get_averaged_data


class LEMONEEGDataModule():
    """Data module to upload input data and split it into train and validation sets on
    time dimension
    """
    def __init__(self,
                 data_dir: Path = Path('data/LEMON_data/eeg_EC_BaseCorr_Norm_Clamp_with_pos.nc5'),
                 train_ratio: float = 0.7,
                 segment_size: int = 128,
                 batch_size: int = 32,
                 preprocess: bool = False,
                 n_subjects: int = 10,
                 target: str = 'gender',
                 split_type: Literal['time', 'subject'] = 'subject',
                 shuffling: Literal['split_shuffle', 'shuffle_split', 'no_shuffle'] = 'no_shuffle',
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

        self.train_dataset = {
            'x': X_train.numpy(),
            'sub_id': subject_ids_train.numpy(),
            'pos': positions_train.numpy(),
            'y': target_train.numpy()
        }

        self.val_dataset = {
            'x': X_test,
            'sub_id': subject_ids_test,
            'pos': positions_test,
            'y': target_test
        }
        return self
