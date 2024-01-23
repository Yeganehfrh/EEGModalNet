import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader
from pathlib import Path
import mne



class OtkaTimeDimSplit(pl.LightningDataModule):
    """Data module to upload input data and split it into train and validation sets on
    time dimension
    """
    def __init__(self,
                 data_dir: Path = Path('data/otka.nc5'),
                 train_ratio: float = 0.7,
                 segment_size: int = 128,
                 batch_size: int = 32,
                 filter: bool = False,
                 bandpass: list = [30, 50],
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.train_ratio = train_ratio
        self.segment_size = segment_size
        self.batch_size = batch_size
        self.filter = filter
        self.bandpass = bandpass


    def prepare_data(self):
        # read data from file
        ds = xr.open_dataset(self.data_dir)
        X_input = torch.from_numpy(ds['hypnotee'].values).float().permute(0, 2, 1)

        if self.filter:
            X_input = bandpass_filter(X_input, bandpass=self.bandpass)

        # segment
        X_input = X_input.unfold(1, self.segment_size, self.segment_size).permute(0, 1, 3, 2)

        # create subject ids
        subject_ids = torch.arange(0, X_input.shape[0]).reshape(-1, 1, 1).repeat(1, X_input.shape[1], 1)

        # cut point for train/test split
        cut_point = int(X_input.shape[1] * self.train_ratio)

        # train/test split & normalization
        # TODO: should we normalize after flatening? or instead normalize over dimension 1?
        X_train = F.normalize(X_input[:, :cut_point, :, :], dim=2)
        X_test = F.normalize(X_input[:, cut_point:, :, :], dim=2)

        self.train_dataset = torch.utils.data.TensorDataset(
            X_train.flaten(0, 1),
            subject_ids[:, :cut_point, :].flatten(0, 1),
        )

        self.val_dataset = torch.utils.data.TensorDataset(
            X_test.flaten(0, 1),
            subject_ids[:, cut_point:, :].flatten(0, 1),
        )

    def train_dataloader(self):
        rnd_g = torch.Generator()
        rnd_g.manual_seed(42)   # TODO: remove manual seed
        return DataLoader(self.train_dataset, batch_size=self.batch_size, generator=rnd_g)

    def val_dataloader(self):
        rnd_g = torch.Generator()
        rnd_g.manual_seed(42)  # TODO: remove manual seed
        return DataLoader(self.val_dataset, batch_size=self.batch_size, generator=rnd_g)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...
