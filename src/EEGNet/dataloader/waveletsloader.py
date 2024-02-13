from pathlib import Path
import pytorch_lightning as pl
import torch
import xarray as xr
from torch.utils.data import DataLoader


class Wavelets(pl.LightningDataModule):
    """Data module to upload wavelets data and split it across the subject dimention
    into train and validation sets
    """
    def __init__(self,
                 data_dir: Path = Path('data/wavelets/wavelets.nc5'),
                 train_ratio: float = 0.8,
                 batch_size: int = 32,
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.train_ratio = train_ratio
        self.batch_size = batch_size

    def prepare_data(self):
        # read data from file
        da = xr.open_dataarray(self.data_dir)
        X_input = torch.from_numpy(da.values).float()

        # # create subject ids
        # subject_ids = torch.arange(0, X_input.shape[0]).reshape(-1, 1, 1).repeat(1, X_input.shape[1], 1)

        # train/test split
        cut_point = int(X_input.shape[0] * self.train_ratio)  # cutoff
        X_train = y_train = X_input[:cut_point, :, :].unsqueeze(1)
        X_test = y_test = X_input[cut_point:, :, :].unsqueeze(1)

        self.train_dataset = torch.utils.data.TensorDataset(
            X_train,
            y_train,
            # subject_ids[:, :cut_point, :].flatten(0, 1),
        )

        self.val_dataset = torch.utils.data.TensorDataset(
            X_test,
            y_test,
            # subject_ids[:, cut_point:, :].flatten(0, 1),
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
