import torch
import torch.nn as nn
import pytorch_lightning as pl


class CNN(pl.LightningModule):
    def __init__(self, n_channels=61, n_embeddings=32):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
                       nn.Conv1d(n_channels, n_channels, kernel_size=4, stride=1),
                       nn.ReLU(),
                       nn.Conv1d(n_channels, n_channels * 2, kernel_size=4, stride=1),
                       nn.ReLU(),
                       nn.Conv1d(n_channels * 2, n_channels * 4, kernel_size=4, stride=1),
                       nn.ReLU(),
                       nn.Conv1d(n_channels * 4, n_channels * 8, kernel_size=4, stride=1),
                       nn.ReLU(),
                       nn.Conv1d(n_channels * 8, n_channels * 16, kernel_size=4, stride=1),
                       nn.ReLU(),
                       nn.Flatten(),
                       nn.Linear(n_channels * 16, n_embeddings)
                )

        self.decoder = nn.Sequential(
                       nn.Linear(n_embeddings, n_channels * 16),
                       nn.Unflatten(dim=1, unflattened_size=(n_channels * 16, 1)),
                       nn.ConvTranspose1d(n_channels * 16, n_channels * 8, kernel_size=4, stride=1),
                       nn.ReLU(),
                       nn.ConvTranspose1d(n_channels * 8, n_channels * 4, kernel_size=4, stride=1),
                       nn.ReLU(),
                       nn.ConvTranspose1d(n_channels * 4, n_channels * 2, kernel_size=4, stride=1),
                       nn.ReLU(),
                       nn.ConvTranspose1d(n_channels * 2, n_channels, kernel_size=4, stride=1),
                       nn.ReLU(),
                       nn.ConvTranspose1d(n_channels, n_channels, kernel_size=4, stride=1),
                       nn.ReLU(),
                       nn.Sigmoid()
                )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        h = self.encoder(x)
        x_hat = self.decoder(h)
        return x_hat

    def training_step(self, batch):
        x, sub_id = batch
        x_hat = self(x)
        x = x.permute(0, 2, 1)
        loss = nn.functional.cross_entropy(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch):
        x, sub_id = batch
        x_hat = self(x)
        x = x.permute(0, 2, 1)
        loss = nn.functional.cross_entropy(x_hat, x)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
