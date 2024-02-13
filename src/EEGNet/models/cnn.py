import torch
import torch.nn as nn
import pytorch_lightning as pl


class CNN(pl.LightningModule):
    def __init__(self, input_size=61, output_size=61):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = nn.Sequential(nn.Conv2d(1, 32, kernel_size=2, bias=False),
                                     nn.Conv2d(32, 16, kernel_size=2, bias=False))
        self.decoder = nn.Sequential(nn.ConvTranspose2d(16, 32, 2),
                                     nn.ConvTranspose2d(32, 1, 2))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
