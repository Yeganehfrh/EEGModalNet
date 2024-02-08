import torch
import torch.nn as nn
import pytorch_lightning as pl

class CNN(pl.LightningModule):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (1, 51), padding=(0, 25), bias=False)
        self.bnorm1 = nn.BatchNorm2d(16, False)
        self.conv2 = nn.Conv2d(16, 32, (2, 1), groups=16, bias=False)
        self.bnorm2 = nn.BatchNorm2d(32, False)
        self.fc1 = nn.Linear(32*2, output_size)
    
    def forward(self, x):
        x = self.bnorm1(self.conv1(x))
        x = self.bnorm2(self.conv2(x))
        x = self.fc1(x.view(x.size(0), -1))
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
