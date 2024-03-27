import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.EEGNet.models.commonBlocks import Classifier
import torchmetrics.functional as tmf


class JointEmbedding(pl.LightningModule):
    def __init__(self,
                 n_channels=61,
                 n_embeddings=32,
                 n_timepoints=62,
                 n_classes=2,
                 use_classifier=True,
                 dropout=0.5):

        super().__init__()
        self.save_hyperparameters()

        # dropout
        self.dropout = nn.Dropout(dropout)  # TODO: implement other form of distorsion

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, n_channels * 2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv1d(n_channels * 2, n_channels * 4, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv1d(n_channels * 4, n_channels * 8, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(n_channels * 8 * n_timepoints, n_embeddings)
        )

        if use_classifier:
            self.classifier = Classifier(n_embeddings, n_classes)

    def forward(self, x):
        x = self.encoder(x)
        if hasattr(self, 'classifier'):
            y_hat = self.classifier(x)
        return x, y_hat

    def training_step(self, batch, batch_idx):
        x, _, _, y = batch
        x = x.permute(0, 2, 1)
        x_ = self.dropout(x)
        x_hat, y_hat = self(x_)
        x = self.encoder(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log('train/loss_recon', loss)
        if hasattr(self, 'classifier'):
            loss_cls = nn.CrossEntropyLoss()(y_hat, y)
            self.log('train/loss_cls', loss_cls)
            loss += loss_cls
            acc = tmf.accuracy(y_hat, y, task='binary', num_classes=2)
            self.log('train/acc', acc)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, _, y = batch
        x = x.permute(0, 2, 1)
        x_ = self.dropout(x)
        x_hat, y_hat = self(x_)
        x = self.encoder(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log('val/loss_recon', loss)
        if hasattr(self, 'classifier'):
            loss_cls = nn.CrossEntropyLoss()(y_hat, y)
            self.log('val/loss_cls', loss_cls)
            loss += loss_cls
            acc = tmf.accuracy(y_hat, y, task='binary', num_classes=2)
            self.log('val/acc', acc)
        self.log('val/loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
