import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from src.EEGNet.models.commonBlocks import ChannelMerger, SubjectLayers


class ConvVAE(pl.LightningModule):
    def __init__(self, n_channels=61, n_embeddings=32, n_subjects=200):
        super().__init__()
        self.save_hyperparameters()

        # Fourier positional embedding
        self.pos_emb = ChannelMerger(
            chout=n_channels, pos_dim=288, n_subjects=n_subjects
        )  # TODO: check if this is the right dimension

        # 1 X 1 convolution
        self.cov11 = nn.Conv1d(n_channels, n_channels, kernel_size=1)

        # subject layers
        self.subject_layers = SubjectLayers(in_channels=n_channels, out_channels=n_channels, n_subjects=n_subjects)

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(n_channels, n_channels * 2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv1d(n_channels * 2, n_channels * 4, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv1d(n_channels * 4, n_channels * 8, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.encoder_fc_mu = nn.Linear(n_channels * 8 * 62, n_embeddings)
        self.encoder_fc_log_var = nn.Linear(n_channels * 8 * 62, n_embeddings)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_embeddings, n_channels * 8 * 62),
            nn.Unflatten(dim=1, unflattened_size=(n_channels * 8, 62)),
            nn.ReLU(),
            nn.ConvTranspose1d(n_channels * 8, n_channels * 4, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(n_channels * 4, n_channels * 2, kernel_size=4, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(n_channels * 2, n_channels, kernel_size=4, stride=2),
            nn.ReLU()
        )

    def encode(self, x):
        x = self.encoder_conv(x)
        mu = self.encoder_fc_mu(x)
        log_var = self.encoder_fc_log_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, batch):
        x, sub, pos = batch
        x = x.permute(0, 2, 1)
        x = self.pos_emb(x, pos)
        x = self.cov11(x)
        x = self.subject_layers(x, sub)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

    def training_step(self, batch, batch_idx):
        x, _, _ = batch
        x = x.permute(0, 2, 1)
        x_hat, mu, log_var = self(batch)
        recon_loss = F.mse_loss(x_hat, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + kl_loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ , _ = batch
        x = x.permute(0, 2, 1)
        x_hat, mu, log_var = self(batch)
        recon_loss = F.mse_loss(x_hat, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + kl_loss
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
