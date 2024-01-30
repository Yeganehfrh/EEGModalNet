from torch import nn
import pytorch_lightning as pl
import torch


class AutoEncoder(pl.LightningModule):
    """Spatio-temporal auto-encoder.

    """

    def __init__(self,
                 n_channels, space_embedding_dim, time_embedding_dim, kernel_size=1):
        super().__init__()

        self.space_embedding_dim = space_embedding_dim
        self.time_embedding_dim = time_embedding_dim

        # spatial encoder
        self.space_encoder = nn.Sequential(
            nn.Conv1d(n_channels, space_embedding_dim * 2, kernel_size),
            nn.ReLU(),
            nn.Conv1d(space_embedding_dim * 2, space_embedding_dim, kernel_size),
            nn.ReLU())

        # temporal auto-encoder
        self.time_encoder = nn.LSTM(
            self.space_embedding_dim,
            self.time_embedding_dim,
            batch_first=True)

        self.time_decoder = nn.LSTM(
            self.time_embedding_dim,
            self.space_embedding_dim,
            batch_first=True)

        # spatial decoder
        self.space_decoder = nn.Sequential(
            nn.ConvTranspose1d(space_embedding_dim, space_embedding_dim * 2, 1, stride=1),
            nn.ReLU(),
            nn.ConvTranspose1d(space_embedding_dim * 2, n_channels, 1, stride=1),
            nn.ReLU())

    def forward(self, x):

        n_timesteps = x.shape[1]

        # spatial encoding
        y_space_enc = self.space_encoder(x.permute(0, 2, 1))

        # temporal encoding
        y_time_enc, (embedding, c_enc) = self.time_encoder(y_space_enc.permute(0, 2, 1))
        h_enc = embedding.permute(1, 0, 2).repeat(1, n_timesteps, 1)
        y_time_dec, (_, _) = self.time_decoder(h_enc)

        # spatial decoding
        x_space_dec = self.space_decoder(y_time_dec.permute(0, 2, 1))

        y_dec = x_space_dec.permute(0, 2, 1)

        return y_dec, embedding

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat, _ = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat, _ = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer
