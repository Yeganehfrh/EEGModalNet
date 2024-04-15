import torch.nn as nn
import torch


class RNNAutoencoder(nn.Module):
    def __init__(self,
                 # overall structure
                 decoder=True,
                 # RNN parameters
                 n_channels=61,
                 hidden_size=128,
                 num_layers=1,
                 latent_size=32,
                 dropout=0.0,
                 bidirectional=False):
        super().__init__()
        self.encoder = nn.LSTM(n_channels, hidden_size, num_layers, batch_first=True,
                               dropout=dropout, bidirectional=bidirectional)
        self.fc_encoder = nn.Linear(hidden_size, latent_size)
        if decoder:
            self.fc_decoder = nn.Linear(latent_size, hidden_size)
            self.decoder = nn.LSTM(hidden_size, n_channels, num_layers, batch_first=True,
                                   dropout=dropout, bidirectional=bidirectional)

    def forward(self, x):
        _, (h_enc, _) = self.encoder(x)
        latent = self.fc_encoder(h_enc.squeeze(0))
        x_hat = None
        if hasattr(self, 'fc_decoder'):
            h_dec = self.fc_decoder(latent)
            x_hat, _ = self.decoder(h_dec.unsqueeze(1).expand(-1, x.shape[1], -1))  # TODO: there is another way to do this: expand h_enc.
        return x_hat, latent
