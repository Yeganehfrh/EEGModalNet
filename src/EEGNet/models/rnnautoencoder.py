import torch.nn as nn
import torch


class RNNAutoencoder(nn.Module):
    def __init__(self,
                 # overall structure
                 decoder,
                 classifier,
                 n_classes,
                 # RNN parameters
                 n_channels,
                 hidden_size,
                 num_layers,
                 latent_size,
                 dropout,
                 bidirectional):
        super().__init__()
        assert decoder or classifier, "At least one of decoder or classifier must be True"
        self.encoder = nn.LSTM(n_channels, hidden_size, num_layers, batch_first=True,
                               dropout=dropout, bidirectional=bidirectional)
        self.fc_encoder = nn.Linear(hidden_size, latent_size)
        if classifier:
            self.fc_classifier = nn.Linear(latent_size, n_classes)
        if decoder:
            self.fc_decoder = nn.Linear(latent_size, hidden_size)
            self.decoder = nn.LSTM(hidden_size, n_channels, num_layers, batch_first=True,
                                   dropout=dropout, bidirectional=bidirectional)

    def forward(self, x):
        _, (h_enc, _) = self.encoder(x)
        latent = self.fc_encoder(h_enc.squeeze(0))
        y_hat = None
        if hasattr(self, 'fc_classifier'):
            y_hat = self.fc_classifier(latent)
        x_hat = None
        if hasattr(self, 'fc_decoder'):
            h_dec = self.fc_decoder(latent)
            x_hat, _ = self.decoder(h_dec.unsqueeze(0))
        return latent, x_hat, y_hat
