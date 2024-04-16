import torch.nn as nn
import torch


class RNNAutoencoder(nn.Module):
    def __init__(self,
                 n_channels=61,
                 hidden_size=128,
                 num_layers=1,
                 latent_size=32,
                 dropout=0.0,
                 bidirectional=False,
                 use_decoder=True):
        super().__init__()
        self.encoder = nn.LSTM(n_channels, hidden_size, num_layers, batch_first=True,
                               dropout=dropout, bidirectional=bidirectional)
        self.fc_encoder = nn.Linear(hidden_size, latent_size)
        if use_decoder:
            self.fc_decoder = nn.Linear(latent_size, hidden_size)
            self.decoder = nn.LSTM(hidden_size, n_channels, num_layers, batch_first=True,
                                   dropout=dropout, bidirectional=bidirectional)

    def forward(self, x):
        _, (h_enc, _) = self.encoder(x)
        latent = self.fc_encoder(h_enc.squeeze(0))
        x_hat = None
        if hasattr(self, 'decoder'):
            h_dec = self.fc_decoder(latent)
            x_hat, _ = self.decoder(h_dec.unsqueeze(1).expand(-1, x.shape[1], -1))  # TODO: there is another way to do this: expand h_enc.
        return x_hat, latent


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels, out_channel, n_embeddings, segment_size, use_decoder=True,
                **kwargs):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.encoder = convNet(in_channels, out_channel, mode='encoder', **kwargs)
        self.out_chan_size, self.length_size = self._calculate_latent_size(in_channels, segment_size)
        self.fc_encoder = nn.Sequential(nn.Flatten(),
                                        nn.Linear(self.out_chan_size*self.length_size, n_embeddings))
        if use_decoder:
            self.fc_decoder = nn.Sequential(nn.Linear(n_embeddings, self.out_chan_size*self.length_size),
                                            nn.Unflatten(1, (self.out_chan_size, self.length_size)))
            self.decoder = convNet(out_channel, in_channels, mode='decoder', **kwargs)

    def forward(self, x):
        x_enc = self.encoder(x)
        latent = self.fc_encoder(x_enc)
        x_hat = None
        if hasattr(self, 'decoder'):
            x_hat = self.fc_decoder(latent)
            x_hat = self.decoder(x_hat)
        return x_hat, latent

    def _calculate_latent_size(self, in_channels, segment_size):
        dummy_input = torch.zeros(1, in_channels, segment_size)
        dummy_output = self.encoder(dummy_input)
        return dummy_output.size(1), dummy_output.size(2)


def convNet(in_channels, out_channels, hidden=128, depth=4, growth=2,
            kernel_size=4, stride=2,
            batch_norm=False, mode='encoder'):
    model = nn.Conv1d if mode == 'encoder' else nn.ConvTranspose1d
    dims = [in_channels]
    if mode == 'encoder':
        dims += ([int(round(hidden * growth ** k)) for k in range(depth)])
    elif mode == 'decoder':
        hidden = hidden * growth ** (depth-2)
        dims += ([int(round(hidden // growth ** k)) for k in range(depth)])
    if out_channels:
        dims[-1] = out_channels
    layers = []
    for chin, chout in zip(dims[:-1], dims[1:]):
        layers.append(model(chin, chout, kernel_size=kernel_size, stride=stride))  # TODO: for strides greater than 1, we need to use padding in every 3 layers and so to keep the same size.
        if batch_norm:
            layers.append(nn.BatchNorm1d(chout))
        layers.append(nn.ReLU())

    return nn.Sequential(*layers)
