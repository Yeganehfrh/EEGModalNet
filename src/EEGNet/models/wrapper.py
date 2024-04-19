import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchaudio.transforms as T
from src.EEGNet.models.commonBlocks import ChannelMerger, SubjectLayers, Classifier
import torchmetrics.functional as tmf
from src.EEGNet.models.autoencoders import RNNAutoencoder, ConvAutoencoder, MLPAutoencoder


class Wrapper(pl.LightningModule):
    def __init__(self,
                 # general structure
                 encoderArc='CNN',
                 n_timepoints=512,
                 n_channels=61,
                 n_subjects=200,
                 n_classes=2,
                 hidden=128,
                 depth=1,
                 dropout=0.2,
                 use_channel_merger=True,
                 use_subject_layers=True,
                 use_classifier=True,
                 use_decoder=True,
                 use_1x1_conv=True,
                 # cnn
                 n_embeddings=32,
                 out_channel=512,
                 kernel_size=4,
                 stride=2,
                 # rnn
                 rnn_dropout=0.0,
                 rnn_bidirectional=False,
                 # MLP
                 # other
                 n_fft=None,
                 cross_val=False,
                 joint_embedding=False,
                 variational=False,):

        super().__init__()
        self.save_hyperparameters()
        self.cross_val = cross_val
        self.encoderArc = encoderArc
        self.joint_embedding = joint_embedding
        self.use_decoder = use_decoder
        assert not (use_decoder and joint_embedding), "Cross validation and joint embedding cannot be used together"
        assert encoderArc in ['CNN', 'MLP', 'RNN'], "Encoder must be either CNN, RNN, MLP"

        # Fourier positional embedding
        if use_channel_merger:
            self.pos_emb = ChannelMerger(
                chout=n_channels, pos_dim=288, n_subjects=n_subjects
            )  # TODO: check if this is the right dimension

        # 1 X 1 convolution
        if use_1x1_conv:
            self.cov11 = nn.Conv1d(n_channels, n_channels, kernel_size=1)

        # subject layers
        if use_subject_layers:
            self.subject_layers = SubjectLayers(in_channels=n_channels,
                                                out_channels=n_channels,
                                                n_subjects=n_subjects)

        # transform to frequency domain
        if n_fft is not None:
            self.n_fft = n_fft
            self.stft = T.Spectrogram(n_fft=n_fft, hop_length=128,
                                      power=1, normalized=False)

        # dropout
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

        if encoderArc == 'CNN':
            kwargs = {
                'hidden': hidden,
                'depth': depth,
                'growth': 2,
                'kernel_size': kernel_size,
                'stride': stride,
            }
            self.encoder = ConvAutoencoder(n_channels, out_channel, n_embeddings,
                                           n_timepoints, use_decoder, **kwargs)

        elif encoderArc == 'RNN':
            self.encoder = RNNAutoencoder(n_channels, hidden, depth,
                                          n_embeddings, rnn_dropout,
                                          rnn_bidirectional,
                                          use_decoder, variational)
        elif encoderArc == 'MLP':
            kwargs = {
                'hidden': hidden,
                'depth': depth,
                'growth': 2,
            }
            self.encoder = MLPAutoencoder(n_timepoints*n_channels, n_embeddings, use_decoder, **kwargs)

        if use_classifier:
            self.classifier = Classifier(n_embeddings, n_classes)

    def forward(self, batch):
        x, sub, pos, _ = batch
        x = x.permute(0, 2, 1)
        if hasattr(self, 'pos_emb'):
            x = self.pos_emb(x, pos)
        if hasattr(self, 'cov11'):
            x = self.cov11(x)
        if hasattr(self, 'subject_layers'):
            x = self.subject_layers(x, sub)
        if hasattr(self, 'stft'):
            x = self.stft(x)
            x = x.mean(dim=-1)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)

        if self.encoderArc == 'RNN':
            x = x.permute(0, 2, 1)

        x_hat, h, mu, log_var = self.encoder(x)

        y_hat = None
        if hasattr(self, 'classifier'):
            y_hat = self.classifier(h)

        return x_hat, h, y_hat, mu, log_var

    def training_step(self, batch, batch_idx):
        loss = 0
        if self.cross_val:
            return self.training_step_kfold(batch, batch_idx)
        x, _, _, y = batch
        x_hat, h_hat, y_hat, mu, log_var = self(batch)
        if self.encoderArc != 'RNN':
            x = x.permute(0, 2, 1)
        if self.joint_embedding:
            if hasattr(self, 'stft'):
                x = self.stft(x)
                x = x.mean(dim=-1)
            h = self.encoder(x)
            loss_rec = nn.functional.mse_loss(h_hat, h)
            self.log('train/loss_recon', loss_rec)
            loss += loss_rec
        if self.encoder.variational:
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            self.log('train/kl_loss', kl_loss)
            loss += kl_loss
        if hasattr(self, 'classifier'):
            loss_class = nn.functional.cross_entropy(y_hat, y)
            self.log('train/loss_cls', loss_class)
            loss += loss_class
            # log accuracy
            accuracy = tmf.accuracy(y_hat, y, task='binary', num_classes=2)
            self.log('train/acc_tmf', accuracy)
        if self.use_decoder:
            loss_rec = nn.functional.mse_loss(x_hat, x)
            self.log('train/loss_recon', loss_rec)
            loss += loss_rec
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = 0
        x, _, _, y = batch
        x_hat, h_hat, y_hat, mu, log_var = self(batch)
        if self.encoderArc != 'RNN':
            x = x.permute(0, 2, 1)
        if self.joint_embedding:
            if hasattr(self, 'stft'):
                x = self.stft(x)
                x = x.mean(dim=-1)
            h = self.encoder(x)
            loss_rec = nn.functional.mse_loss(h_hat, h)
            self.log('val/loss_recon', loss_rec)
            loss += loss_rec
        if self.encoder.variational:
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            self.log('val/kl_loss', kl_loss)
            loss += kl_loss
        if hasattr(self, 'classifier'):
            loss_class = nn.functional.cross_entropy(y_hat, y)
            self.log('val/loss_cls', loss_class)
            loss += loss_class
            accuracy = tmf.accuracy(y_hat, y, task='binary', num_classes=2)
            self.log('val/acc', accuracy)
        if self.use_decoder:
            loss_rec = nn.functional.mse_loss(x_hat, x)
            self.log('val/loss_recon', loss_rec)
            loss += loss_rec
        self.log('val/loss', loss)
        return loss

    def training_step_kfold(self, batch, batch_idx):
        loss = 0
        for fold_batch in batch:
            x, _, _, y = fold_batch
            x_hat, h_hat, y_hat = self(fold_batch)
            if self.encoderArc != 'RNN':
                x = x.permute(0, 2, 1)
            if self.joint_embedding:
                if hasattr(self, 'stft'):
                    x = self.stft(x)
                    x = x.mean(dim=-1)
                h = self.encoder(x)
                loss_rec = nn.functional.mse_loss(h_hat, h)
                self.log('train/loss_recon', loss_rec)
                loss += loss_rec
            if hasattr(self, 'classifier'):
                loss_class = nn.functional.cross_entropy(y_hat, y)
                self.log('train/loss_cls', loss_class)
                loss += loss_class
                # log accuracy
                accuracy = tmf.accuracy(y_hat, y, task='binary', num_classes=2)
                self.log('train/acc_tmf', accuracy)
            if self.use_decoder:
                loss_rec = nn.functional.mse_loss(x_hat, x)
                self.log('train/loss_recon', loss_rec)
                loss += loss_rec
        self.log('train/loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class SeperateClassifier(pl.LightningModule):
    def __init__(self,
                 pretrained_encoder_checkpoint_path,
                 n_labels):
        super().__init__()

        self.save_hyperparameters()

        self.encoder = Wrapper.load_from_checkpoint(pretrained_encoder_checkpoint_path)
        self.embeddings_dim = self.encoder.encoder.time_embedding_dim
        self.model = nn.Sequential(
            nn.Linear(self.embeddings_dim, self.embeddings_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embeddings_dim // 2, n_labels),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.encoder.encoder(x)
        y_cls = self.model(h)
        return y_cls

    def training_step(self, batch, batch_idx):
        x, _, _, y = batch
        x = x.permute(0, 2, 1)
        y_hat = self(x)
        accuracy = tmf.accuracy(y_hat, y, task='binary', num_classes=2)
        loss_cls = nn.functional.cross_entropy(y_hat, y)
        self.log('train/accuracy', accuracy)
        self.log('train/loss_cls', loss_cls)
        return loss_cls

    def validation_step(self, batch, batch_idx):
        x, _, _, y = batch
        x = x.permute(0, 2, 1)
        y_hat = self(x)
        accuracy = tmf.accuracy(y_hat, y, task='binary', num_classes=2)
        loss_cls = nn.functional.cross_entropy(y_hat, y)
        self.log('val/accuracy', accuracy)
        self.log('val/loss_cls', loss_cls)
        return loss_cls

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def SpaceTimeEncoder(n_channels, space_embedding_dim, time_embedding_dim, kernel_size=1,
                     first_encode='Time'):

    assert first_encode in ['Time', 'Space'], "First encoding must be either Time or Space"

    space_encoder = nn.Sequential(
            nn.Conv1d(n_channels, space_embedding_dim * 2, kernel_size),
            nn.ReLU(),
            nn.Conv1d(space_embedding_dim * 2, space_embedding_dim, kernel_size),
            nn.ReLU())

    time_encoder = nn.LSTM(
            space_embedding_dim,
            time_embedding_dim,
            batch_first=True)

    if first_encode == 'Time':
        return nn.Sequential(time_encoder, space_encoder)
    elif first_encode == 'Space':
        return nn.Sequential(space_encoder, time_encoder)


def SpaceTimeDecoder(n_channels, space_embedding_dim, time_embedding_dim, kernel_size=1):
    time_decoder = nn.LSTM(
        time_embedding_dim,
        space_embedding_dim,
        batch_first=True)

    # spatial decoder
    space_decoder = nn.Sequential(
        nn.ConvTranspose1d(space_embedding_dim, space_embedding_dim * 2, 1, stride=1),
        nn.ReLU(),
        nn.ConvTranspose1d(space_embedding_dim * 2, n_channels, 1, stride=1),
        nn.ReLU())
    return nn.Sequential(time_decoder, space_decoder)
