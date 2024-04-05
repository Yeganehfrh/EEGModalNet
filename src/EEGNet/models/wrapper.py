import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchaudio.transforms as T
from src.EEGNet.models.commonBlocks import ChannelMerger, SubjectLayers, Classifier
import torchmetrics.functional as tmf


class Wrapper(pl.LightningModule):
    def __init__(self,
                 # parameters
                 n_channels=61,
                 n_subjects=200,
                 n_classes=2,
                 # encoder-decoder
                 encoder='CNN',
                 decoder='CNN',
                 embedded_time_dim=62,
                 n_embeddings=32,
                 # MLP
                 n_timepoints=512,
                 n_layers=3,
                 # layers
                 use_channel_merger=True,
                 use_subject_layers=True,
                 use_classifier=True,
                 use_decoder=True,
                 use_1x1_conv=True,
                 dropout=0.5,
                 # other
                 n_fft=None,
                 cross_val=False,
                 joint_embedding=False):

        super().__init__()
        self.save_hyperparameters()
        self.cross_val = cross_val
        self.joint_embedding = joint_embedding
        assert not (use_decoder and joint_embedding), "Cross validation and joint embedding cannot be used together"
        assert encoder in ['CNN', 'MLPTime', 'MLPChannels'], "Encoder must be either CNN, MLPTime or MLPChannels"
        assert decoder in ['CNN', 'MLPTime', 'MLPChannels'], "Decoder must be either CNN, MLPTime, or MLPChannels"

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
            self.subject_layers = SubjectLayers(in_channels=n_channels, out_channels=n_channels, n_subjects=n_subjects)

        # transform to frequency domain
        if n_fft is not None:
            self.n_fft = n_fft
            self.stft = T.Spectrogram(n_fft=n_fft, hop_length=128,
                                      power=1, normalized=False)

        # dropout
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

        if encoder == 'CNN':
            self.encoder = CNNEncoder(n_channels, n_embeddings, embedded_time_dim)
        elif encoder == 'MLPTime':
            self.encoder = mlpEncoder(n_timepoints, n_embeddings, n_layers)
        elif encoder == 'MLPChannels':
            self.encoder = mlpEncoder(n_channels, n_embeddings, n_layers)

        if use_classifier:
            self.classifier = Classifier(n_embeddings, n_classes)

        if use_decoder:
            if decoder == 'CNN':
                self.decoder = CNNDecoder(n_channels, embedded_time_dim, n_embeddings)
            elif decoder == 'MLPTime':
                self.decoder = mlpDecoder(n_timepoints, n_embeddings, n_layers)
            elif decoder == 'MLPChannels':
                self.decoder = mlpDecoder(n_channels, n_embeddings, n_layers)

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

        h = self.encoder(x)

        y_hat = None
        if hasattr(self, 'classifier'):
            y_hat = self.classifier(h)

        x_hat = None
        if hasattr(self, 'decoder'):
            x_hat = self.decoder(h)

        return x_hat, h, y_hat

    def training_step(self, batch, batch_idx):
        loss = 0
        if self.cross_val:
            return self.training_step_kfold(batch, batch_idx)
        x, _, _, y = batch
        x_hat, h_hat, y_hat = self(batch)
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
        if hasattr(self, 'decoder'):
            loss_rec = nn.functional.mse_loss(x_hat, x)
            self.log('train/loss_recon', loss_rec)
            loss += loss_rec
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = 0
        x, _, _, y = batch
        x_hat, h_hat, y_hat = self(batch)
        x = x.permute(0, 2, 1)
        if self.joint_embedding:
            if hasattr(self, 'stft'):
                x = self.stft(x)
                x = x.mean(dim=-1)
            h = self.encoder(x)
            loss_rec = nn.functional.mse_loss(h_hat, h)
            self.log('val/loss_recon', loss_rec)
            loss += loss_rec
        if hasattr(self, 'classifier'):
            loss_class = nn.functional.cross_entropy(y_hat, y)
            self.log('val/loss_cls', loss_class)
            loss += loss_class
            accuracy = tmf.accuracy(y_hat, y, task='binary', num_classes=2)
            self.log('val/acc', accuracy)
        if hasattr(self, 'decoder'):
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
            if hasattr(self, 'decoder'):
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


def CNNEncoder(n_channels, n_embeddings, embedded_time_dim):
    return nn.Sequential(
                 nn.Conv1d(n_channels, n_channels * 2, kernel_size=4, stride=2),
                 nn.ReLU(),
                 nn.Conv1d(n_channels * 2, n_channels * 4, kernel_size=4, stride=2),
                 nn.ReLU(),
                 nn.Conv1d(n_channels * 4, n_channels * 8, kernel_size=4, stride=2),
                 nn.ReLU(),
                 nn.Flatten(),
                 nn.Linear(n_channels * 8 * embedded_time_dim, n_embeddings)
                )


def CNNDecoder(n_channels, embedded_time_dim, n_embeddings):
    return nn.Sequential(
                nn.Linear(n_embeddings, n_channels * 8 * embedded_time_dim),
                nn.Unflatten(dim=1, unflattened_size=(n_channels * 8, embedded_time_dim)),
                nn.ReLU(),
                nn.ConvTranspose1d(n_channels * 8, n_channels * 4, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.ConvTranspose1d(n_channels * 4, n_channels * 2, kernel_size=4, stride=2, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(n_channels * 2, n_channels, kernel_size=4, stride=2),
                nn.ReLU()
        )


def mlpEncoder(n_features, n_embeddings, n_layers):
    assert n_layers > 0 and n_features > n_embeddings, "Invalid number of layers or embeddings"
    assert n_features // (2 ** n_layers) >= n_embeddings, "n_features must be greater than or equal to n_embeddings after passing through the layers"
    layers = []
    for i in range(n_layers):
        layers.append(nn.Linear(n_features, n_features // 2))
        layers.append(nn.ReLU())
        n_features = n_features // 2
    layers.append(nn.Linear(n_features, n_embeddings))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def mlpDecoder(n_features, n_embeddings, n_layers):
    assert n_layers > 0 and n_features > n_embeddings, "Invalid number of layers or embeddings"
    assert n_embeddings * 2 ** n_layers <= n_features, "n_embeddings * 2 ** n_layers must be less than or equal to n_features"
    layers = []
    for i in range(n_layers):
        layers.append(nn.Linear(n_embeddings, n_embeddings * 2))
        layers.append(nn.ReLU())
        n_embeddings = n_embeddings * 2
    layers.append(nn.Linear(n_embeddings, n_features))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)
