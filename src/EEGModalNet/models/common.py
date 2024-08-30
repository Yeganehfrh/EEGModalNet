import math
import typing as tp
from typing import List, Union
import mne
import torch
from torch import nn
import keras
from keras import layers


class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size, activation='relu', **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = layers.Conv1D(filters, kernel_size, padding='same', activation=activation)
        self.conv2 = layers.Conv1D(filters // 2, kernel_size, padding='same', activation=activation,)
        self.conv3 = layers.Conv1D(filters // 4, kernel_size, padding='same')
        self.activation = layers.Activation(activation)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = layers.add([x, inputs])  # shortcut connection
        x = self.activation(x)
        return x


class PositionGetter:
    INVALID = -0.1

    def __init__(self) -> None:
        self._cache: tp.Dict[int, torch.Tensor] = {}
        self._invalid_names: tp.Set[str] = set()

    def get_recording_layout(self, info) -> torch.Tensor:
        layout = mne.channels.find_layout(info)
        positions = torch.full((len(info.ch_names), 2), self.INVALID)
        x, y = layout.pos[:, :2].T
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        positions[:, 0] = x
        positions[:, 1] = y
        return positions

    def get_positions(self, batch):
        eeg, _, info = batch
        B, C, _ = eeg.shape
        positions = torch.full((B, C, 2), self.INVALID, device=eeg.device)
        for idx in range(len(batch)):
            # recording = batch._recordings[idx]
            rec_pos = self.get_recording_layout(info)
            positions[idx, :len(rec_pos)] = rec_pos.to(eeg.device)
        return positions

    def is_invalid(self, positions):
        return (positions == self.INVALID).all(dim=-1)


class FourierEmb(nn.Module):
    """
    Fourier positional embedding.
    Unlike trad. embedding this is not using exponential periods
    for cosines and sinuses, but typical `2 pi k` which can represent
    any function over [0, 1]. As this function would be necessarily periodic,
    we take a bit of margin and do over [-0.2, 1.2].
    """
    def __init__(self, dimension: int = 256, margin: float = 0.2):
        super().__init__()
        n_freqs = (dimension // 2)**0.5
        assert int(n_freqs ** 2 * 2) == dimension
        self.dimension = dimension
        self.margin = margin

    def forward(self, positions):
        *O, D = positions.shape
        assert D == 2
        *O, D = positions.shape
        n_freqs = (self.dimension // 2)**0.5
        freqs_y = torch.arange(n_freqs).to(positions)
        freqs_x = freqs_y[:, None]
        width = 1 + 2 * self.margin
        positions = positions + self.margin
        p_x = 2 * math.pi * freqs_x / width
        p_y = 2 * math.pi * freqs_y / width
        positions = positions[..., None, None, :]
        loc = (positions[..., 0] * p_x + positions[..., 1] * p_y).view(*O, -1)
        emb = torch.cat([
            torch.cos(loc),
            torch.sin(loc),
        ], dim=-1)
        return emb


class ChannelMerger(nn.Module):
    def __init__(self, chout: int, pos_dim: int = 256,
                 dropout: float = 0, usage_penalty: float = 0.,
                 n_subjects: int = 200, per_subject: bool = False):
        super().__init__()
        assert pos_dim % 4 == 0
        self.position_getter = PositionGetter()
        self.per_subject = per_subject
        if self.per_subject:
            self.heads = nn.Parameter(torch.randn(n_subjects, chout, pos_dim, requires_grad=True))
        else:
            self.heads = nn.Parameter(torch.randn(chout, pos_dim, requires_grad=True))
        self.heads.data /= pos_dim ** 0.5
        self.dropout = dropout
        self.embedding = FourierEmb(pos_dim)
        self.usage_penalty = usage_penalty
        self._penalty = torch.tensor(0.)

    @property
    def training_penalty(self):
        return self._penalty.to(next(self.parameters()).device)

    def forward(self, eeg, positions):
        eeg = eeg.permute(0, 2, 1)
        B, C, T = eeg.shape
        eeg = eeg.clone()
        # positions = self.position_getter.get_positions(batch)
        embedding = self.embedding(positions)
        score_offset = torch.zeros(B, C, device=eeg.device)
        # score_offset[self.position_getter.is_invalid(positions)] = float('-inf')

        if self.training and self.dropout:
            center_to_ban = torch.rand(2, device=eeg.device)
            radius_to_ban = self.dropout
            banned = (positions - center_to_ban).norm(dim=-1) <= radius_to_ban
            score_offset[banned] = float('-inf')

        if self.per_subject:
            _, cout, pos_dim = self.heads.shape
            subject = batch.subject_index
            heads = self.heads.gather(0, subject.view(-1, 1, 1).expand(-1, cout, pos_dim))
        else:
            heads = self.heads.unsqueeze(0).repeat(B, 1, 1)

        scores = torch.einsum("bcd,bod->boc", embedding, heads)
        scores += score_offset[:, None]
        weights = torch.softmax(scores, dim=2)
        out = torch.einsum("bct,boc->bot", eeg, weights)
        if self.training and self.usage_penalty > 0.:
            usage = weights.mean(dim=(0, 1)).sum()
            self._penalty = self.usage_penalty * usage
        return out


class SubjectLayers(nn.Module):
    """Per subject linear layer."""
    def __init__(self, in_channels: int, out_channels: int, n_subjects: int, init_id: bool = False):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_subjects, in_channels, out_channels))
        if init_id:
            assert in_channels == out_channels
            self.weights.data[:] = torch.eye(in_channels)[None]
        self.weights.data *= 1 / in_channels**0.5

    def forward(self, x, subjects):
        _, C, D = self.weights.shape
        weights = self.weights.gather(0, subjects.view(-1, 1, 1).expand(-1, C, D))
        x_ = torch.einsum("bct,bcd->bdt", x, weights)
        return x_

    def __repr__(self):
        S, C, D = self.weights.shape
        return f"SubjectLayers({C}, {D}, {S})"


class SubjectLayers_v2(nn.Module):
    """Per subject linear layer."""
    def __init__(self, n_subjects: int, emb_dim: int):
        super().__init__()
        self.sub_emb = nn.Embedding(n_subjects, emb_dim)

    def forward(self, x, subjects):
        weights = self.sub_emb(subjects)
        x_ = torch.einsum("btc,bcd->btc", x, weights)
        return x_


def convBlock(filters: List[int],
              kernel_sizes: List[Union[int, tuple]],
              upsampling: List[Union[bool, int]],
              stride: int,
              padding: str,
              negative_slope: float = 0.2,
              kernel_initializer='glorot_uniform',
              batch_norm: bool = True) -> List[layers.Layer]:
    lyrs = []
    for i, (filter, kernel_size) in enumerate(zip(filters, kernel_sizes), 1):
        if upsampling[i - 1]:
            lyrs.append(layers.UpSampling1D(2, name=f'upsample_{i}'))
        lyrs.append(layers.Conv1D(filter, kernel_size, stride, padding, kernel_initializer=kernel_initializer, name=f'conv_{i}'))
        if batch_norm:
            lyrs.append(layers.BatchNormalization(name=f'bn_{i}'))
        lyrs.append(layers.LeakyReLU(negative_slope=negative_slope, name=f'leaky_relu_{i}'))
    return lyrs
