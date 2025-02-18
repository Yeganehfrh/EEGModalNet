import numpy as np
import math
import typing as tp
from typing import List, Union
# import mne
import torch
from torch import nn
import keras
from keras import layers, regularizers, ops


class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size, groups, kernel_initializer, activation='relu', **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.groups = groups
        self.activation = activation
        self.conv1 = layers.Conv1D(filters, 3, padding='same', groups=groups, kernel_initializer=kernel_initializer, activation=activation)
        self.conv2 = layers.Conv1D(filters, 3, padding='same', groups=groups, dilation_rate=2, kernel_initializer=kernel_initializer, activation=activation)
        self.conv3 = layers.Conv1D(filters, 3, padding='same', groups=groups, dilation_rate=4, kernel_initializer=kernel_initializer)
        self.activation_layer = layers.Activation(activation)

    def build(self, input_shape):
        self.conv1.build(input_shape)

        conv1_output_shape = self.conv1.compute_output_shape(input_shape)
        self.conv2.build(conv1_output_shape)

        conv2_output_shape = self.conv2.compute_output_shape(conv1_output_shape)
        self.conv3.build(conv2_output_shape)

        super(ResidualBlock, self).build(input_shape)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = layers.add([x, inputs])  # shortcut connection
        return self.activation_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "kernel_initializer": self.kernel_initializer,
            "activation": self.activation,
            "groups": self.groups
        })
        return config


# transformer encoder based on example on https://keras.io/examples/timeseries/timeseries_classification_transformer/
class TransformerEncoder(layers.Layer):
    def __init__(self, feature_dim, head_size, num_heads, ff_dim, dropout=0.0):
        super(TransformerEncoder, self).__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.head_size, dropout=self.dropout
        )
        self.dropout_layer = layers.Dropout(self.dropout)
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)

        self.conv1 = layers.Conv1D(filters=self.ff_dim, kernel_size=1, activation="relu")
        self.dropout_layer2 = layers.Dropout(self.dropout)
        self.conv2 = layers.Conv1D(filters=feature_dim, kernel_size=1)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x = self.attention(inputs, inputs)
        x = self.dropout_layer(x)
        x = self.layer_norm1(x)
        res = x + inputs

        x = self.conv1(res)
        x = self.dropout_layer2(x)
        x = self.conv2(x)
        x = self.layer_norm2(x)
        return x + res


class PositionGetter:
    """PositionGetter class is from Défossez et al. 2022 (https://github.com/facebookresearch/brainmagick)"""

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
    This class is taken from Défossez et al. 2022 (https://github.com/facebookresearch/brainmagick):
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
        n_freqs = (self.dimension // 2)**0.5
        freqs_y = torch.arange(n_freqs, device=positions.device)
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
    """ChannelMerger class is based from Défossez et al. 2022 (https://github.com/facebookresearch/brainmagick)"""
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

    def forward(self, eeg, sub, positions):
        eeg = eeg.permute(0, 2, 1)
        B, C, T = eeg.shape
        eeg = eeg.clone()
        # positions = self.position_getter.get_positions(batch)
        embedding = self.embedding(positions)
        # score_offset = torch.zeros(B, C, device=eeg.device)
        # score_offset[self.position_getter.is_invalid(positions)] = float('-inf')

        if self.training and self.dropout:
            center_to_ban = torch.rand(2, device=eeg.device)
            radius_to_ban = self.dropout
            banned = (positions - center_to_ban).norm(dim=-1) <= radius_to_ban
            score_offset[banned] = float('-inf')

        if self.per_subject:
            _, cout, pos_dim = self.heads.shape
            subject = sub
            heads = self.heads.gather(0, subject.view(-1, 1, 1).expand(-1, cout, pos_dim))
        else:
            heads = self.heads.unsqueeze(0).repeat(B, 1, 1)

        scores = torch.einsum("bcd,bod->boc", embedding, heads)
        # scores += score_offset[:, None]
        if keras.mixed_precision.global_policy().name == 'mixed_float16':
            weights = torch.softmax(scores, dim=2, dtype=torch.float16)
        else:
            weights = torch.softmax(scores, dim=2)
        out = torch.einsum("bct,boc->bot", eeg, weights)  # It's in fact "bct,bcc->bct" that's why it doesn't raise an error
        if self.training and self.usage_penalty > 0.:
            usage = weights.mean(dim=(0, 1)).sum()
            self._penalty = self.usage_penalty * usage
        return out.permute(0, 2, 1)


class SubjectLayers(nn.Module):
    """subject layer is based on Défossez et al. 2022 (https://github.com/facebookresearch/brainmagick)"""
    def __init__(self, in_channels: int, out_channels: int, n_subjects: int, init_id: bool = False):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_subjects, in_channels, out_channels))
        if init_id:
            assert in_channels == out_channels
            self.weights.data[:] = torch.eye(in_channels)[None]
        self.weights.data *= 1 / in_channels**0.5

    def forward(self, x, subjects):
        _, C, D = self.weights.shape  # n_subjects, channels_in, channels_out
        weights = self.weights.gather(0, subjects.view(-1, 1, 1).expand(-1, C, D))  # batch size, channels_in, channels_out
        if keras.mixed_precision.global_policy().name == 'mixed_float16':
            weights = weights.half()
        x = torch.einsum("bct,bcd->bdt", x.permute(0, 2, 1), weights)
        return x.permute(0, 2, 1)

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
              interpolation: str,
              negative_slope: float = 0.2,
              kernel_initializer: str = 'glorot_uniform',
              batch_norm: bool = True) -> List[layers.Layer]:
    lyrs = []
    for i, (filter, kernel_size) in enumerate(zip(filters, kernel_sizes), 1):
        if upsampling[i - 1]:
            lyrs.append(CustomUpSampling1D(2, method=interpolation))
        lyrs.append(layers.Conv1D(filter, kernel_size, stride, padding, kernel_initializer=kernel_initializer, name=f'conv_{i}'))
        if batch_norm:
            lyrs.append(layers.BatchNormalization(name=f'bn_{i}'))
        lyrs.append(layers.LeakyReLU(negative_slope=negative_slope, name=f'leaky_relu_{i}'))
    return lyrs


# Transformer Encoder Block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    # Multi-Head Self-Attention
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)

    # Residual Connection + Layer Normalization
    x = layers.Add()([x, inputs])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Feed-Forward Network
    x_ff = layers.Dense(ff_dim, activation='relu')(x)
    x_ff = layers.Dense(inputs.shape[-1])(x_ff)

    # Residual Connection + Layer Normalization
    x = layers.Add()([x, x_ff])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    return x


# EEG Model for n-Channel Data
def build_eeg_transformer(sequence_length, embed_dim, num_heads, ff_dim, num_layers, n_channels):
    # Input layer for EEG data (n-channel input)
    inputs = layers.Input(shape=(sequence_length, n_channels))

    # Linear projection from 4 channels to embed_dim
    x = layers.Dense(embed_dim)(inputs)

    # Positional Embedding
    x = PositionalEmbedding(sequence_length, embed_dim)(x)

    # Stack multiple transformer encoders
    for _ in range(num_layers):
        x = transformer_encoder(x, head_size=embed_dim, num_heads=num_heads, ff_dim=ff_dim)

    # Global average pooling before final classification or regression
    x = layers.GlobalAveragePooling1D()(x)

    # Output layer (assuming regression or binary classification)
    outputs = layers.Dense(2, activation='sigmoid')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


class CustomUpSampling1D(layers.Layer):
    def __init__(self, size=2, method='bilinear', **kwargs):
        super(CustomUpSampling1D, self).__init__(**kwargs)
        self.size = size
        self.method = method

    def call(self, inputs):
        # Expand dimensions to 2D (batch, time, 1) -> (batch, time, width=1, channels)
        inputs_expanded = ops.expand_dims(inputs, axis=2)

        # Apply resize operation with the chosen interpolation method
        upsampled = ops.image.resize(inputs_expanded,
                                     size=[inputs.shape[1] * self.size, 1],
                                     interpolation=self.method)

        # Remove the width dimension and return (batch, time * size, channels)
        return ops.squeeze(upsampled, axis=2)

    def get_config(self):
        config = super().get_config()
        config.update({
            "size": self.size,
            "method": self.method,
        })
        return config


class Classifier(keras.Model):
    def __init__(self, feature_dim, l2_lambda=0.01, dropout_rate=0.4):
        super(Classifier, self).__init__()

        self.conv1 = ResidualBlock(feature_dim * 4, 5, activation='relu')
        self.conv2 = layers.Conv1D(2, 5, padding='same', activation='relu', kernel_regularizer=regularizers.L2(l2_lambda))
        self.conv3 = layers.Conv1D(1, 5, padding='same', activation='relu', kernel_regularizer=regularizers.L2(l2_lambda))
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.L2(l2_lambda))
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.L2(l2_lambda))
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dense3 = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.L2(l2_lambda))
        self.dropout3 = layers.Dropout(dropout_rate)
        self.dense4 = layers.Dense(8, activation='relu', kernel_regularizer=regularizers.L2(l2_lambda))
        self.dropout4 = layers.Dropout(dropout_rate)
        self.output_layer = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.L2(l2_lambda))

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        x = self.dropout3(x)
        x = self.dense4(x)
        x = self.dropout4(x)
        return self.output_layer(x)


class LearnablePositionalEmbedding(layers.Layer):
    """
    A simple trainable positional embedding layer.
    Each position 'i' in the sequence has a learned embedding of dimension 'embedding_dim'.
    """
    def __init__(self, sequence_length: int = 128, embedding_dim: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.pos_emb = self.add_weight(
            name="pos_emb",
            shape=(sequence_length, embedding_dim),
            initializer="uniform",
            trainable=True,
        )

    def call(self, inputs):
        """
        inputs: (batch_size, time, embedding_dim)
        We'll add the positional embeddings up to 'time' steps.
        """
        seq_len = inputs.shape[1]
        # slice the first 'seq_len' embeddings (if seq_len < sequence_length)
        pos_slice = self.pos_emb[None, :seq_len, :]
        return inputs + pos_slice

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "embedding_dim": self.embedding_dim,
        })
        return config


# Custom Positional Embedding Layer
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)
        self.sequence_length = sequence_length

    def call(self, inputs):
        positions = torch.arange(start=0, end=self.sequence_length, step=1)
        position_embeddings = self.position_embeddings(positions)
        return inputs + position_embeddings


class SinePositionalEncoding(layers.Layer):
    """
    A sinusoidal positional encoding as introduced in the original Transformer paper.
    This is non-trainable and encodes positions using sines and cosines of different frequencies.
    """
    def __init__(self, max_len: int = 1024, embedding_dim: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.embedding_dim = embedding_dim

        # Precompute the positional encodings in a [max_len, embedding_dim] array
        pe = np.zeros((max_len, embedding_dim))
        position = np.arange(0, max_len)[:, np.newaxis]  # shape (max_len, 1)
        div_term = np.exp(
            -math.log(10000.0) * (np.arange(0, embedding_dim, 2) / embedding_dim)
        )
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        # Convert to constant so we don't recalc every call
        self.register_buffer('pe', torch.from_numpy(pe))   # shape (max_len, embedding_dim)

    def call(self, inputs):
        """
        inputs: (batch, time, embedding_dim)
        """
        seq_len = inputs.shape[1]  # actual time dimension
        # slice the first 'seq_len' positions: shape (seq_len, embedding_dim)
        pos_slice = self.pe[:seq_len, :]
        if inputs.device != pos_slice.device:
            pos_slice = pos_slice.float().to(inputs.device)
        # broadcast-add to (batch, seq_len, embedding_dim)
        return inputs + pos_slice[None, :, :]


class SelfAttention1D(layers.Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super(SelfAttention1D, self).__init__(**kwargs)

        self.num_heads = num_heads
        self.key_dim = key_dim

        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.layer_norm = layers.LayerNormalization()

    def build(self, input_shape):
        self.attention.build(input_shape, input_shape)
        self.layer_norm.build(input_shape)
        super(SelfAttention1D, self).build(input_shape)

    def call(self, inputs):
        x = self.attention(inputs, inputs)  # (query=x, value=x)
        x = x + inputs
        x = self.layer_norm(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
        })
        return config


def SingleConvBlock(filter: int,
                    kernel_size: Union[int, tuple],
                    upsampling: Union[bool, int],
                    stride: int = 1,
                    padding: str = 'same',
                    negative_slope: float = 0.2,
                    kernel_initializer: str = 'glorot_uniform',
                    batch_norm: bool = True,
                    activation: bool = True) -> List[layers.Layer]:
    lyrs = []
    if upsampling:
        lyrs.append(layers.UpSampling1D(2))
    lyrs.append(layers.Conv1D(filter, kernel_size, stride, padding, kernel_initializer=kernel_initializer))
    if batch_norm:
        lyrs.append(layers.BatchNormalization())
    if activation:
        lyrs.append(layers.LeakyReLU(negative_slope=negative_slope))
    return lyrs


class SkipBlock(layers.Layer):
    def __init__(self, filter, kernel_size, kernel_initializer, **kwargs) -> None:
        super(SkipBlock, self).__init__(**kwargs)
        # First residual sub-block: block1 and block2
        self.block1 = keras.Sequential(SingleConvBlock(filter, kernel_size, upsampling=True, kernel_initializer=kernel_initializer))
        self.block2 = keras.Sequential(SingleConvBlock(filter, kernel_size, upsampling=False, kernel_initializer=kernel_initializer))

        # Second residual sub-block: block3 and block4
        self.block3 = keras.Sequential(SingleConvBlock(filter, kernel_size, upsampling=True, kernel_initializer=kernel_initializer))
        self.block4 = keras.Sequential(SingleConvBlock(filter, kernel_size, upsampling=False, kernel_initializer=kernel_initializer))

        self.activation = layers.LeakyReLU(negative_slope=0.2)

    def call(self, inputs):
        # First residual connection
        residual1 = self.block1(inputs)
        out1 = self.block2(residual1)
        out1 = self.activation(layers.add([out1, residual1]))

        # Second residual connection
        residual2 = self.block3(out1)
        out2 = self.block4(residual2)
        out2 = self.activation(layers.add([out2, residual2]))
        return out2
