import numpy as np
import math
import typing as tp
from typing import List, Union
# import mne
import torch
from torch import nn
import keras
from keras import layers, regularizers, ops
import torch.nn.functional as F


class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size, groups, kernel_initializer, activation='relu', **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.groups = groups
        self.activation = activation
        self.conv1 = layers.Conv1D(filters, kernel_size, padding='same', groups=groups, kernel_initializer=kernel_initializer, activation=activation)
        self.conv2 = layers.Conv1D(filters, kernel_size, padding='same', groups=groups, dilation_rate=2, kernel_initializer=kernel_initializer, activation=activation)
        self.conv3 = layers.Conv1D(filters, kernel_size, padding='same', groups=groups, dilation_rate=4, kernel_initializer=kernel_initializer)
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


class StridedResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size, strides, kernel_initializer, activation='relu', **kwargs):
        super(StridedResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.strides = strides
        self.activation = activation
        self.conv1 = layers.Conv1D(filters, 3, padding='same', strides=strides, kernel_initializer=kernel_initializer, activation=activation)
        self.conv2 = layers.Conv1D(2 * filters, 3, padding='same', strides=strides, kernel_initializer=kernel_initializer, activation=activation)
        self.conv3 = layers.Conv1D(4 * filters, 3, padding='same', strides=strides, kernel_initializer=kernel_initializer)
        self.activation_layer = layers.Activation(activation)

    def call(self, inputs):
        # skip = inputs
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        # # match time dimension
        # if x.shape[1] != skip.shape[1]:
        #     skip = layers.MaxPool1D(pool_size=skip.shape[1] // x.shape[1])(skip)

        # # Match feature dimension
        # if x.shape[-1] != skip.shape[-1]:
        #     skip = layers.Conv1D(x.shape[-1], kernel_size=1, padding='same', name='skip_conv')(skip)

        # x = layers.add([x, skip])  # shortcut connection
        return self.activation_layer(x)


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
        return inputs + self.pos_emb

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
    def __init__(self, num_heads, key_dim, use_ffn=False, fn_inner_d=64, disable_attention=False, **kwargs):
        super(SelfAttention1D, self).__init__(**kwargs)

        self.num_heads = num_heads
        self.key_dim = key_dim
        self.disable_attention = disable_attention

        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.layer_norm = layers.LayerNormalization()

        if use_ffn:
            self.layer_norm_ffn = layers.LayerNormalization()
            self.ffn = keras.Sequential([
                layers.Dense(ffn_inner_d, activation='gelu'),
                layers.Dense(key_dim * num_heads),  # match the attention output dimension
            ])

    def build(self, input_shape):
        self.attention.build(input_shape, input_shape)
        self.layer_norm.build(input_shape)
        super(SelfAttention1D, self).build(input_shape)

    def call(self, inputs):
        if self.disable_attention:
            return inputs
    
        attn_output = self.attention(inputs, inputs)  # (query=x, value=x)
        x = self.layer_norm(inputs + attn_output)
        if hasattr(self, 'ffn'):
            ff_output = self.ffn(x)
            x = self.layer_norm_ffn(x + ff_output)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
        })
        return config


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


class TorchLinearUpsample1D(layers.Layer):
    """1D linear interpolation using torch.nn.functional.interpolate."""

    def __init__(self, scale_factor=2, **kwargs):
        super().__init__(**kwargs)
        self.scale_factor = scale_factor

    def call(self, x):
        x = x.permute(0, 2, 1)  #(B, C, T)

        # Linear 1D interpolation
        x = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="linear",
            align_corners=False
        )

        return x.permute(0, 2, 1)  #(B, T', C)


class HighPass1D(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        x_t = x.permute(0, 2, 1)
        B, C, T = x_t.shape

        # Fixed high-pass kernel [-1, 2, -1] / 2
        k = torch.tensor([-1.0, 2.0, -1.0],
                         device=x_t.device,
                         dtype=x_t.dtype) / 2.0
        k = k.view(1, 1, 3)           # (out_ch=1, in_ch=1, k)
        k = k.repeat(C, 1, 1)         # (out_ch=C, in_ch=1, k) for depthwise

        # Depthwise conv: groups=C
        y_t = torch.nn.functional.conv1d(
            x_t,
            k,
            bias=None,
            padding=1,
            groups=C,
        )
        y = y_t.permute(0, 2, 1)
        return y


class ChannelAttention(layers.Layer):
    def __init__(self, num_heads, key_dim, num_ch, use_norm=True, **kwargs):
        """
        num_heads: Number of attention heads.
        key_dim: Dimension of each attention head.
        use_norm: If True, apply layer normalization after the residual connection.
        """
        super(ChannelAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.num_ch = num_ch
        self.use_norm = use_norm
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.channel_pos_emb = LearnablePositionalEmbedding(num_ch, num_heads * key_dim)
        if self.use_norm:
            self.norm = layers.LayerNormalization(axis=-1)

    def call(self, inputs):
        x = inputs.permute(0, 2, 1)  # Transpose to shape (batch, channels, time)
        x = self.channel_pos_emb(x)
        attn_output = self.attention(x, x)  # Apply multi-head attention over channels (treating channels as tokens)
        attn_output = attn_output.permute(0, 2, 1)
        output = inputs + attn_output
        if self.use_norm:
            output = self.norm(output)
        return output

    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "use_norm": self.use_norm,
            "num_ch": self.num_ch
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


class SubjectLayers_FiLM(nn.Module):
    """FiLM-style subject layer."""
    def __init__(self, in_channels: int, out_channels: int, d_sub: int, init_id: bool = False):
        super().__init__()
        assert in_channels == out_channels, "FiLM version expects C_in == C_out"
        self.linear = nn.Linear(d_sub, 2 * in_channels)

        if init_id:
            # start near identity (γ≈1, β≈0)
            with torch.no_grad():
                self.linear.weight.zero_()
                self.linear.bias.zero_()

    def forward(self, x, subj_emb):
        w_dtype = self.linear.weight.dtype

        subj_emb = subj_emb.to(device=x.device, dtype=w_dtype)
        gamma_beta = self.linear(subj_emb)             # (B, 2C)
        gamma, beta = gamma_beta.chunk(2, dim=-1)      # (B, C), (B, C)

        # residual, small-gain FiLM
        gamma = 1.0 + 0.1 * gamma
        beta  = 0.1 * beta

        gamma = gamma.to(x.dtype).unsqueeze(1)  # (B, 1, C)
        beta  = beta.to(x.dtype).unsqueeze(1)   # (B, 1, C)

        return gamma * x + beta


class DualFiLMBlock(nn.Module):
    def __init__(self, n_channels: int, d_sub: int, d_state: int = 16, init_id: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.d_sub = d_sub
        self.d_state = d_state

        self.sub_film   = nn.Linear(d_sub,   2 * n_channels)
        self.state_film = nn.Linear(d_state, 2 * n_channels)

        # Optional learnable gates (lets the model downweight a conditioner if noisy)
        self.g_sub  = nn.Parameter(torch.tensor(1.0))
        self.g_state = nn.Parameter(torch.tensor(1.0))

        if init_id:
            with torch.no_grad():
                self.sub_film.weight.zero_();   self.sub_film.bias.zero_()
                self.state_film.weight.zero_(); self.state_film.bias.zero_()

    def forward(self, x, subj_emb, state_emb):
        """
        x:        (B, T, C)
        subj_emb: (B, d_sub)
        state_emb:(B, d_state)
        """
        xdtype = x.dtype
        device = x.device

        w_sub_dtype = self.sub_film.weight.dtype
        w_state_dtype = self.state_film.weight.dtype

        subj_emb  = subj_emb.to(device=device, dtype=w_sub_dtype)
        state_emb = state_emb.to(device=device, dtype=w_state_dtype)

        sub_params   = self.sub_film(subj_emb)      # (B, 2C)
        state_params = self.state_film(state_emb)   # (B, 2C)

        g_sub, b_sub       = sub_params.chunk(2, dim=-1)    # (B,C)
        g_state, b_state   = state_params.chunk(2, dim=-1)  # (B,C)

        gsub = self.g_sub.to(device=device, dtype=w_sub_dtype)
        gst  = self.g_state.to(device=device, dtype=w_sub_dtype)

        # bounded residual FiLM (your style)
        gamma = 1.0 + 0.1 * (gsub * g_sub + gst * g_state)
        beta  = 0.1 * (gsub * b_sub + gst * b_state)

        gamma = gamma.to(xdtype).unsqueeze(1)  # (B,1,C)
        beta  = beta.to(xdtype).unsqueeze(1)   # (B,1,C)
        return gamma * x + beta
    

class SubjectStateLayers_FiLM(nn.Module):
    """FiLM-style subject layer with additional EO/EC state conditioning."""
    def __init__(self, channels: int, d_sub: int, d_state: int = 16, init_id: bool = True):
        super().__init__()
        self.channels = channels
        self.d_sub = d_sub
        self.d_state = d_state

        self.sub_linear   = nn.Linear(d_sub,   2 * channels)
        self.state_linear = nn.Linear(d_state, 2 * channels)

        self.g_sub   = nn.Parameter(torch.tensor(1.0))
        self.g_state = nn.Parameter(torch.tensor(1.0))

        if init_id:
            with torch.no_grad():
                self.sub_linear.weight.zero_();   self.sub_linear.bias.zero_()
                self.state_linear.weight.zero_(); self.state_linear.bias.zero_()

    def forward(self, x, subj_emb, state_emb):
        if x.shape[-1] != self.channels:
            raise ValueError(
                f"SubjectStateLayers_FiLM expected x.shape[-1] == {self.channels}, got {x.shape[-1]}"
            )
        if subj_emb.shape[-1] != self.d_sub:
            raise ValueError(
                f"SubjectStateLayers_FiLM expected subj_emb.shape[-1] == {self.d_sub}, got {subj_emb.shape[-1]}"
            )
        if state_emb.shape[-1] != self.d_state:
            raise ValueError(
                f"SubjectStateLayers_FiLM expected state_emb.shape[-1] == {self.d_state}, got {state_emb.shape[-1]}"
            )

        xdtype = x.dtype
        device = x.device

        w_sub_dtype = self.sub_linear.weight.dtype
        w_state_dtype = self.state_linear.weight.dtype

        subj_emb  = subj_emb.to(device=device, dtype=w_sub_dtype)
        state_emb = state_emb.to(device=device, dtype=w_state_dtype)

        sub_gb   = self.sub_linear(subj_emb)        # (B,2C)
        st_gb    = self.state_linear(state_emb)     # (B,2C)

        g_sub, b_sub = sub_gb.chunk(2, dim=-1)
        g_st,  b_st  = st_gb.chunk(2, dim=-1)

        gsub = self.g_sub.to(device=device, dtype=w_sub_dtype)
        gst  = self.g_state.to(device=device, dtype=w_state_dtype)
        
        gamma = 1.0 + 0.1 * (gsub * g_sub + gst * g_st)
        beta  = 0.1 * (gsub * b_sub + gst * b_st)

        gamma = gamma.to(xdtype).unsqueeze(1)
        beta  = beta.to(xdtype).unsqueeze(1)
        return gamma * x + beta
    

class FiLMBlock(nn.Module):
    def __init__(self, n_channels, d_sub):
        super().__init__()
        
        self.film = nn.Linear(d_sub, 2 * n_channels)
        self.n_channels = n_channels
        self.d_sub = d_sub

    def forward(self, x, subj_emb):

        self.film.to(x.dtype)
        film_params = self.film(subj_emb.to(x.dtype))          # (B, 2*C)
        gamma, beta = film_params.chunk(2, dim=-1) # each (B, C)

        # bounded residual FiLM
        gamma = 1.0 + 0.1 * gamma
        beta  = 0.1 * beta

        gamma = gamma.unsqueeze(1)  # (B, 1, C)
        beta  = beta.unsqueeze(1)   # (B, 1, C)

        return gamma * x + beta  # x: (B, T, C)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "n_channels": self.n_channels,
            "d_sub": self.d_sub
        })
        return config


class SubjectLayers_v2(nn.Module):
    """Per subject linear layer."""
    def __init__(self, n_subjects: int, emb_dim: int):
        super().__init__()
        self.sub_emb = nn.Embedding(n_subjects, emb_dim)

    def forward(self, x, subjects):
        weights = self.sub_emb(subjects)
        x_ = torch.einsum("btc,bcd->btc", x, weights)
        return x_


class NoiseInjection(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weight = None

    def build(self, input_shape):
        # input: [B, T, C]
        C = input_shape[-1]
        self.weight = self.add_weight(
            name="noise_weight",
            shape=(1, 1, C),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x, noise=None):
        shape = ops.shape(x)
        B, T, _ = shape[0], shape[1], shape[2]

        if noise is None:
            noise = keras.random.normal((B, T, 1), dtype=x.dtype)

        # broadcast noise over channels and scale
        return x + self.weight * noise


class MinibatchStdDev(layers.Layer):
    def __init__(self, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, x):
        # x: [B, T, C]
        shape = ops.shape(x)
        B, T, C = shape[0], shape[1], shape[2]

        # If batch = 1 → no variance possible
        def no_batch_var():
            zeros = ops.zeros((B, T, 1), dtype=x.dtype)
            return ops.concatenate([x, zeros], axis=-1)

        def compute_mbstd():
            # mean over batch
            x_ng = ops.stop_gradient(x)

            mean = ops.mean(x_ng, axis=0, keepdims=True)       # [1, T, C]
            var  = ops.mean((x_ng - mean)**2, axis=0, keepdims=True)
            std  = ops.sqrt(var + self.eps)                 # [1, T, C]

            # scalar std
            std_mean = ops.mean(std)                        # scalar

            # expand to [B, T, 1]
            std_map = ops.expand_dims(std_mean, axis=0)     # [1]
            std_map = ops.reshape(std_map, (1, 1, 1))        # [1,1,1]
            std_map = ops.broadcast_to(std_map, (B, T, 1))   # [B,T,1]

            # concat with original features
            return ops.concatenate([x, std_map], axis=-1)

        return ops.cond(B <= 1, no_batch_var, compute_mbstd)


def convBlock(filters: List[int],
              kernel_sizes: List[Union[int, tuple]],
              upsampling: List[Union[bool, int]],
              noiseinjection: List[Union[bool, int]],
              stride: int,
              padding: str,
              interpolation: str,
              negative_slope: float = 0.2,
              kernel_initializer: str = 'glorot_uniform',
              batch_norm: bool = True) -> List[layers.Layer]:
    lyrs = []
    for i, (filter, kernel_size) in enumerate(zip(filters, kernel_sizes), 1):
        if upsampling[i - 1]:
            lyrs.append(TorchLinearUpsample1D(2))
        lyrs.append(layers.Conv1D(filter, kernel_size, stride, padding, kernel_initializer=kernel_initializer, name=f'conv_{i}'))
        if noiseinjection[i - 1]:
            lyrs.append(NoiseInjection(name=f"g_noise_{i}"))
        if batch_norm:
            lyrs.append(layers.BatchNormalization(name=f'bn_{i}'))
        lyrs.append(layers.LeakyReLU(negative_slope=negative_slope, name=f'leaky_relu_{i}'))
    return lyrs


class ConvBlockResidual(layers.Layer):
    def __init__(self, filters, kernel_sizes, upsampling, stride=1, padding='same',
                 interpolation='bilinear', negative_slope=0.2, kernel_initializer='glorot_uniform',
                 batch_norm=True):
        super().__init__()
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.kernel_initializer = kernel_initializer
        self.upsampling = upsampling
        self.conv_layers = []
        self.batch_norm = batch_norm
        self.interpolation = interpolation
        self.negative_slope = negative_slope

        # Create convolutional layers
        for i, (filter, kernel_size, up) in enumerate(zip(filters, kernel_sizes, upsampling)):
            if up:  # Upsampling before Conv
                self.conv_layers.append(CustomUpSampling1D(size=2, method=interpolation))  # Fixing time mismatch

            self.conv_layers.append(layers.Conv1D(filter, kernel_size, strides=stride, padding=padding,
                                                  kernel_initializer=kernel_initializer, name=f'conv_{i}'))
            if batch_norm:
                self.conv_layers.append(layers.BatchNormalization(name=f'bn_{i}'))
            self.conv_layers.append(layers.LeakyReLU(negative_slope=negative_slope, name=f'leaky_relu_{i}'))

        # 1x1 Conv to Match Feature Dimension in Residual Connection
        self.match_features = layers.Conv1D(filters[-1], kernel_size=1, padding='same', name='skip_conv')

    def build(self, input_shape):
        self.conv_layers = []  # Reset in case of re-build
        self.match_features = None  # Reset matching layer

        for i, (filter, kernel_size, up) in enumerate(zip(self.filters, self.kernel_sizes, self.upsampling)):
            if up:  # Upsampling before Conv
                self.conv_layers.append(CustomUpSampling1D(size=2, method=self.interpolation))

            self.conv_layers.append(layers.Conv1D(filter, kernel_size, strides=1, padding='same',
                                                  kernel_initializer='glorot_uniform', name=f'conv_{i}'))
            if self.batch_norm:
                self.conv_layers.append(layers.BatchNormalization(name=f'bn_{i}'))
            self.conv_layers.append(layers.LeakyReLU(negative_slope=self.negative_slope, name=f'leaky_relu_{i}'))

        # # Define the 1x1 Conv1D to match feature dimensions if needed
        # self.match_features = layers.Conv1D(self.filters[-1], kernel_size=1, padding='same', name='skip_conv')

        # Mark layer as built
        super().build(input_shape)

    def call(self, x):
        # skip = x  # Save input for residual connection

        for layer in self.conv_layers:
            x = layer(x)

        # # Match time dimension (if upsampling happened)
        # if x.shape[1] != skip.shape[1]:
        #     skip = CustomUpSampling1D(size=x.shape[1] // skip.shape[1], method=self.interpolation)(skip)

        # # Match feature dimension
        # if x.shape[-1] != skip.shape[-1]:
        #     skip = self.match_features(skip)

        # # Residual Addition
        # x = layers.Add(name='residual_addition')([x, skip])
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_sizes": self.kernel_sizes,
            "upsampling": self.upsampling,
            "interpolation": self.interpolation,
            "negative_slope": self.negative_slope,
            "kernel_initializer": self.kernel_initializer,
            "batch_norm": self.batch_norm
        })
        return config


# class ConvBlockResidual(layers.Layer):
#     """Residual Convolutional Block with Upsampling."""
#     def __init__(self,
#                  filters: List[int],
#                  kernel_sizes: List[Union[int, tuple]],
#                  upsampling: List[Union[bool, int]],
#                  interpolation: str = 'linear',
#                  negative_slope: float = 0.2,
#                  kernel_initializer: str = 'glorot_uniform',
#                  batch_norm: bool = True,
#                  stride=1,
#                  padding='same',
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.filters = filters
#         self.kernel_sizes = kernel_sizes
#         self.kernel_initializer = kernel_initializer
#         self.batch_norm = batch_norm
#         self.negative_slope = negative_slope
#         self.upsampling = upsampling
#         self.interpolation = interpolation

#     def call(self, x):
#         skip = x
#         for i, (filter, kernel_size) in enumerate(zip(self.filters, self.kernel_sizes), 1):
#             if self.upsampling[i - 1]:  # Check if upsampling is needed
#                 x = CustomUpSampling1D(2, method=self.interpolation)(x)

#             x = layers.Conv1D(filter, kernel_size, strides=1, padding='same',
#                               kernel_initializer=self.kernel_initializer, name=f'conv_{i}')(x)

#             if self.batch_norm:
#                 x = layers.BatchNormalization(name=f'bn_{i}')(x)

#             x = layers.LeakyReLU(negative_slope=self.negative_slope, name=f'leaky_relu_{i}')(x)

#         # Match dimensions of skip and x before addition
#         if skip.shape[-1] != x.shape[-1]:  # If the number of channels differs
#             skip = layers.Conv1D(self.filters[-1], kernel_size=1, padding='same', name='skip_conv')(skip)

#         x = layers.Add(name='residual_addition')([x, skip])  # Residual connection
#         return x

#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "filters": self.filters,
#             "kernel_sizes": self.kernel_sizes,
#             "upsampling": self.upsampling,
#             "interpolation": self.interpolation,
#             "negative_slope": self.negative_slope,
#             "kernel_initializer": self.kernel_initializer,
#             "batch_norm": self.batch_norm
#         })
#         return config


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


class WeightNormConv1D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate=1, padding='same', activation=None, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.activation = activation

    def build(self, input_shape):
        # PyTorch Conv1d expects (batch, channels, length)
        self.conv = torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(
            in_channels=input_shape[-1],
            out_channels=self.filters,
            kernel_size=self.kernel_size,
            dilation=self.dilation_rate,
            padding='same',
        ))

        if self.activation:
            self.act = getattr(torch.nn, self.activation)() if isinstance(self.activation, str) else self.activation
        else:
            self.act = None

    def call(self, inputs):
        # Keras uses (batch, time, channels); PyTorch Conv1d uses (batch, channels, time)
        x = torch.permute(inputs, (0, 2, 1))
        x = self.conv(x)
        if self.act:
            x = self.act(x)
        x = torch.permute(x, (0, 2, 1))
        return x


class TCNResidualBlock(keras.Model):
    def __init__(self, filters, kernel_size=7, dilation_rates=[1, 2, 4], dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.layers_ = []
        for d in dilation_rates:
            # self.layers_.append(WeightNormConv1D(filters=filters, kernel_size=kernel_size, dilation_rate=d, activation=None, name=f'conv_{d}'))
            self.layers_.append(layers.Conv1D(filters, kernel_size, dilation_rate=d, padding='same', name=f'conv_{d}'))
            self.layers_.append(layers.LayerNormalization())
            self.layers_.append(layers.LeakyReLU(0.2, name=f'act_{d}'))
            self.layers_.append(layers.Dropout(dropout_rate, name=f'drp_{d}'))
        self.final_activation = layers.LeakyReLU(0.2, name='final_act')
        self.projection = layers.Conv1D(self.layers_[0].filters, kernel_size=1, padding='same', name='conv_proj')
        self.built = True

    def call(self, x):
        if x.shape[-1] != self.layers_[0].filters:
            residual = self.projection(x)
        else:
            residual = x
        out = x
        for layer in self.layers_:
            out = layer(out)
        out = layers.add([out, residual])
        return self.final_activation(out)
