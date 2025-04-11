import keras
from keras import layers, Model
import torch
import torch.nn.functional as F


class AttentionPooling1D(keras.layers.Layer):
    def __init__(self, pool_size=4, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], self.pool_size),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(self.pool_size,),
                                 initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch, time, features)
        score = torch.tanh(torch.matmul(inputs, self.W) + self.b)  # shape: (batch, time, 1)
        weights = F.softmax(score, dim=1)  # shape: (batch, time, 1)
        weighted_sum = torch.einsum('btd,btk->bkd', inputs, weights)  # (batch, pool_size, features)

        return weighted_sum.reshape(inputs.shape[0], -1)  # (batch, pool_size * features)


def build_attention_feature_extractor(base_model, conv_layers):
    layer_outputs = [base_model.get_layer(name).output for name in conv_layers]
    pooled_outputs = [AttentionPooling1D()(layer_out) for layer_out in layer_outputs]
    combined = layers.Concatenate(name="concat_features")(pooled_outputs)
    feature_extractor = Model(inputs=base_model.layers[0].input, outputs=combined, name="feature_extractor")
    return feature_extractor
