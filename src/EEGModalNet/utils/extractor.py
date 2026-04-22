import numpy as np
import keras
from keras import layers, Model
import torch
import torch.nn.functional as F
import hashlib


def extract_features_batched(critic, X_input, subj_emb, state_ids, batch_size=256, device='cpu'):
    """
    Extract features in batches to avoid memory overflow.
    
    Args:
        critic: The critic model
        X_input: Input tensor (N, T, C)
        subj_emb: Subject embeddings (N, D)
        state_ids: State IDs (N,)
        batch_size: Number of samples per batch
        device: Device to process on
    
    Returns:
        Concatenated features as numpy array
    """
    all_h1 = []
    all_h = []
    n_samples = X_input.shape[0]
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_inputs = {
            'x': X_input[i:end_idx].float().to(device),
            'sub': subj_emb[i:end_idx].to(device),
            'pos': state_ids[i:end_idx].to(device)
        }
        h1_batch, h_batch = critic.encode_features(batch_inputs)
        all_h1.append(h1_batch.cpu().detach())
        all_h.append(h_batch.cpu().detach())
    
    h1_full = torch.cat(all_h1, dim=0)
    h_full = torch.cat(all_h, dim=0)
    feats = torch.cat([h1_full.mean(1), h_full.mean(1)], dim=-1).numpy()
    
    return feats, h1_full, h_full


def extract_features_adaptive_pool(critic, X_input, subj_emb, state_ids, pool_size=4, batch_size=256, device='mps'):
    """
    Adaptive pooling: Convert temporal dimension to fixed size.
    This preserves spatial information better than mean pooling.
    """
    all_h1 = []
    all_h = []
    n_samples = X_input.shape[0]
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_inputs = {
            'x': X_input[i:end_idx].float().to(device),
            'sub': subj_emb[i:end_idx].to(device),
            'pos': state_ids[i:end_idx].to(device)
        }
        
        h1_batch, h_batch = critic.encode_features(batch_inputs)
        all_h1.append(h1_batch.cpu().detach())
        all_h.append(h_batch.cpu().detach())
    
    h1_full = torch.cat(all_h1, dim=0)  # (N, T1, C1)
    h_full = torch.cat(all_h, dim=0)    # (N, T2, C2)
    
    # Adaptive pooling to (N, pool_size, C)
    h1_pooled = F.adaptive_avg_pool1d(h1_full.transpose(1, 2), pool_size)  # (N, C1, pool_size)
    h_pooled = F.adaptive_avg_pool1d(h_full.transpose(1, 2), pool_size)    # (N, C2, pool_size)
    
    # Flatten: (N, C1*pool_size + C2*pool_size)
    feats = torch.cat([h1_pooled.flatten(1), h_pooled.flatten(1)], dim=-1).numpy()
    
    return feats


def extract_features_batched_deterministic(
    critic, X_input, subj_emb, state_ids, 
    batch_size=256, device='cpu', seed=42
):
    """
    Extract features in batches with full determinism control.
    """
    # Set all random seeds BEFORE computation
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    # Ensure deterministic algorithms
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Ensure critic is truly in eval mode and frozen
    critic.eval()
    for param in critic.parameters():
        param.requires_grad = False
    
    # Disable any dropout/batch norm randomness
    for module in critic.modules():
        if hasattr(module, 'training'):
            module.train(False)
    
    all_h1 = []
    all_h = []
    n_samples = X_input.shape[0]
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_inputs = {
                'x': X_input[i:end_idx].float().to(device),
                'sub': subj_emb[i:end_idx].to(device),
                'pos': state_ids[i:end_idx].to(device)
            }
            h1_batch, h_batch = critic.encode_features(batch_inputs)
            all_h1.append(h1_batch.cpu().detach())
            all_h.append(h_batch.cpu().detach())
    
    h1_full = torch.cat(all_h1, dim=0)
    h_full = torch.cat(all_h, dim=0)
    feats = torch.cat([h1_full.mean(1), h_full.mean(1)], dim=-1).numpy()
    
    # Compute checksum for verification
    checksum = float(np.sum(feats))
    checksum_hash = hashlib.md5(feats.tobytes()).hexdigest()
    
    print(f"Checksum (sum): {checksum}")
    print(f"Checksum (hash): {checksum_hash}")
    
    return feats, h1_full, h_full


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
