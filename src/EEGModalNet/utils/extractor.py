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
                'subj_emb': subj_emb[i:end_idx].to(device),
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


def fit_new_subject_embeddings(
    critic,
    X_input,                 # (N, T, C)
    new_subject_ids,         # (N,) original IDs from new dataset, any labels
    state_ids,               # (N, ...) pos/state input for critic
    e_mean,                  # (32,)
    batch_size=256,
    n_steps=300,
    max_segments_per_subject=128,
    lr=1e-2,
    reg_lambda=1e-3,
    device="mps",
    clamp_norm=False,
):
    critic.eval()

    # freeze critic weights, but keep graph so gradients flow into E_new
    for p in critic.parameters():
        p.requires_grad = False

    X_input = X_input.float().to(device)
    state_ids = state_ids.to(device)
    new_subject_ids = new_subject_ids.view(-1).cpu()

    # map arbitrary subject labels -> local 0..n_new-1
    unique_subjects = torch.unique(new_subject_ids)
    n_new = len(unique_subjects)

    id_to_local = {int(s.item()): i for i, s in enumerate(unique_subjects)}
    local_ids = torch.tensor(
        [id_to_local[int(s.item())] for s in new_subject_ids],
        dtype=torch.long,
        device=device,
    )

    # calibration subset: max K segments per subject
    calib_indices = []
    for local_s in range(n_new):
        idx = torch.where(local_ids.cpu() == local_s)[0]
        if len(idx) > max_segments_per_subject:
            perm = torch.randperm(len(idx))[:max_segments_per_subject]
            idx = idx[perm]
        calib_indices.append(idx)

    calib_indices = torch.cat(calib_indices).to(device)

    # learn one embedding per new subject, initialized from mean training embedding
    E_new = torch.nn.Parameter(
        e_mean.detach().clone().to(device).unsqueeze(0).repeat(n_new, 1)
    )  # (n_new, 32)

    opt = torch.optim.Adam([E_new], lr=lr)

    n_calib = len(calib_indices)

    for step in range(n_steps):
        batch_idx = calib_indices[
            torch.randint(0, n_calib, (batch_size,), device=device)
        ]

        x_batch = X_input[batch_idx]
        pos_batch = state_ids[batch_idx]
        sub_idx_batch = local_ids[batch_idx]          # 0..n_new-1
        subj_emb_batch = E_new[sub_idx_batch]         # (B, 32)

        inputs = {
            "x": x_batch,
            "subj_emb": subj_emb_batch,
            "pos": pos_batch,
        }

        out = critic(inputs)

        # fit embedding so this subject looks in-distribution to frozen critic
        loss_fit = -out.mean()

        # keep embeddings near training subject manifold
        loss_reg = reg_lambda * ((E_new - e_mean.to(device).unsqueeze(0)) ** 2).mean()

        loss = loss_fit + loss_reg

        opt.zero_grad()
        loss.backward()
        opt.step()

        if clamp_norm:
            train_dist_95 = 7.95  # pre-computed 95th percentile of train embedding distances to mean
            with torch.no_grad():
                center = e_mean.to(device)[None, :]
                delta = E_new - center
                dist = delta.norm(dim=1, keepdim=True)

                max_dist = train_dist_95
                scale = torch.clamp(max_dist / dist.clamp_min(1e-8), max=1.0)

                E_new.copy_(center + delta * scale)

        if step % 50 == 0:
            print(
                f"step {step:04d} | loss={loss.item():.4f} | "
                f"fit={loss_fit.item():.4f} | reg={loss_reg.item():.6f}"
            )
            dist = torch.norm(E_new - e_mean[None, :].to(device), dim=1).mean()
            print("mean embedding distance:", dist.item())

    return E_new.detach().cpu(), unique_subjects
