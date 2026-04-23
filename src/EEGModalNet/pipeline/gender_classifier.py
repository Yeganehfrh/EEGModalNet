
import os
os.environ['KERAS_BACKEND'] = 'torch'

from typing import List
import torch
import keras
import xarray as xr
from datetime import datetime
import pickle
import json 
from ...EEGModalNet import FiLMGAN, preprocess_data, extract_features_batched_deterministic
from scipy.signal import butter, sosfiltfilt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from keras import regularizers, layers
from meegkit import dss
import argparse



def load_OTKA_data(eeg_path: str,
                   demo_path: str,
                   channels: List[str],
                   downsample_data: bool = True,
                   time_dim: int = 512,
                   return_sub_ids: bool = False) -> tuple:
    
    EEG = xr.open_dataarray(eeg_path, engine='h5netcdf')
    behavioral = pd.read_csv(demo_path)
    classes = behavioral[['gender', 'bids_id']].dropna().set_index('bids_id')
    classes['gender'] = classes['gender'].apply(lambda x: 0 if x == 'Male' else 1)

    def format_subject_id(subject_id):
        return f"sub-{int(subject_id):02d}"

    sub_ids = classes.index
    if downsample_data:
        n_y0 = (classes == 0).sum().values
        n_y1 = (classes == 1).sum().values
        n_min = min(n_y0, n_y1)
        n_subjects = n_min * 2
        y0_sub_ids = classes.query("gender == 0").index[:n_min[0]]
        y1_sub_ids = classes.query("gender == 1").index[:n_min[0]]
        sub_ids = y1_sub_ids.append(y0_sub_ids)
    
    if return_sub_ids:
        return sub_ids

    sub_ids_formatted = [format_subject_id(sub_id) for sub_id in sub_ids]

    # X_input
    x = EEG.sel(subject=sub_ids_formatted, channel=channels).to_numpy()
    x = x.reshape(-1, *x.shape[2:])
    # remove those two missing recordings from the last participants if it's among the sub ids
    if 52 in sub_ids:
        x = np.concatenate([x[:-3], x[-1:]])

    # Process
    x = preprocess_data(x, sampling_rate=128)

    # Highpass filter
    sos = butter(4, 0.5, btype='high', fs=128, output='sos')
    x = sosfiltfilt(sos, x, axis=-1)

    # Remove the line noise
    x, _ = dss.dss_line(x.T, fline=50, sfreq=128, nremove=1)
    x = x.T

    X_input = torch.tensor(x.copy()).unfold(2, time_dim, time_dim).permute(0, 2, 3, 1).flatten(0, 1)

    # Classes
    n_subjects = len(sub_ids)
    y = classes.loc[sub_ids].values
    y = y.repeat(416)  # X_input.shape[0] // n_subjects = 416

    # Groups
    sub = torch.tensor(np.arange(0, n_subjects).repeat(416)[:, np.newaxis])
    groups = sub.squeeze().numpy()

    # Remove NaNs
    y = y[:-208]
    groups = groups[:-208]

    return X_input, y, groups


def extract_features(checkpoint_path, device = "cpu"):

    model = keras.saving.load_model(
        checkpoint_path,
        custom_objects={"FiLMGAN": FiLMGAN},
        compile=False,
    )
    critic = model.critic
    critic.eval()
    critic.to(device)

    for p in critic.parameters():
        p.requires_grad = False

    # Force-build the Keras conv layers on the SAME device
    dummy = {
        "x": torch.zeros((1, critic.time_dim, critic.feature_dim), dtype=torch.float32, device=device),
        "sub": torch.zeros((1, critic.d_sub), dtype=torch.float32, device=device),
        "pos": torch.zeros((1, 1), dtype=torch.long, device=device),
    }
    with torch.no_grad():
        _ = critic.encode_features(dummy)

    # Reload checkpoint weights after build
    model.load_weights(checkpoint_path)

    # Keep all feature inputs on the SAME device
    e_mean = critic.sub_emb.module.weight.mean(0).detach().to(device)
    B = X_input.shape[0]
    subj_emb = e_mean[None, :].repeat(B, 1)
    state_ids = torch.zeros((B, 1), dtype=torch.long, device=device)

    feats, _, _ = extract_features_batched_deterministic(
        critic,
        X_input.to(device),
        subj_emb,
        state_ids,
        batch_size=256,
        device=device,
    )

    return feats

def load_CBraMod_features(feature_path, idx_int):
        path_sub_52 = feature_path.replace('gender', 'sub_52')
        cbramod_dict = torch.load(feature_path, weights_only=False)
        X_e_sub_52 = torch.load(path_sub_52, weights_only=False)
        X_e = np.asarray(cbramod_dict['features'])
        seg_per_sub = X_e.shape[0] // 51
        X_e = X_e.reshape(51, seg_per_sub, -1)
        X_e_sub_52 = np.concatenate([X_e_sub_52[None], np.full((1, *X_e_sub_52.shape), np.nan)], axis=1) # reshape to (1, 416, 3200) with NaNs for missing part
        X_e = np.concatenate([X_e, X_e_sub_52])

        X_e = X_e[idx_int]
        y = np.asarray(cbramod_dict['gender'])
        y = np.concatenate([np.asarray(cbramod_dict['gender']), np.array([1])]) # last subject gender
        y = np.where(y == 0, 1, 0)
        y = y[idx_int]
        y = np.repeat(y, seg_per_sub)  # X_e.shape[0]//52 = 832
        groups = np.asarray(cbramod_dict['subject_ids'])
        groups = np.concatenate([groups, np.array([52])])  # last subject id
        groups = groups[idx_int]
        groups = np.repeat(groups, seg_per_sub)

        # remove NaNs
        to_be_rm = seg_per_sub//2
        X_e = X_e.reshape(-1, X_e.shape[-1])[:-to_be_rm]
        y = y[:-to_be_rm]
        groups = groups[:-to_be_rm]

        return X_e, y, groups

def run_classification(feats, y, groups, epochs=100, batch_size=128, scale=True):
    sgkf = StratifiedGroupKFold(n_splits=5, random_state=None).split(feats, y, groups=groups)
    fold_summaries = []  # Store fold-level summary stats
    all_fold_histories = {}  # Store complete epoch-by-epoch history per fold

    for folds, (train_idx, val_idx) in enumerate(sgkf):
        # Class balance information
        train_class_mean = y[train_idx].mean()
        val_class_mean = y[val_idx].mean()
        train_size = len(train_idx)
        val_size = len(val_idx)
        
        print(f"Fold {folds}: Train class balance={train_class_mean:.4f}, Val class balance={val_class_mean:.4f}")

        if scale:
            scaler = StandardScaler().fit(feats)
            feats_train = scaler.transform(feats[train_idx])
            feats_val = scaler.transform(feats[val_idx])
        else:
            feats_train = feats[train_idx]
            feats_val = feats[val_idx]

        cls_model = keras.models.Sequential([
            layers.Dense(1, activation='sigmoid')
        ])

        cls_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[
                keras.metrics.AUC(name="auc"),
                keras.metrics.BinaryAccuracy(threshold=0.5, name="accuracy")
            ]
        )

        callback = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=50,
            restore_best_weights=True
        )

        cw = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y[train_idx]),
            y=y[train_idx]
        )
        
        class_weights = dict(zip(np.unique(y[train_idx]), cw))

        history = cls_model.fit(
            feats_train, y[train_idx],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(feats_val, y[val_idx]),
            class_weight=class_weights,
            callbacks=callback,
            shuffle=True,
            verbose=0
        )

        # Store complete epoch-by-epoch history
        all_fold_histories[folds] = history.history
        
        # Store fold summary (best metrics and class balance)
        fold_summary = {
            'fold': folds,
            'train_class_balance': train_class_mean,
            'val_class_balance': val_class_mean,
            'train_size': train_size,
            'val_size': val_size,
            'best_epoch': np.argmax(history.history['val_accuracy']),
            'best_val_accuracy': np.max(history.history['val_accuracy']),
            'best_val_auc': np.max(history.history['val_auc']),
            'best_val_loss': np.min(history.history['val_loss']),
            'final_train_accuracy': history.history['accuracy'][-1],
            'final_train_auc': history.history['auc'][-1],
            'final_train_loss': history.history['loss'][-1],
            'num_epochs_trained': len(history.history['loss'])
        }
        fold_summaries.append(fold_summary)

    # Create summary DataFrame
    fold_summary_df = pd.DataFrame(fold_summaries)
    print("\n" + "="*80)
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print("="*80)
    print(fold_summary_df.to_string(index=False))
    print("\nOverall Results:")
    print(f"  Mean Val Accuracy: {fold_summary_df['best_val_accuracy'].mean():.4f} ± {fold_summary_df['best_val_accuracy'].std():.4f}")
    print(f"  Mean Val AUC:      {fold_summary_df['best_val_auc'].mean():.4f} ± {fold_summary_df['best_val_auc'].std():.4f}")
    print(f"  Mean Val Loss:     {fold_summary_df['best_val_loss'].mean():.4f} ± {fold_summary_df['best_val_loss'].std():.4f}")
    return all_fold_histories, fold_summaries


def create_and_save_detailed_metrics(fold_summaries):
    # 3. Create detailed metrics table (one row per epoch per fold)
    detailed_metrics = []
    for fold_id, history in all_fold_histories.items():
        fold_data = fold_summary_df[fold_summary_df['fold'] == fold_id].iloc[0]
        for epoch in range(len(history['loss'])):
            detailed_metrics.append({
                'fold': fold_id,
                'train_class_balance': fold_data['train_class_balance'],
                'val_class_balance': fold_data['val_class_balance'],
                'epoch': epoch + 1,
                'train_loss': history['loss'][epoch],
                'train_accuracy': history['accuracy'][epoch],
                'train_auc': history['auc'][epoch],
                'val_loss': history['val_loss'][epoch],
                'val_accuracy': history['val_accuracy'][epoch],
                'val_auc': history['val_auc'][epoch],
            })

    detailed_df = pd.DataFrame(detailed_metrics)
    detailed_df.to_csv(f'{output_dir}/detailed_metrics_per_epoch.csv', index=False)
    print(f"✓ Saved detailed metrics (CSV): {output_dir}/detailed_metrics_per_epoch.csv")

    # 4. Save metadata about the results
    metadata = {
        # 'task': TASK,
        'n_folds': len(fold_summaries),
        'timestamp': datetime.now().isoformat(),
        'mean_val_accuracy': float(fold_summary_df['best_val_accuracy'].mean()),
        'std_val_accuracy': float(fold_summary_df['best_val_accuracy'].std()),
        'mean_val_auc': float(fold_summary_df['best_val_auc'].mean()),
        'std_val_auc': float(fold_summary_df['best_val_auc'].std()),
    }
    with open(f'{output_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {output_dir}/metadata.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, default='yaregan', choices=['yaregan', 'cbra', 'raw'], help='Features to be used in the classifier')
    # parser.add_argument('--model-path', type=str, default='logs/gender_cls_OTKA', help='Path for saving model and logs')
    parser.add_argument('--n-epochs', type=int, default=100, help='Number of epochs')
    args = parser.parse_args()

    CHANNELS = ['O1', 'O2', 'P1', 'P2', 'C1', 'C2', 'F1', 'F2']
    FEATURES = args.features
    # MODEL_PATH = args.model_path
    EPOCHS = args.n_epochs

    if FEATURES == 'yaregan':
        print(f'>>>> Use Features Extracted from Yare-GAN')
        X_input, y, groups = load_OTKA_data('data/OTKA/experiment_EEG_data.nc5',
                                    'data/OTKA/PLB_HYP_data_MASTER.csv',
                                     channels=CHANNELS,
                                     time_dim=512)
        X_e = extract_features('logs/eo/20260330_epoch_100.model.keras')

    elif FEATURES == 'cbra':
        print(f'>>>> Use Features Extracted from CBraMod')
        sub_ids = load_OTKA_data('data/OTKA/experiment_EEG_data.nc5', 'data/OTKA/PLB_HYP_data_MASTER.csv', channels=CHANNELS, return_sub_ids=True)
        X_e, y, groups = load_CBraMod_features('data/benchmarking/CBraMod_features_gender_seg-4s_balanced.pt', sub_ids)

    elif FEATURES == 'raw':
        print(f'>>>> Use Raw Signal')
        X_input, y, groups = load_OTKA_data('data/OTKA/experiment_EEG_data.nc5',
                            'data/OTKA/PLB_HYP_data_MASTER.csv',
                             channels=CHANNELS,
                             time_dim=512)
        X_e = X_input.flatten(1, 2)

    else:
        raise ValueError(f'Unknown feature type {FEATURES}')

    ##### Classifier
    all_fold_histories, fold_summaries = run_classification(X_e, y, groups, epochs=EPOCHS, batch_size=128)

    ##### Save Results
    print("="*80)
    print("SAVING K-FOLD CROSS-VALIDATION RESULTS")
    print("="*80)

    output_dir = f'logs/{FEATURES}_classifier_kfold_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(output_dir, exist_ok=True)

    fold_summary_df = pd.DataFrame(fold_summaries)
    fold_summary_df.to_csv(f'{output_dir}/fold_summary.csv', index=False)
    print(f"✓ Saved fold summary (CSV): {output_dir}/fold_summary.csv")

    with open(f'{output_dir}/all_fold_histories.pkl', 'wb') as f:
        pickle.dump(all_fold_histories, f)
    print(f"✓ Saved fold histories (Pickle): {output_dir}/all_fold_histories.pkl")

    create_and_save_detailed_metrics(fold_summaries)
