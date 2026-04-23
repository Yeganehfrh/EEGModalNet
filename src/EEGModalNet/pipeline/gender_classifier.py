
import os
os.environ['KERAS_BACKEND'] = 'torch'

from typing import List
import torch
import keras
import xarray as xr
from datetime import datetime
import pickle
import json 
from ...EEGModalNet import FiLMGAN, preprocess_data, extract_features_batched_deterministic, BalancedAccuracy
from scipy.signal import butter, sosfiltfilt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from keras import layers
from meegkit import dss
import argparse


def _preprocess(x):
    # Process
    x = preprocess_data(x, sampling_rate=128)

    # Highpass filter
    sos = butter(4, 0.5, btype='high', fs=128, output='sos')
    x = sosfiltfilt(sos, x, axis=-1)

    # Remove the line noise
    x, _ = dss.dss_line(x.T, fline=50, sfreq=128, nremove=1)
    return x.T

def _balance_classes(classes):
    n_y0 = (classes == 0).sum()
    n_y1 = (classes == 1).sum()
    n_min = min(n_y0, n_y1)
    y0_sub_ids = classes[classes == 0].index[:n_min]
    y1_sub_ids = classes[classes == 1].index[:n_min]
    sub_ids = y0_sub_ids.append(y1_sub_ids)
    return sub_ids

def _format_subject_id(subject_id):
    return f"sub-{int(subject_id):02d}"

def _gender_subject_id(subject_id):
    return f"{int(subject_id):02d}"

def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _validate_samples(feats, y, groups):
    n_samples = len(feats)
    if len(y) != n_samples or len(groups) != n_samples:
        raise ValueError(
            f"Feature/label/group length mismatch: "
            f"features={n_samples}, y={len(y)}, groups={len(groups)}"
        )
    if np.isnan(_as_numpy(feats)).any():
        raise ValueError("Features contain NaNs after loading/preprocessing.")
    if np.isnan(np.asarray(y, dtype=float)).any():
        raise ValueError("Labels contain NaNs after loading/preprocessing.")

def load_data(task: str, 
              channels: List[str],
              balance_classes: bool = True,
              time_dim: int = 512,
              return_sub_ids: bool = False):
    
    if task == 'age':
        eeg_path = 'data/ds005385/ds005385.nc5'
        demog_path = 'data/ds005385/demographic.csv'
    elif task == 'gender':
        eeg_path = 'data/OTKA/experiment_EEG_data_missing_as_nan.nc5'
        demog_path = 'data/OTKA/PLB_HYP_data_MASTER.csv'
    else:
        raise ValueError(f'Unknown task {task}')

    EEG = xr.open_dataarray(eeg_path, engine='h5netcdf')
    demog = pd.read_csv(demog_path)

    if task == 'age':
        demog = demog.set_index('participant_id')
        classes = demog['age_group'].apply(lambda x: 0 if x == 'young' else 1)

    elif task == 'gender':
        classes = demog[['gender', 'bids_id']].dropna().set_index('bids_id')
        classes = classes['gender'].apply(lambda x: 0 if x == 'Male' else 1)

    sub_ids = classes.index
    if balance_classes:
        sub_ids = _balance_classes(classes)
    
    if return_sub_ids:
        return sub_ids
    
    # X_input
    if task == 'age':
        x = EEG.sel(subject=sub_ids, channel=channels).to_numpy()
        recording_y = classes.loc[sub_ids].to_numpy()
        recording_groups = np.asarray(sub_ids)
    
    elif task == 'gender':
        sub_ids_formatted = [_format_subject_id(sub_id) for sub_id in sub_ids]
        x = EEG.sel(subject=sub_ids_formatted, channel=channels).to_numpy()
        n_subjects, n_tasks = x.shape[:2]
        x = x.reshape(n_subjects * n_tasks, *x.shape[2:])
        recording_y = np.repeat(classes.loc[sub_ids].to_numpy(), n_tasks)
        recording_groups = np.repeat([_gender_subject_id(sub_id) for sub_id in sub_ids], n_tasks)

    recording_nan_mask = np.isnan(x).reshape(x.shape[0], -1)
    missing_recordings = recording_nan_mask.all(axis=1)
    partial_nan_recordings = recording_nan_mask.any(axis=1) & ~missing_recordings
    if partial_nan_recordings.any():
        raise ValueError("Found partially NaN EEG recordings; refusing to silently drop them.")
    if missing_recordings.any():
        valid_recordings = ~missing_recordings
        x = x[valid_recordings]
        recording_y = recording_y[valid_recordings]
        recording_groups = recording_groups[valid_recordings]

    x = _preprocess(x)

    X_windows = torch.as_tensor(x.copy(), dtype=torch.float32).unfold(2, time_dim, time_dim)
    n_windows_per_recording = X_windows.shape[2]
    X_input = X_windows.permute(0, 2, 3, 1).flatten(0, 1)
    y = np.repeat(recording_y, n_windows_per_recording).astype(int)
    groups = np.repeat(recording_groups, n_windows_per_recording)
    _validate_samples(X_input, y, groups)

    return X_input, y, groups


def extract_features(X_input, checkpoint_path, device="cpu"):

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

def load_CBraMod_features(task, feature_path, sub_ids):
    cbramod_dict = torch.load(feature_path, weights_only=False, map_location="cpu")
    sub_ids = np.asarray(sub_ids)

    if task == 'gender':
        X_e = _as_numpy(cbramod_dict['features'])
        y = _as_numpy(cbramod_dict['gender']).astype(int)
        groups = np.asarray(cbramod_dict['subject_ids']).astype(str)
        selected_groups = np.array([_gender_subject_id(sub_id) for sub_id in sub_ids])

        mask = np.isin(groups, selected_groups)
        X_e = X_e[mask]
        y = y[mask]
        groups = groups[mask]
        missing = sorted(set(selected_groups) - set(groups))
        if missing:
            raise ValueError(f"CBraMod gender features are missing selected subjects: {missing}")

    elif task == 'age':
        X_e = _as_numpy(cbramod_dict['features'])
        subject_ids = np.asarray(cbramod_dict['subject_ids']).astype(str)
        subject_y = _as_numpy(cbramod_dict['age']).astype(int)
        if X_e.shape[0] % subject_ids.shape[0] != 0:
            raise ValueError("CBraMod age features cannot be evenly grouped by subject.")

        segments_per_subject = X_e.shape[0] // subject_ids.shape[0]
        feature_by_subject = X_e.reshape(subject_ids.shape[0], segments_per_subject, -1)
        subject_index = {subject_id: idx for idx, subject_id in enumerate(subject_ids)}
        selected_subjects = sub_ids.astype(str)
        missing = sorted(set(selected_subjects) - set(subject_index))
        if missing:
            raise ValueError(f"CBraMod age features are missing selected subjects: {missing[:10]}")

        selected_idx = np.array([subject_index[subject_id] for subject_id in selected_subjects])
        X_e = feature_by_subject[selected_idx].reshape(-1, feature_by_subject.shape[-1])
        y = np.repeat(subject_y[selected_idx], segments_per_subject)
        groups = np.repeat(subject_ids[selected_idx], segments_per_subject)

    else:
        raise ValueError(f'Unknown task {task}')

    _validate_samples(X_e, y, groups)
    return X_e, y, groups

def run_classification(feats, y, groups, epochs=100, batch_size=128, scale=True):
    feats = _as_numpy(feats)
    y = _as_numpy(y).astype(int).reshape(-1)
    groups = _as_numpy(groups).reshape(-1)
    _validate_samples(feats, y, groups)

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
            scaler = StandardScaler().fit(feats[train_idx])
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
                keras.metrics.BinaryAccuracy(threshold=0.5, name="accuracy"),
                BalancedAccuracy(threshold=0.5, name="balanced_accuracy"),
            ]
        )

        callback = keras.callbacks.EarlyStopping(
            monitor='val_auc',
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

        eval_metrics = cls_model.evaluate(
            feats_val,
            y[val_idx],
            batch_size=batch_size,
            verbose=0,
            return_dict=True,
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
            'best_epoch': int(np.argmax(history.history['val_accuracy']) + 1),
            'best_val_accuracy': np.max(history.history['val_accuracy']),
            'best_val_auc': np.max(history.history['val_auc']),
            'best_val_balanced_accuracy': np.max(history.history['val_balanced_accuracy']),
            'best_val_loss': np.min(history.history['val_loss']),
            'eval_val_accuracy': eval_metrics['accuracy'],
            'eval_val_auc': eval_metrics['auc'],
            'eval_val_balanced_accuracy': eval_metrics['balanced_accuracy'],
            'eval_val_loss': eval_metrics['loss'],
            'final_train_accuracy': history.history['accuracy'][-1],
            'final_train_auc': history.history['auc'][-1],
            'final_train_balanced_accuracy': history.history['balanced_accuracy'][-1],
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
    print(f"  Mean Val Bal Acc:  {fold_summary_df['best_val_balanced_accuracy'].mean():.4f} ± {fold_summary_df['best_val_balanced_accuracy'].std():.4f}")
    print(f"  Mean Val Loss:     {fold_summary_df['best_val_loss'].mean():.4f} ± {fold_summary_df['best_val_loss'].std():.4f}")
    print("\nPost-early-stopping evaluate() Results:")
    print(f"  Mean Eval Val Accuracy: {fold_summary_df['eval_val_accuracy'].mean():.4f} ± {fold_summary_df['eval_val_accuracy'].std():.4f}")
    print(f"  Mean Eval Val AUC:      {fold_summary_df['eval_val_auc'].mean():.4f} ± {fold_summary_df['eval_val_auc'].std():.4f}")
    print(f"  Mean Eval Val Bal Acc:  {fold_summary_df['eval_val_balanced_accuracy'].mean():.4f} ± {fold_summary_df['eval_val_balanced_accuracy'].std():.4f}")
    print(f"  Mean Eval Val Loss:     {fold_summary_df['eval_val_loss'].mean():.4f} ± {fold_summary_df['eval_val_loss'].std():.4f}")
    return all_fold_histories, fold_summaries


def create_and_save_detailed_metrics(all_fold_histories, fold_summaries, fold_summary_df, output_dir, task):
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
                'train_balanced_accuracy': history['balanced_accuracy'][epoch],
                'val_loss': history['val_loss'][epoch],
                'val_accuracy': history['val_accuracy'][epoch],
                'val_auc': history['val_auc'][epoch],
                'val_balanced_accuracy': history['val_balanced_accuracy'][epoch],
            })

    detailed_df = pd.DataFrame(detailed_metrics)
    detailed_df.to_csv(f'{output_dir}/detailed_metrics_per_epoch.csv', index=False)
    print(f"✓ Saved detailed metrics (CSV): {output_dir}/detailed_metrics_per_epoch.csv")

    # 4. Save metadata about the results
    metadata = {
        'task': task,
        'n_folds': len(fold_summaries),
        'timestamp': datetime.now().isoformat(),
        'mean_val_accuracy': float(fold_summary_df['best_val_accuracy'].mean()),
        'std_val_accuracy': float(fold_summary_df['best_val_accuracy'].std()),
        'mean_val_auc': float(fold_summary_df['best_val_auc'].mean()),
        'std_val_auc': float(fold_summary_df['best_val_auc'].std()),
        'mean_val_balanced_accuracy': float(fold_summary_df['best_val_balanced_accuracy'].mean()),
        'std_val_balanced_accuracy': float(fold_summary_df['best_val_balanced_accuracy'].std()),
        'mean_eval_val_accuracy': float(fold_summary_df['eval_val_accuracy'].mean()),
        'std_eval_val_accuracy': float(fold_summary_df['eval_val_accuracy'].std()),
        'mean_eval_val_auc': float(fold_summary_df['eval_val_auc'].mean()),
        'std_eval_val_auc': float(fold_summary_df['eval_val_auc'].std()),
        'mean_eval_val_balanced_accuracy': float(fold_summary_df['eval_val_balanced_accuracy'].mean()),
        'std_eval_val_balanced_accuracy': float(fold_summary_df['eval_val_balanced_accuracy'].std()),
    }
    with open(f'{output_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {output_dir}/metadata.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, default='yaregan', choices=['yaregan', 'cbra', 'raw'], help='Features to be used in the classifier')
    parser.add_argument('--task', type=str, default='gender', choices=['gender', 'age'])
    parser.add_argument('--n-epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Classifier batch size')
    args = parser.parse_args()

    CHANNELS = ['O1', 'O2', 'P1', 'P2', 'C1', 'C2', 'F1', 'F2']
    FEATURES = args.features
    TASK = args.task
    EPOCHS = args.n_epochs
    BATCH_SIZE = args.batch_size

    if FEATURES == 'yaregan':
        print(f'>>>> Use Features Extracted from Yare-GAN')
        X_input, y, groups = load_data(TASK, channels=CHANNELS)
        X_e = extract_features(X_input, 'logs/eo/20260330_epoch_100.model.keras')

    elif FEATURES == 'cbra':
        print(f'>>>> Use Features Extracted from CBraMod')
        sub_ids = load_data(TASK, channels=CHANNELS, return_sub_ids=True)
        cbra_paths = {
            'gender': 'data/benchmarking/CBraMod_features_gender_seg-4s_balanced.pt',
            'age': 'data/benchmarking/ds005385_extracted_CBraMod_features_seg-4s.pt',
        }
        X_e, y, groups = load_CBraMod_features(TASK, cbra_paths[TASK], sub_ids)

    elif FEATURES == 'raw':
        print(f'>>>> Use Raw Signal')
        X_input, y, groups = load_data(TASK, channels=CHANNELS)
        X_e = X_input.flatten(1, 2).numpy()

    else:
        raise ValueError(f'Unknown feature type {FEATURES}')

    ##### Classifier
    all_fold_histories, fold_summaries = run_classification(X_e, y, groups, epochs=EPOCHS, batch_size=BATCH_SIZE)

    ##### Save Results
    print("="*80)
    print("SAVING K-FOLD CROSS-VALIDATION RESULTS")
    print("="*80)

    output_dir = f'logs/{FEATURES}_{TASK}_classifier_kfold_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(output_dir, exist_ok=True)

    fold_summary_df = pd.DataFrame(fold_summaries)
    fold_summary_df.to_csv(f'{output_dir}/fold_summary.csv', index=False)
    print(f"✓ Saved fold summary (CSV): {output_dir}/fold_summary.csv")

    with open(f'{output_dir}/all_fold_histories.pkl', 'wb') as f:
        pickle.dump(all_fold_histories, f)
    print(f"✓ Saved fold histories (Pickle): {output_dir}/all_fold_histories.pkl")

    create_and_save_detailed_metrics(all_fold_histories, fold_summaries, fold_summary_df, output_dir, TASK)
