
import os
os.environ['KERAS_BACKEND'] = 'torch'

from typing import List
import torch
import keras
import xarray as xr
from ...EEGModalNet import WGAN_GP_V0, preprocess_data
from scipy.signal import butter, sosfiltfilt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import class_weight
from keras import regularizers, layers
from meegkit import dss
import argparse


def load_OTKA_data(eeg_path: str,
                   demo_path: str,
                   channels: List[str],
                   downsample_data: bool = True,
                   time_dim: int = 512) -> tuple:
    
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

    sub_ids_formatted = [format_subject_id(sub_id) for sub_id in sub_ids]

    # X_input
    x = EEG.sel(subject=sub_ids_formatted, channel=channels).to_numpy()
    x = x.reshape(-1, *x.shape[2:])
    if downsample_data:  # remove NaNs
        x = np.concatenate([x[:101], x[103:]])
    else:
        x = np.concatenate([x[:205], x[207:]])

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


def load_CBraMod_features(feature_path):
        cbramod_dict = torch.load(feature_path, weights_only=False)
        X_e = np.asarray(cbramod_dict['features'])
        y = cbramod_dict['gender']
        groups = cbramod_dict['subject_ids']
        return X_e, y, groups


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-raw', action='store_true', help='Use flattened (raw) signal instead of features')
    parser.add_argument('--use-cbramod', action='store_true', help='Use features extracted from CBraMod instead of Yare-GAN')
    parser.add_argument('--model-path', type=str, default='logs/gender_cls_OTKA', help='Path for saving model and logs')
    parser.add_argument('--classifier', type=str, default='MLP', choices=['MLP', 'Convolution'],
                        help='Type of classifier to use: MLP or Convolution')
    args = parser.parse_args()

    CHANNELS = ['O1', 'O2', 'P1', 'P2', 'C1', 'C2', 'F1', 'F2']
    USE_CBRAMOD = args.use_cbramod
    USE_RAW = args.use_raw
    MODEL_PATH = args.model_path
    CLASSIFIER = args.classifier

    # Load weights
    model = WGAN_GP_V0(time_dim=512, feature_dim=len(CHANNELS),
                       latent_dim=128, n_subjects=202,
                       use_sublayer_generator=True,
                       use_sublayer_critic=True,
                       use_channel_merger_g=False,
                       use_channel_merger_c=False,
                       interpolation='bilinear')

    model.load_weights('logs/20250605/20250605_7th_epoch_1280.model.keras')
    critic = model.critic.model

    if USE_CBRAMOD:
        print(f'>>>> Use Features Extracted from CBraMod in {CLASSIFIER} Classifier')
        X_e, y, groups = load_CBraMod_features('data/benchmarking/CBraMod_features_gender_seg-4s_balanced.pt')
        if CLASSIFIER == 'Convolution':
            print('CbraMod Features shape', X_e.shape)
            X_e = X_e.reshape(X_e.shape[0], 200, -1)
            print(X_e.shape)
    else:
        X_input, y, groups = load_OTKA_data('data/OTKA/experiment_EEG_data.nc5',
                                            'data/OTKA/PLB_HYP_data_MASTER.csv',
                                            channels=CHANNELS,
                                            time_dim=512)
        if USE_RAW:
            print(f'>>>> Use Raw Signal in {CLASSIFIER} Classifier')
            if CLASSIFIER == 'MLP':
                X_e = X_input.flatten(1, 2)
            elif CLASSIFIER == 'Convolution':
                X_e = X_input

        else:
            print(f'>>>> Use Intermediate Features Extracted from Yare-GAN in {CLASSIFIER} Classifier')
            extractor = keras.Sequential([
                                          critic.layers[4],
                                          critic.layers[6],     
                                          ])
            X_e = extractor(X_input).detach().cpu()
            if CLASSIFIER == 'MLP':
                X_e = X_e.flatten(1, 2)

    random_state = 0 if USE_CBRAMOD else 8  # to ensure a balanced split
    group_shuffle = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=random_state)
    train_idx, val_idx = next(group_shuffle.split(X_e, y, groups=groups))
    print('Chance level', y.mean(), y[train_idx].mean(), y[val_idx].mean())

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = {'0': class_weights[0], '1': class_weights[1]}

    ##### Classifier
    if CLASSIFIER == 'MLP':
        cls_model = keras.models.Sequential([   
                    layers.Dense(512, activation='gelu',),
                    layers.Dropout(0.4),
                    layers.Dense(1, activation='sigmoid')
                    ])
    elif CLASSIFIER == 'Convolution':
        cls_model = keras.Sequential([
                    layers.Conv1D(filters=64, kernel_size=5, activation='gelu'),
                    layers.MaxPooling1D(pool_size=2),
                    layers.Flatten(),
                    layers.Dense(128, activation='relu'),
                    layers.Dense(1, activation='sigmoid', name='output')
                    ])
    else:
        raise ValueError(f'Unknown classifier type: {CLASSIFIER}')
    
    cls_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(f'{MODEL_PATH}_best_val_acc.model.keras',
                                        monitor='val_accuracy',
                                        save_best_only=True),
        keras.callbacks.CSVLogger(f'{MODEL_PATH}.csv'),
        keras.callbacks.TerminateOnNaN()
    ]

    history = cls_model.fit(X_e[train_idx],
                            y[train_idx],
                            epochs=1000,
                            batch_size=256,
                            validation_data=(X_e[val_idx], y[val_idx]),
                            class_weight=class_weights,
                            callbacks=callbacks,
                            shuffle=True)
