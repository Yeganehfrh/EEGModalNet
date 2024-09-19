import os
import numpy as np
import xarray as xr
import pandas as pd
os.environ['KERAS_BACKEND'] = 'torch'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU usage
from sklearn.model_selection import StratifiedGroupKFold
from scipy.signal import butter, sosfilt

from ...EEGModalNet import ResidualBlock

import torch
import keras
from keras import layers
from keras import regularizers
import xarray as xr
import pandas as pd


def load_data(eeg_data_path,
              session_data_path,
              channels,
              time_dim=512,
              n_subject=51,
              n_sessions=2,
              n_splits=5,
              highpass_filter=True,):

    EEG_data = xr.open_dataarray(eeg_data_path, engine='h5netcdf')

    # open session data
    session_data = pd.read_csv(session_data_path)
    described_as_hyp = session_data.query('description == "hypnosis"')[['bids_id', 'session', 'score', 'procedure']].set_index('bids_id')

    X_input = EEG_data.sel(subject=EEG_data.subject[:n_subject], channel=channels).to_numpy()

    # including only hypnosis sessions
    X_input_ = np.zeros([n_subject, n_sessions, len(channels), X_input.shape[-1]])
    for i in range(X_input.shape[0]):
        ses = described_as_hyp.loc[i+1, 'session'].values - 1
        X_input_[i] = X_input[i, ses, :, :]

    # preparing x
    if highpass_filter:
        sos = butter(4, 1, btype='high', fs=128, output='sos')
        X_input_ = sosfilt(sos, X_input_, axis=-1)

    X_input_ = torch.tensor(X_input_.squeeze()).unfold(-1, time_dim, time_dim).permute(0, 1, 3, 4, 2).flatten(0, 1)

    # prepare y
    described_as_hyp['binary_scores'] = described_as_hyp['score'].apply(lambda x: 0 if x > 6 else 1)
    y = described_as_hyp.loc[:n_subject, 'binary_scores']
    groups = y.index.to_numpy() - 1  # -1 because the bids_id are not zero based unlike the indices
    y = y.values

    # prepare Kfold cross validation
    group_kfold = StratifiedGroupKFold(n_splits=n_splits)
    train_val_splits = []
    for train_idx, val_idx in group_kfold.split(X_input_, y, groups=groups):
        train_val_splits.append((train_idx, val_idx))

    y = torch.tensor(y).reshape(-1, 1).repeat(1, X_input_.shape[1])

    return X_input_, y, train_val_splits


class Critic(keras.Model):
    def __init__(self, time_dim, feature_dim, l2_lambda=0.01, dropout_rate=0.2, use_sublayer=False):
        super(Critic, self).__init__()

        self.input_shape = (time_dim, feature_dim)
        self.use_sublayer = use_sublayer

        self.model = keras.Sequential([
            keras.Input(shape=self.input_shape),
            ResidualBlock(feature_dim * 4, 5, activation='relu'),
            layers.Conv1D(2, 5, padding='same', activation='relu', name='conv3', kernel_regularizer=regularizers.L2(l2_lambda)),
            layers.Conv1D(1, 5, padding='same', activation='relu', name='conv4', kernel_regularizer=regularizers.L2(l2_lambda)),
            layers.Flatten(name='dis_flatten'),
            layers.Dense(512, name='dis_dense1', activation='relu', kernel_regularizer=regularizers.L2(l2_lambda)),
            layers.Dropout(dropout_rate),
            layers.Dense(128, name='dis_dense2', activation='relu', kernel_regularizer=regularizers.L2(l2_lambda)),
            layers.Dropout(dropout_rate),
            layers.Dense(32, name='dis_dense3', activation='relu', kernel_regularizer=regularizers.L2(l2_lambda)),
            layers.Dropout(dropout_rate),
            layers.Dense(8, name='dis_dense4', activation='relu', kernel_regularizer=regularizers.L2(l2_lambda)),
            layers.Dropout(dropout_rate),
            layers.Dense(1, name='sigmoid', activation='sigmoid', kernel_regularizer=regularizers.L2(l2_lambda))
        ], name='critic')

    def call(self, inputs):
        return self.model(inputs)


def build_model(time_dim, feature_dim,):
    model = Critic(time_dim=time_dim, feature_dim=feature_dim)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':

    eeg_data_path = 'data/OTKA/experiment_EEG_data.nc5'
    session_data_path = 'data/OTKA/behavioral_data.csv'
    channels = ['Oz', 'Fz', 'Cz', 'Pz', 'Fp1', 'Fp2', 'F1', 'F2']
    time_dim = 512
    n_splits = 5
    n_epochs = 300
    n_subject = 51
    X_input_hyp, y, train_val_splits = load_data(eeg_data_path, session_data_path, channels, time_dim=time_dim,
                                                 n_subject=n_subject, n_splits=n_splits)
    all_val_acc = []
    all_acc = []
    all_loss = []
    all_val_loss = []

    for i in range(2, n_splits):
        print(f'>>>>>> Fold {i+1}')
        model = build_model(time_dim, len(channels))
        train_idx, val_idx = train_val_splits[i]
        history = model.fit(X_input_hyp[train_idx].flatten(0, 1), y[train_idx].flatten(0, 1),
                            epochs=n_epochs,
                            batch_size=64,
                            validation_data=(X_input_hyp[val_idx].flatten(0, 1), y[val_idx].flatten(0, 1)))

        all_val_acc.append(history.history['val_accuracy'])
        all_acc.append(history.history['accuracy'])
        all_loss.append(history.history['loss'])
        all_val_loss.append(history.history['val_loss'])

        # save the model
        model.save(f'logs/model_{i+1}_2nd.model.keras')

    # restore all the parameters into one dataframe and save it
    all_val_acc, all_acc = np.array(all_val_acc), np.array(all_acc)
    all_loss, all_val_loss = np.array(all_loss), np.array(all_val_loss)
    history = {'val_accuracy': all_val_acc.flatten(), 'accuracy': all_acc.flatten(),
               'loss': all_loss.flatten(), 'val_loss': all_val_loss.flatten()}

    pd.DataFrame(history).to_csv('logs/history_last3folds.csv')
