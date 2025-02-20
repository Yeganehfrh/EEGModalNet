
import os
os.environ['KERAS_BACKEND'] = 'torch'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import keras
import xarray as xr
from ...EEGModalNet import WGAN_GP_old
from scipy.signal import butter, sosfiltfilt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def load_data(data_path: str,
              n_subjects: int = 202,
              bandpass_filter: float = 1.0,
              time_dim: int = 1024,
              exclude_sub_ids=None) -> tuple:

    channels = ['O1', 'O2', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2']

    xarray = xr.open_dataarray(data_path, engine='h5netcdf')
    x = xarray.sel(subject=xarray.subject[:n_subjects], channel=channels)

    if exclude_sub_ids is not None:
        x = x.sel(subject=~x.subject.isin(exclude_sub_ids))

    x = x.to_numpy()
    n_subjects = x.shape[0]

    if bandpass_filter is not None:
        sos = butter(4, bandpass_filter, btype='high', fs=98, output='sos')
        x = sosfiltfilt(sos, x, axis=-1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.tensor(x.copy(), device=device).unfold(2, time_dim, time_dim).permute(0, 2, 3, 1).flatten(0, 1)  # TODO: copy was added because of an error, look into this

    sub = torch.tensor(np.arange(0, n_subjects).repeat(x.shape[0] // n_subjects)[:, np.newaxis])
    labels = xarray.gender - 1
    y = labels.repeat(x.shape[0] // 202)
    y = torch.tensor(y, device=device)
    sub_ids_classifier = sub.squeeze().cpu().numpy()

    return x, y, sub_ids_classifier


if __name__ == '__main__':

    if torch.cuda.is_available():
        print('GPU is available')
        torch.cuda.current_device()
    else:
        print('GPU is not available!!')
        exit()

    print(f'Running on {torch.cuda.device_count()} GPUs')
    print(f'Using CUDA device: {torch.cuda.get_device_name(0)}')

    # Explicitly set the CUDA device
    torch.cuda.set_device(0)

    # preload CUDA libraries with a dummy tensor
    _ = torch.randn(1, device="cuda")

    # Apply mixed precision policy
    keras.mixed_precision.set_global_policy('mixed_float16')
    print(f'Global policy is {keras.mixed_precision.global_policy().name}')

    X_input, y, groups = load_data('data/LEMON_DATA/EC_all_channels_processed_downsampled.nc5',
                                   n_subjects=202,
                                   bandpass_filter=0.5,
                                   time_dim=512,
                                   exclude_sub_ids=None)

    group_shuffle = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=5)  # random state == 9
    print('>>>>>>>>>>>')
    train_idx, val_idx = next(group_shuffle.split(X_input, y, groups=groups))
    # print('Chance level',
    #       np.unique(y[train_idx], return_counts=True)[1] / len(y[train_idx]), np.unique(y[val_idx], return_counts=True)[1] / len(y[val_idx]))

    from sklearn.utils import class_weight
    # class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y.cpu().numpy()), y=y)
    # class_weights = {'0': class_weights[0], '1': class_weights[1]}

    model = WGAN_GP_old(time_dim=512, feature_dim=8,
                        latent_dim=128, n_subjects=202,
                        use_sublayer_generator=True,
                        use_sublayer_critic=True,
                        use_channel_merger_g=False,
                        use_channel_merger_c=False,
                        interpolation='bilinear')

    model.load_weights('logs/06022025/06.02.2025_epoch_2500.model.keras')
    critic = model.critic.model

    critic_output = critic.get_layer('dis_flatten').output  # the 4096-dim layer
    critic_output = keras.layers.Dropout(0.2)(critic_output)
    new_output = keras.layers.Dense(1, activation='sigmoid', name='classification_head')(critic_output)

    new_model = keras.Model(inputs=critic.layers[0].input, outputs=new_output)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    new_model.to(device)
    print(f'>>>> Model is on {device}')

    # 4. Freeze the original layers
    for layer in new_model.layers[:-7]:
        layer.trainable = False

    # 5. Compile and train
    new_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    torch.cuda.synchronize()

    history = new_model.fit(X_input[train_idx], y[train_idx],
                            epochs=1000,
                            batch_size=128,
                            validation_data=(X_input[val_idx], y[val_idx]),
                            # class_weight=class_weights
                            )

    pd.DataFrame(history.history).to_csv('logs/geneder_classification_gpu.csv')
    new_model.save('logs/classifier_gpu.model.keras')
