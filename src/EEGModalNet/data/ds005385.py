import numpy as np
import pandas as pd

import mne
import xarray as xr
from scipy.signal import butter, sosfiltfilt
from meegkit import dss
from src.EEGModalNet.preprocessing.preprocessing import preprocess_data


def subset_dataset(demographic_path,
                   n_participants,
                   output_path):
    """Subset the dataset based on demographic information."""

    demo = pd.read_csv(demographic_path, sep='\t')
    male_id = demo.query('sex == "M"')['participant_id'].iloc[:n_participants].values
    female_id = demo.query('sex == "F"')['participant_id'].iloc[:n_participants].values

    demo['age_group'] = demo['age'].apply(lambda x: 'young' if x <= 45 else 'elderly')
    young_id = demo.query('age_group == "young"')['participant_id'].iloc[:n_participants].values
    elderly_id = demo.query('age_group == "elderly"')['participant_id'].iloc[:n_participants].values

    subset_id = np.concatenate([male_id, female_id, young_id, elderly_id])
    subset_id = sorted(list(set(subset_id)))

    subset_id.remove('sub-230') # drop sub-230 that has partial data

    # save subject ids list
    with open(output_path, 'w') as f:
        for sub_id in subset_id:
            f.write(f"{sub_id}\n")


def create_dataset(bids_root_path,
                   output_path,
                   subject_id,
                   channels):
    """Create a dataset based on the participants ids"""

    # open each data and concate them and save them in a xarray dataset
    for sub_id in subject_id:
        print(sub_id)
        raw = mne.io.read_raw_edf(f'{bids_root_path}/{sub_id}/ses-1/eeg/{sub_id}_ses-1_task-EyesClosed_acq-pre_eeg.edf', verbose=0)
        raw.pick_channels(channels, verbose=0)
        x = raw.get_data()

        # preprocess
        x = raw.get_data()
        x = x.reshape(1, *x.shape)
        x = preprocess_data(x, sampling_rate=1000)
        x, _ = dss.dss_line(x.T, sfreq=1000, fline=50)
        x = x.T

        from scipy.signal import resample
        sos = butter(4, 0.5, btype='high', fs=1000, output='sos')
        x = sosfiltfilt(sos, x, axis=-1)

        n_samples = int((x.shape[-1] / 1000) * 128)
        x = resample(x, num=n_samples, axis=-1)
        cutoff = 23168
        x = x[:, :, :cutoff]  # HACK to prevent padding, 23168 is the shortest duration after resampling

        x = xr.DataArray(x, dims=['subject', 'channel', 'time'], coords={'subject': [sub_id], 'channel': channels, 'time': np.arange(0, cutoff, 1)})
        if sub_id == subject_id[0]:
            dataset = x
        else:
            dataset = xr.concat([dataset, x], dim='subject')

    # save
    dataset.to_netcdf(output_path, engine='h5netcdf')
