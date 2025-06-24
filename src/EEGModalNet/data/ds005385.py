from typing import List

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


def process_ds005385_for_CBraMod(sub_id: str,
                                 channels: List[str],
                                 resampling_frq=200,
                                 notchfilter_frq=50,
                                 filter_bounds=[0.3, 75],
                                 verbose=False,
                                 microvolts=False,
                                 pretrain=False):

    # open data
    raw = mne.io.read_raw_edf(f'ds005385/{sub_id}/ses-1/eeg/{sub_id}_ses-1_task-EyesClosed_acq-pre_eeg.edf', verbose=verbose, preload=True)
    raw.set_montage('standard_1020')

    # pick eeg channels
    print('picking eeg channels...')
    # ['T3', 'T4', 'T5', 'T6'] channels that are only in the 10-20 system
    # replaced with their equivalent name in the 10-10 system [T7, T8, P7, P8]
    raw.pick(channels, verbose=verbose)

    # interpolate bad channels if there is any
    print('interpolating bad channels...')
    raw.interpolate_bads(verbose=verbose)

    # resampling
    if resampling_frq is not None:
        raw.resample(resampling_frq, verbose=verbose)

    raw.filter(l_freq=filter_bounds[0], h_freq=filter_bounds[1], verbose=verbose)
    raw.notch_filter((notchfilter_frq), verbose=verbose)
    eeg_array = raw.get_data().T
    points, chs = eeg_array.shape
    if microvolts:
        eeg_array = eeg_array * 10**6

    if pretrain:  # pretrain mode follows their pretraining's preprocessing steps
        a = points % (30 * 200)
        eeg_array = eeg_array[60 * 200:-(a+60 * 200), :]
        eeg_array = eeg_array.reshape(-1, 30, 200, chs)

    else:
        eeg_array = eeg_array[:36000, :]  # HACK: trim the data based on the shortest recording and its divisiblity by 200
        eeg_array = eeg_array.reshape(-1, 2, 200, chs)

    eeg_array = eeg_array.transpose(0, 3, 1, 2)

    return eeg_array
