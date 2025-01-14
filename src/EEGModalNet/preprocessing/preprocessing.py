# =============================================================================
import pandas as pd
from pathlib import Path
from typing import List
import numpy as np
from mne.io import read_raw_eeglab
from sklearn.preprocessing import RobustScaler, StandardScaler


def preprocess_data(data, baseline_duration=0.5, sampling_rate=128):
    # Step 1: Baseline correction (subtract the mean of the first 0.5 seconds for each channel)
    sample_size = data.shape[0]
    baseline_samples = int(baseline_duration * sampling_rate)
    baseline_mean = np.mean(data[:, :, :baseline_samples], axis=-1, keepdims=True)
    data_corrected = data - baseline_mean
    print_stats(data_corrected, 'Corrected')

    # Step 2: Normalize using median and IQR
    scaler = RobustScaler()
    data_corrected = data_corrected.transpose(0, 2, 1)  # Transpose for sklearn (samples, times, features)
    data_scaled = np.array([scaler.fit_transform(data_corrected[i]) for i in range(sample_size)])
    print_stats(data_scaled, 'Scaled')

    # Step 3: Z-score normalization
    normalizer = StandardScaler()
    data_normalized = np.array([normalizer.fit_transform(data_scaled[i]) for i in range(sample_size)]).transpose(0, 2, 1)
    print_stats(data_normalized, 'Normalized')

    # Step 4: Clamp values greater than 20 standard deviations
    std_threshold = 20
    data_clamped = np.clip(data_normalized, -std_threshold, std_threshold)
    print_stats(data_clamped, 'Clamped')

    return data_clamped


def print_stats(x, name):
    print(f">>>>{name} Mean: {x.mean()}, Std: {x.std()}, Max: {x.max()}, Min: {x.min()}")


def find_excluded_channels(data_path: str, full_ch_list: List[str], pattern: str):
    exc_chs = {}
    for i in Path(data_path).glob(pattern):
        raw = read_raw_eeglab(i, verbose=False)
        sub = i.stem[:-3]
        excluded_chs = [i for i in full_ch_list if i not in raw.ch_names]
        exc_chs[sub] = [excluded_chs]
        df = pd.DataFrame.from_dict(exc_chs, orient='index', columns=['excluded_channels'])
        # replace the empty list with None
        df['bad_channels'] = df['bad_channels'].apply(lambda x: None if x == [] else x)
    return df
