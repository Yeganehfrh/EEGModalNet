import numpy as np
import xarray as xr
from scipy.signal import resample
from sklearn.preprocessing import RobustScaler, StandardScaler


def print_stats(x, name):
    print(f">>>>{name} Mean: {x.mean()}, Std: {x.std()}, Max: {x.max()}, Min: {x.min()}")


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


if __name__ == '__main__':
    downsample_frq = 98
    sampling_rate = 128
    data_path = 'data/LEMON_data/eeg_eo_ec.nc5'
    xarray = xr.open_dataset(data_path, engine='h5netcdf')

    subjects = xarray.subject.to_numpy()
    x = xarray['eye_closed'].to_numpy()
    channels = xarray.channel.to_numpy()

    print(f'>>> downsampling the data to {downsample_frq} Hz')
    n_samples = int((x.shape[-1] / sampling_rate) * downsample_frq)
    x = resample(x, num=n_samples, axis=-1)
    sampling_rate = downsample_frq

    print('preprocessing...')
    data = preprocess_data(x, sampling_rate=sampling_rate)

    # create a dataaraay and save
    xarray = xr.DataArray(data, dims=['subject', 'channel', 'time'])
    xarray = xarray.assign_coords(subject=subjects, channel=channels)

    # Assign Attributes to the new xarray
    xarray_old = xr.open_dataarray('data/LEMON_data/eeg_EC_BaseCorr_Norm_Clamp_with_pos.nc5', engine='h5netcdf')
    xarray = xarray.assign_attrs({'gender':xarray_old.attrs['gender'],
                                'ch_positions':xarray_old.attrs['ch_positions']},)

    # save
    data_path_save = 'data/LEMON_data/EC_all_channels_processed_downsampled.nc5'
    xarray.to_netcdf(data_path_save, engine='h5netcdf')
