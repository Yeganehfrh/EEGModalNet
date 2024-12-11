import torch


def compute_power(signal):
    fft = torch.fft.rfft(signal, dim=1)
    return torch.abs(fft) ** 2


def smoothness_penalty(power_spectrum):
    """
    Compute a smoothness penalty for the power spectrum.
    power_spectrum: Tensor of shape (batch_size, freq_dim).
    Returns: Scalar penalty term.
    """
    # Compute differences between adjacent frequency bins
    diff = power_spectrum[:, 1:] - power_spectrum[:, :-1]
    # Penalize large differences (L2 norm)
    penalty = torch.mean(torch.square(diff))
    return penalty


def smoothness_loss(penalty_real, penalty_fake):
    return torch.abs(penalty_real - penalty_fake) / 1e5


def spectral_match_loss(real_psd, fake_psd):
    return torch.mean(torch.square(real_psd - fake_psd)) / 1e5


def spectral_regularization_loss(real_data, fake_data, lambda_smooth=1.0, lambda_match=1.0, include_smooth=False):
    real_psd = compute_power(real_data)
    fake_psd = compute_power(fake_data)
    penalty_real = smoothness_penalty(real_psd)
    penalty_fake = smoothness_penalty(fake_psd)
    smoothness_loss_value = lambda_smooth * smoothness_loss(penalty_real, penalty_fake)
    spectral_match_loss_value = lambda_match * spectral_match_loss(real_psd, fake_psd)
    if include_smooth:
        return smoothness_loss_value + spectral_match_loss_value
    else:
        return spectral_match_loss_value
