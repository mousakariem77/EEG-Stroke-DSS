"""
Module 1 (Pre-treatment): Adapted Power Spectral Density (PSD) Feature Extraction

Bouazizi & Ltifi (2024), Section 4.3.2:
"subjecting artifact-free 4-s EEG epochs (with a 1/4 Hz resolution) to Fast Fourier
Transforms (FFT)... relative power (RP) and the delta/alpha ratio (DAR) and
delta/theta ratio (DTR) are calculated..."
"""
from typing import Dict, Tuple, List, Optional
import numpy as np
from scipy import signal


# Standard EEG Frequency Bands defined in Paper
FREQUENCY_BANDS = {
    'Delta': (0.5, 4.0),
    'Theta': (4.0, 8.0),
    'Alpha': (8.0, 13.0),
    'Beta': (13.0, 30.0)
}


def compute_epoch_psd(
    epoch: np.ndarray,
    sampling_rate: float = 128.0,
    window: str = 'hann'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density (PSD) of a single 4-second epoch using FFT.

    With epoch length = 4s, the frequency resolution Delta_f = 1 / 4s = 0.25 Hz (1/4 Hz),
    matching the paper specification.

    Args:
        epoch: 1D array of length N (e.g. 512 samples for 4s at 128 Hz).
        sampling_rate: Sampling frequency in Hz.
        window: Window function to apply before FFT ('hann' or 'boxcar').

    Returns:
        freqs: Frequency bins array (Hz).
        psd: Power spectral density values (uV^2 / Hz).
    """
    n_samples = len(epoch)
    freqs, psd = signal.welch(
        epoch,
        fs=sampling_rate,
        window=window,
        nperseg=n_samples,
        noverlap=0,
        scaling='density'
    )
    return freqs, psd


def extract_band_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    band: Tuple[float, float]
) -> float:
    """
    Compute absolute power in a specific frequency band by integrating the PSD.

    Args:
        freqs: Frequency bins.
        psd: PSD values.
        band: (low_freq, high_freq) tuple.

    Returns:
        Absolute band power (uV^2).
    """
    idx_band = np.logical_and(freqs >= band[0], freqs < band[1])
    if not np.any(idx_band):
        return 0.0
    
    # Numerical integration using composite trapezoidal rule
    freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 0.25
    band_power = float(np.trapezoid(psd[idx_band], dx=freq_res))
    return max(band_power, 1e-12)


def extract_eeg_features_from_epochs(
    clean_epochs: np.ndarray,
    sampling_rate: float = 128.0,
    bands: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, float]:
    """
    Extract all frequency-based features from artifact-free EEG epochs as described in the paper:
    - Absolute power: Delta, Theta, Alpha, Beta, Total Power
    - Relative power: RP Delta, RP Theta, RP Alpha, RP Beta
    - Clinical ratios: DAR (Delta/Alpha Ratio), DTR (Delta/Theta Ratio)
    - Epoch: Number of valid clean epochs

    Averages the PSD across all valid epochs before computing band powers and ratios.

    Args:
        clean_epochs: Array of shape (n_epochs, samples_per_epoch).
        sampling_rate: Sampling frequency in Hz.
        bands: Dictionary of frequency bands. Defaults to FREQUENCY_BANDS.

    Returns:
        Dictionary of extracted clinical EEG features.
    """
    if bands is None:
        bands = FREQUENCY_BANDS

    n_epochs = len(clean_epochs)
    if n_epochs == 0:
        raise ValueError("No clean epochs provided for feature extraction.")

    # Compute PSD for each epoch and average across epochs
    all_psd = []
    freqs = None
    for ep in clean_epochs:
        f, p = compute_epoch_psd(ep, sampling_rate=sampling_rate)
        all_psd.append(p)
        if freqs is None:
            freqs = f

    mean_psd = np.mean(all_psd, axis=0)
    if freqs is None:
        raise ValueError("Failed to compute PSD: frequency bins could not be determined.")

    # Compute absolute power in each band
    abs_powers = {}
    for band_name, band_range in bands.items():
        abs_powers[band_name] = extract_band_power(freqs, mean_psd, band_range)

    # Total power across 0.5 - 30.0 Hz
    total_power = extract_band_power(freqs, mean_psd, (0.5, 30.0))
    if total_power <= 1e-12:
        total_power = sum(abs_powers.values())

    # Relative Powers (RP)
    rp_delta = abs_powers['Delta'] / total_power
    rp_theta = abs_powers['Theta'] / total_power
    rp_alpha = abs_powers['Alpha'] / total_power
    rp_beta = abs_powers['Beta'] / total_power

    # Clinical ratios: DAR and DTR
    dtr = abs_powers['Delta'] / max(abs_powers['Theta'], 1e-9)
    dar = abs_powers['Delta'] / max(abs_powers['Alpha'], 1e-9)

    features = {
        'Delta': float(abs_powers['Delta']),
        'Theta': float(abs_powers['Theta']),
        'Alpha': float(abs_powers['Alpha']),
        'Beta': float(abs_powers['Beta']),
        'Total Power': float(total_power),
        'RP Delta': float(rp_delta),
        'RP Theta': float(rp_theta),
        'RP Alpha': float(rp_alpha),
        'RP Beta': float(rp_beta),
        'DTR': float(dtr),
        'DAR': float(dar),
        'Epoch': float(n_epochs)
    }

    return features
