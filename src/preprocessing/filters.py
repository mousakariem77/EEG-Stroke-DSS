"""
Module 1 (Pre-treatment): Digital Filtering for EEG Signals

Implements bandpass filtering (0.5 - 30.0 Hz) as specified in Bouazizi & Ltifi (2024),
Section 4.3.1: "amplifying the raw EEG waveform data using a band-pass filter (0.5–30 Hz)
to remove noise outside the desired frequency range".
"""
from typing import Tuple, Union
import numpy as np
from scipy import signal


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 30.0,
    sampling_rate: float = 128.0,
    order: int = 4
) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass filter to EEG signal.

    Args:
        data: 1D or 2D array of EEG signal in microvolts (uV).
        lowcut: Low cutoff frequency in Hz (default: 0.5 Hz).
        highcut: High cutoff frequency in Hz (default: 30.0 Hz).
        sampling_rate: Sampling frequency in Hz (default: 128.0 Hz).
        order: Filter order (default: 4).

    Returns:
        Filtered EEG signal with identical shape and zero phase distortion.
    """
    nyquist = 0.5 * sampling_rate
    if lowcut <= 0 or highcut >= nyquist:
        raise ValueError(
            f"Frequencies must be 0 < lowcut ({lowcut}) < highcut ({highcut}) < Nyquist ({nyquist})."
        )
    
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='bandpass')
    
    # Zero-phase forward-backward filtering to avoid phase delay
    filtered = signal.filtfilt(b, a, data, axis=-1)
    return filtered


def notch_filter(
    data: np.ndarray,
    freq: float = 50.0,
    sampling_rate: float = 128.0,
    quality_factor: float = 30.0
) -> np.ndarray:
    """
    Apply notch filter to suppress powerline electrical interference (50 Hz or 60 Hz).

    Args:
        data: EEG signal array.
        freq: Powerline frequency to notch out in Hz (50.0 or 60.0).
        sampling_rate: Sampling frequency in Hz.
        quality_factor: Quality factor Q (default: 30.0).

    Returns:
        Notch-filtered EEG signal.
    """
    nyquist = 0.5 * sampling_rate
    if freq >= nyquist:
        return data  # Powerline frequency is above Nyquist; no-op
    
    b, a = signal.iirnotch(freq, quality_factor, sampling_rate)
    return signal.filtfilt(b, a, data, axis=-1)
