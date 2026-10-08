"""
Module 1 (Pre-treatment): EEG Epoching

Segments continuous EEG recording into fixed 4-second epochs as specified in
Bouazizi & Ltifi (2024), Section 4.3.2:
"subjecting artifact-free 4-s EEG epochs (with a 1/4 Hz resolution) to Fast Fourier Transforms"
"""
from typing import Tuple
import numpy as np


def create_epochs(
    raw_signal: np.ndarray,
    sampling_rate: float = 128.0,
    epoch_duration_sec: float = 4.0,
    overlap_ratio: float = 0.0
) -> np.ndarray:
    """
    Split continuous 1D EEG signal into fixed 4-second epochs.

    With epoch_duration = 4.0 seconds, FFT frequency resolution will be
    Delta_f = 1 / 4.0 = 0.25 Hz (1/4 Hz), exactly matching the paper.

    Args:
        raw_signal: Continuous 1D EEG signal in microvolts (uV).
        sampling_rate: Sampling frequency in Hz (e.g., 128 Hz or 256 Hz).
        epoch_duration_sec: Epoch length in seconds (default: 4.0 seconds).
        overlap_ratio: Overlap between consecutive epochs (0.0 = non-overlapping).

    Returns:
        Array of shape (n_epochs, samples_per_epoch).
    """
    samples_per_epoch = round(epoch_duration_sec * sampling_rate)
    if len(raw_signal) < samples_per_epoch:
        raise ValueError(
            f"Signal length ({len(raw_signal)}) is shorter than one epoch ({samples_per_epoch} samples)."
        )

    step_size = round(samples_per_epoch * (1.0 - overlap_ratio))
    if step_size < 1:
        step_size = 1

    n_epochs = (len(raw_signal) - samples_per_epoch) // step_size + 1
    epochs = np.zeros((n_epochs, samples_per_epoch), dtype=np.float64)

    for i in range(n_epochs):
        start = i * step_size
        end = start + samples_per_epoch
        epochs[i] = raw_signal[start:end]

    return epochs
