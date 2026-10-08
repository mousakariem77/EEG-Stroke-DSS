"""
Module 1 (Pre-treatment): End-to-End Raw EEG Processing Pipeline

Combines filtering, artifact rejection, epoching, and adapted PSD feature extraction
into a unified callable pipeline as defined in Bouazizi & Ltifi (2024), Section 3.1 & 4.3.
"""
from typing import Dict, Tuple, Optional
import numpy as np

from src.preprocessing.filters import butter_bandpass_filter, notch_filter
from src.preprocessing.epoching import create_epochs
from src.preprocessing.artifacts import reject_artifact_epochs
from src.preprocessing.psd import extract_eeg_features_from_epochs


def process_raw_eeg_signal(
    raw_signal: np.ndarray,
    sampling_rate: float = 128.0,
    epoch_duration_sec: float = 4.0,
    amplitude_threshold_uv: float = 100.0,
    lowcut: float = 0.5,
    highcut: float = 30.0
) -> Tuple[Dict[str, float], Dict]:
    """
    Run full Module 1 pipeline on continuous 1D raw EEG waveform:
    1. Bandpass filter (0.5 - 30 Hz zero-phase Butterworth).
    2. Segment into fixed 4-second epochs (0.25 Hz FFT resolution).
    3. Exclude epochs with amplitude exceeding +/- 100 uV or severe movement/muscle artifact.
    4. Compute Adapted PSD and extract band powers, Relative Powers, DAR, and DTR.

    Args:
        raw_signal: Continuous 1D EEG signal in microvolts (uV).
        sampling_rate: Sampling rate in Hz (default: 128 Hz).
        epoch_duration_sec: Epoch length in seconds (default: 4.0s).
        amplitude_threshold_uv: Amplitude threshold (default: 100.0 uV).
        lowcut: Filter lower frequency bound (default: 0.5 Hz).
        highcut: Filter upper frequency bound (default: 30.0 Hz).

    Returns:
        features: Dictionary containing:
            'Delta', 'Theta', 'Alpha', 'Beta', 'Total Power',
            'RP Delta', 'RP Theta', 'RP Alpha', 'RP Beta',
            'DTR', 'DAR', 'Epoch'
        pipeline_report: Information about processing, retained epochs, and filtration.
    """
    # Step 1: Bandpass filtering (0.5 - 30 Hz)
    filtered = butter_bandpass_filter(
        raw_signal,
        lowcut=lowcut,
        highcut=highcut,
        sampling_rate=sampling_rate
    )

    # Step 2: Epoching (4s chunks)
    epochs = create_epochs(
        filtered,
        sampling_rate=sampling_rate,
        epoch_duration_sec=epoch_duration_sec
    )

    # Step 3: Artifact exclusion (+/- 100 uV threshold)
    clean_epochs, retained_mask, rejection_report = reject_artifact_epochs(
        epochs,
        threshold_uv=amplitude_threshold_uv
    )

    if len(clean_epochs) == 0:
        raise ValueError(
            "All epochs were rejected due to artifacts or excessive amplitude exceeding +/- 100 uV."
        )

    # Step 4: Adapted PSD feature extraction
    features = extract_eeg_features_from_epochs(
        clean_epochs,
        sampling_rate=sampling_rate
    )

    pipeline_report = {
        "sampling_rate": sampling_rate,
        "filter_band": (lowcut, highcut),
        "amplitude_threshold_uv": amplitude_threshold_uv,
        "total_epochs": rejection_report["total_epochs"],
        "clean_epochs": rejection_report["clean_epochs"],
        "rejected_epochs": rejection_report["rejected_epochs"],
        "retention_rate": rejection_report["retention_rate"]
    }

    return features, pipeline_report
