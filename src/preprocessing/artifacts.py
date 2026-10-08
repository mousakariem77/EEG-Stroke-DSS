"""
Module 1 (Pre-treatment): Artifact Rejection and Amplitude Thresholding

Bouazizi & Ltifi (2024), Section 4.3.1:
"Finally, any remaining epochs with amplitudes exceeding +/- 100 uV are removed
to eliminate noisy or distorted data unsuitable for further analysis."
"""
from typing import Tuple, List, Dict
import numpy as np


def threshold_amplitude_check(
    epoch: np.ndarray,
    threshold_uv: float = 100.0
) -> bool:
    """
    Check if an EEG epoch exceeds the +/- threshold in microvolts (uV).

    Args:
        epoch: 1D array of EEG samples in uV.
        threshold_uv: Maximum allowed absolute amplitude in uV (default: 100.0 uV).

    Returns:
        True if the epoch is clean (within [-threshold, +threshold]), False if contaminated.
    """
    max_abs = np.max(np.abs(epoch))
    return bool(max_abs <= threshold_uv)


def reject_artifact_epochs(
    epochs: np.ndarray,
    threshold_uv: float = 100.0,
    variance_z_threshold: float = 3.5
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Filter epochs by rejecting those exceeding +/- 100 uV or having abnormal muscle/movement variance.

    Args:
        epochs: Array of shape (n_epochs, n_samples) containing EEG epochs.
        threshold_uv: Amplitude threshold in uV (default: 100.0 uV).
        variance_z_threshold: Z-score threshold on epoch variance for movement/muscle artifact detection.

    Returns:
        clean_epochs: Array of retained clean epochs.
        retained_indices: 1D boolean array indicating retained epochs.
        rejection_report: Summary dictionary with counts and rejection reasons.
    """
    n_epochs = len(epochs)
    if n_epochs == 0:
        return epochs, np.array([], dtype=bool), {"total": 0, "clean": 0, "rejected": 0}

    # Amplitude check (+/- 100 uV threshold)
    amp_mask = np.array([threshold_amplitude_check(ep, threshold_uv) for ep in epochs], dtype=bool)

    # Statistical variance check (detects high-frequency muscle / movement burst artifacts)
    variances = np.var(epochs, axis=1)
    mean_var = np.mean(variances)
    std_var = np.std(variances)
    if std_var > 1e-9:
        var_z = (variances - mean_var) / std_var
        var_mask = var_z < variance_z_threshold
    else:
        var_mask = np.ones(n_epochs, dtype=bool)

    retained_mask = amp_mask & var_mask
    clean_epochs = epochs[retained_mask]

    report = {
        "total_epochs": n_epochs,
        "clean_epochs": int(np.sum(retained_mask)),
        "rejected_epochs": int(n_epochs - np.sum(retained_mask)),
        "rejected_by_amplitude": int(np.sum(~amp_mask)),
        "rejected_by_variance": int(np.sum(~var_mask & amp_mask)),
        "retention_rate": float(np.mean(retained_mask))
    }

    return clean_epochs, retained_mask, report
