"""
Module 1 (Pre-treatment): Synthetic Resting-State EEG Signal Generator

Generates realistic 1-channel resting-state EEG signals (FP1 frontal electrode)
for testing the end-to-end signal processing and classification pipeline.

Simulates electrophysiological signatures:
- Control/Healthy: Prominent Alpha rhythm (8-12 Hz) with low slow-wave Delta/Theta.
- Acute Stroke: Significant delta-theta slowing (0.5-7 Hz), reduced Alpha power,
  and potential occasional blink/muscle artifact spikes.
"""
from typing import Tuple, Dict, Optional
import numpy as np


def generate_synthetic_raw_eeg(
    duration_sec: float = 60.0,
    sampling_rate: float = 128.0,
    patient_type: str = 'stroke',
    add_artifacts: bool = True,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Generate a continuous single-channel EEG signal (in microvolts, uV).

    Args:
        duration_sec: Recording duration in seconds (e.g., 60 seconds).
        sampling_rate: Sampling frequency in Hz (e.g., 128 Hz).
        patient_type: 'stroke' or 'control'.
        add_artifacts: Whether to inject realistic occasional blinks/muscle artifacts.
        random_state: Seed for reproducibility.

    Returns:
        signal: 1D array of continuous EEG signal in uV.
        metadata: Dictionary of generation parameters.
    """
    rng = np.random.RandomState(random_state)
    n_samples = int(duration_sec * sampling_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    # Base noise (pink / 1/f noise component typical of brain EEG)
    pink_noise = np.cumsum(rng.randn(n_samples)) * 0.05
    pink_noise = pink_noise - np.mean(pink_noise)

    if patient_type.lower() == 'stroke':
        # Stroke signature: strong Delta (0.5-3.5 Hz) and Theta (4-7 Hz), attenuated Alpha (9-11 Hz)
        delta_component = 35.0 * np.sin(2 * np.pi * 1.5 * t + rng.uniform(0, 2*np.pi)) + \
                          25.0 * np.sin(2 * np.pi * 2.8 * t + rng.uniform(0, 2*np.pi))
        theta_component = 20.0 * np.sin(2 * np.pi * 5.5 * t + rng.uniform(0, 2*np.pi))
        alpha_component = 5.0 * np.sin(2 * np.pi * 9.5 * t + rng.uniform(0, 2*np.pi))
        beta_component  = 4.0 * np.sin(2 * np.pi * 18.0 * t + rng.uniform(0, 2*np.pi))
    else:
        # Control signature: prominent Alpha rhythm (9-11 Hz), minimal Delta/Theta
        delta_component = 8.0 * np.sin(2 * np.pi * 1.5 * t + rng.uniform(0, 2*np.pi))
        theta_component = 7.0 * np.sin(2 * np.pi * 5.5 * t + rng.uniform(0, 2*np.pi))
        alpha_component = 30.0 * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 2*np.pi))
        beta_component  = 10.0 * np.sin(2 * np.pi * 16.0 * t + rng.uniform(0, 2*np.pi))

    signal_uv = delta_component + theta_component + alpha_component + beta_component + pink_noise

    # Optionally inject artifacts
    if add_artifacts:
        # Add 1 high-amplitude blink spike (> 120 uV) to test the +/- 100 uV threshold rejection
        spike_time = int(duration_sec * 0.4 * sampling_rate)
        if spike_time < n_samples:
            spike_len = int(0.25 * sampling_rate)  # 250ms blink
            t_spike = np.linspace(-np.pi, np.pi, min(spike_len, n_samples - spike_time))
            signal_uv[spike_time:spike_time + len(t_spike)] += 140.0 * np.cos(t_spike)

    metadata = {
        "duration_sec": duration_sec,
        "sampling_rate": sampling_rate,
        "patient_type": patient_type.lower(),
        "n_samples": n_samples,
        "channel": "FP1 (Pre-frontal)"
    }

    return signal_uv, metadata
