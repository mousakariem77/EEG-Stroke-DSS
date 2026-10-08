"""
Module 1: Pre-treatment (EEG Signal Preprocessing & Feature Extraction)
"""
from src.preprocessing.filters import butter_bandpass_filter, notch_filter
from src.preprocessing.artifacts import threshold_amplitude_check, reject_artifact_epochs
from src.preprocessing.epoching import create_epochs
from src.preprocessing.psd import compute_epoch_psd, extract_band_power, extract_eeg_features_from_epochs
from src.preprocessing.simulator import generate_synthetic_raw_eeg
from src.preprocessing.pipeline import process_raw_eeg_signal

__all__ = [
    'butter_bandpass_filter',
    'notch_filter',
    'threshold_amplitude_check',
    'reject_artifact_epochs',
    'create_epochs',
    'compute_epoch_psd',
    'extract_band_power',
    'extract_eeg_features_from_epochs',
    'generate_synthetic_raw_eeg',
    'process_raw_eeg_signal'
]
