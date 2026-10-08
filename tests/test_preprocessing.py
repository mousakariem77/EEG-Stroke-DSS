"""
Unit tests for Module 1: Pre-treatment & EEG Signal Processing
"""
import sys
import numpy as np
import pytest

from src.preprocessing.filters import butter_bandpass_filter, notch_filter
from src.preprocessing.artifacts import threshold_amplitude_check, reject_artifact_epochs
from src.preprocessing.epoching import create_epochs
from src.preprocessing.psd import (
    compute_epoch_psd, extract_band_power, extract_eeg_features_from_epochs
)
from src.preprocessing.simulator import generate_synthetic_raw_eeg
from src.preprocessing.pipeline import process_raw_eeg_signal


class TestFilters:
    def test_butter_bandpass(self):
        sr = 128.0
        t = np.linspace(0, 10, int(10 * sr), endpoint=False)
        # 10 Hz signal (in-band) + 50 Hz noise (out-of-band) + 0.1 Hz drift (out-of-band)
        sig_in = 10.0 * np.sin(2 * np.pi * 10.0 * t)
        sig_noise = 20.0 * np.sin(2 * np.pi * 50.0 * t)
        sig_drift = 15.0 * np.sin(2 * np.pi * 0.1 * t)
        raw = sig_in + sig_noise + sig_drift

        filtered = butter_bandpass_filter(raw, lowcut=0.5, highcut=30.0, sampling_rate=sr)

        assert filtered.shape == raw.shape
        # Power of 50 Hz and 0.1 Hz components should be heavily attenuated
        assert np.std(filtered) < np.std(raw)
        # Power of the in-band 10 Hz signal should be largely preserved
        corr = np.corrcoef(filtered[int(sr):-int(sr)], sig_in[int(sr):-int(sr)])[0, 1]
        assert corr > 0.85

    def test_notch_filter(self):
        sr = 128.0
        t = np.linspace(0, 4, int(4 * sr), endpoint=False)
        raw = 5.0 * np.sin(2 * np.pi * 10 * t) + 15.0 * np.sin(2 * np.pi * 50 * t)
        filtered = notch_filter(raw, freq=50.0, sampling_rate=sr)
        assert filtered.shape == raw.shape
        assert np.std(filtered) < np.std(raw)


class TestArtifacts:
    def test_amplitude_check(self):
        clean_epoch = np.random.uniform(-80.0, 80.0, 512)
        noisy_epoch = np.array([120.0] + list(clean_epoch[1:]))

        assert threshold_amplitude_check(clean_epoch, threshold_uv=100.0) is True
        assert threshold_amplitude_check(noisy_epoch, threshold_uv=100.0) is False

    def test_reject_artifact_epochs(self):
        epochs = np.random.uniform(-50.0, 50.0, (10, 512))
        # Contaminate epoch 2 and epoch 6 with spikes > 100 uV
        epochs[2, 50] = 135.0
        epochs[6, 100] = -150.0

        clean, mask, report = reject_artifact_epochs(epochs, threshold_uv=100.0)
        assert len(clean) == 8
        assert mask[2] == False
        assert mask[6] == False
        assert report['clean_epochs'] == 8
        assert report['rejected_epochs'] == 2


class TestEpochingAndPSD:
    def test_create_epochs_and_resolution(self):
        sr = 128.0
        duration_sec = 20.0
        raw = np.random.randn(int(duration_sec * sr))
        epochs = create_epochs(raw, sampling_rate=sr, epoch_duration_sec=4.0)

        # 20 seconds / 4 seconds = 5 epochs
        assert epochs.shape == (5, 512)

        # FFT resolution check
        freqs, psd = compute_epoch_psd(epochs[0], sampling_rate=sr)
        freq_res = freqs[1] - freqs[0]
        assert np.isclose(freq_res, 0.25, atol=1e-3)

    def test_extract_features(self):
        sr = 128.0
        epochs = np.random.uniform(-40, 40, (4, 512))
        features = extract_eeg_features_from_epochs(epochs, sampling_rate=sr)

        expected_keys = [
            'Delta', 'Theta', 'Alpha', 'Beta', 'Total Power',
            'RP Delta', 'RP Theta', 'RP Alpha', 'RP Beta',
            'DTR', 'DAR', 'Epoch'
        ]
        for key in expected_keys:
            assert key in features
            assert isinstance(features[key], (float, int))
            assert features[key] >= 0

        # Relative power properties
        rp_sum = (features['RP Delta'] + features['RP Theta'] +
                  features['RP Alpha'] + features['RP Beta'])
        assert np.isclose(rp_sum, 1.0, atol=0.05)


class TestFullRawPipeline:
    def test_pipeline_control_vs_stroke(self):
        sr = 128.0
        # Generate synthetic signals
        stroke_signal, _ = generate_synthetic_raw_eeg(
            duration_sec=32.0, sampling_rate=sr, patient_type='stroke', random_state=42
        )
        control_signal, _ = generate_synthetic_raw_eeg(
            duration_sec=32.0, sampling_rate=sr, patient_type='control', random_state=42
        )

        stroke_feat, stroke_rep = process_raw_eeg_signal(stroke_signal, sampling_rate=sr)
        control_feat, control_rep = process_raw_eeg_signal(control_signal, sampling_rate=sr)

        assert stroke_rep['clean_epochs'] > 0
        assert control_rep['clean_epochs'] > 0

        # Stroke physiological profile: higher DTR and RP Delta than Control
        assert stroke_feat['DTR'] > control_feat['DTR']
        assert stroke_feat['RP Delta'] > control_feat['RP Delta']
        # Control physiological profile: higher RP Alpha than Stroke
        assert control_feat['RP Alpha'] > stroke_feat['RP Alpha']
