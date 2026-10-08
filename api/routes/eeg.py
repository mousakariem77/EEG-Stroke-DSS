"""
Module 1 Signal Processing & Simulation Endpoints.
Simulates continuous raw 1-channel FP1 EEG signals and executes zero-phase Butterworth filtering,
50Hz notch filtering, amplitude artifact thresholding (+-100 uV), 4-second epoching (0.25 Hz FFT resolution),
and adapted PSD band power extraction.
"""

from fastapi import APIRouter, HTTPException
import numpy as np
from typing import List
from api.schemas import (
    RawEEGSimulateRequest,
    RawEEGSimulateResponse,
    RawEEGProcessRequest,
    RawEEGProcessResponse
)
from api.state import state
from src.preprocessing.simulator import generate_synthetic_raw_eeg
from src.preprocessing.pipeline import process_raw_eeg_signal

router = APIRouter(prefix="/api/eeg", tags=["EEG Module 1"])


@router.post("/simulate", response_model=RawEEGSimulateResponse)
def simulate_eeg_waveform(req: RawEEGSimulateRequest):
    """Simulate realistic 1-channel FP1 resting-state EEG (acute stroke vs healthy control)."""
    raw_sig, _ = generate_synthetic_raw_eeg(
        duration_sec=req.duration_sec,
        sampling_rate=req.sampling_rate,
        patient_type=req.pattern,
        random_state=req.seed
    )
    time_ax = np.arange(len(raw_sig)) / req.sampling_rate

    # Return downsampled if very dense, or full for 8s (8s * 128Hz = 1024 points, very fast for Canvas)
    return RawEEGSimulateResponse(
        signal=[float(round(v, 2)) for v in raw_sig],
        time_axis=[float(round(t, 4)) for t in time_ax],
        sampling_rate=req.sampling_rate,
        duration_sec=req.duration_sec,
        pattern=req.pattern
    )


@router.post("/process", response_model=RawEEGProcessResponse)
def process_eeg_signal(req: RawEEGProcessRequest):
    """
    Execute complete Module 1 Preprocessing:
    - Zero-phase Butterworth Bandpass (0.5 - 30.0 Hz)
    - Notch Filter (50.0 Hz)
    - Artifact Rejection (+-100 uV)
    - 4.0-second Epoching (exact 0.25 Hz FFT resolution)
    - Adapted PSD Band Power Integration
    """
    if not state.initialized:
        state.initialize()

    raw_arr = np.array(req.signal, dtype=np.float64)
    if len(raw_arr) < int(req.sampling_rate * req.epoch_duration_sec):
        raise HTTPException(
            status_code=400,
            detail=f"Signal length ({len(raw_arr)} samples) must be at least one 4.0s epoch ({int(req.sampling_rate * req.epoch_duration_sec)} samples)."
        )

    features, report = process_raw_eeg_signal(
        raw_signal=raw_arr,
        sampling_rate=req.sampling_rate,
        epoch_duration_sec=req.epoch_duration_sec
    )

    from src.preprocessing.filters import butter_bandpass_filter, notch_filter
    notched = notch_filter(raw_arr, freq=50.0, sampling_rate=req.sampling_rate)
    filtered_sig = butter_bandpass_filter(notched, lowcut=0.5, highcut=30.0, sampling_rate=req.sampling_rate)
    time_ax = np.arange(len(filtered_sig)) / req.sampling_rate

    # Normalize extracted features
    feat_vector = np.zeros((1, len(state.feature_names)))
    for i, fname in enumerate(state.feature_names):
        feat_vector[0, i] = float(features.get(fname, 0.0))
    norm_features = {f: float(state.normalizer.transform(feat_vector)[0, i]) for i, f in enumerate(state.feature_names)}

    return RawEEGProcessResponse(
        filtered_signal=[float(round(v, 2)) for v in filtered_sig],
        time_axis=[float(round(t, 4)) for t in time_ax],
        total_epochs=int(report['total_epochs']),
        clean_epochs=int(report['clean_epochs']),
        rejected_epochs=int(report['rejected_epochs']),
        frequency_resolution_hz=float(1.0 / req.epoch_duration_sec),
        features={k: float(round(v, 4)) for k, v in features.items()},
        normalized_features={k: float(round(v, 4)) for k, v in norm_features.items()}
    )
