"""
Module 2 Ensemble ESN Prediction Endpoints.
Feeds extracted/normalized EEG features into the 7-estimator Echo State Network ensemble.
Calculates acute stroke probability, clinical risk categorization, and plain-English alerts.
"""

from fastapi import APIRouter, HTTPException
import numpy as np
from typing import Dict
from api.schemas import PredictRequest, PredictResponse
from api.state import state

router = APIRouter(prefix="/api/predict", tags=["Prediction Module 2"])


@router.post("", response_model=PredictResponse)
def predict_stroke_risk(req: PredictRequest):
    """
    Run Ensemble ESN inference (7 estimators, 200 reservoir neurons, SR=0.95).
    Accepts either an existing patient ID or custom feature dictionary.
    """
    if not state.initialized:
        state.initialize()

    if (
        state.raw_df is None
        or state.X is None
        or state.X_norm is None
        or state.feature_names is None
        or state.model is None
        or state.normalizer is None
    ):
        raise HTTPException(status_code=500, detail="Prediction model not initialized.")

    raw_df = state.raw_df
    X = state.X
    X_norm = state.X_norm
    feature_names = state.feature_names
    model = state.model
    normalizer = state.normalizer

    if req.patient_id is not None:
        p_id = req.patient_id
        if p_id < 0 or p_id >= len(raw_df):
            raise HTTPException(status_code=404, detail=f"Patient ID {p_id} not found.")
        x_norm = X_norm[p_id:p_id+1]
        raw_feat_dict = {f: float(X[p_id, i]) for i, f in enumerate(feature_names)}
    elif req.features is not None:
        # Build vector matching feature_names
        feat_vector = np.zeros((1, len(feature_names)))
        raw_feat_dict = {}
        for i, fname in enumerate(feature_names):
            val = float(req.features.get(fname, 0.0))
            feat_vector[0, i] = val
            raw_feat_dict[fname] = val
        x_norm = normalizer.transform(feat_vector)
    else:
        raise HTTPException(status_code=400, detail="Must provide either patient_id or features dictionary.")

    # Model inference
    proba = model.predict_proba(x_norm)[0]
    stroke_prob = float(proba[1])
    control_prob = float(proba[0])

    if stroke_prob >= 0.70:
        risk_level = "High"
        pred = "Acute Stroke"
        alert = "CRITICAL: High acute stroke probability detected. Immediate neuroimaging (Brain MRI/CT) is strongly indicated."
    elif stroke_prob >= 0.40:
        risk_level = "Medium"
        pred = "Acute Stroke" if stroke_prob >= 0.50 else "Healthy Control"
        alert = "MODERATE: Equivocal neuro-electric findings. Secondary vascular screening and close monitoring recommended."
    else:
        risk_level = "Low"
        pred = "Healthy Control"
        alert = "LOW RISK: Resting-state EEG spectral dynamics remain within healthy age-matched baseline limits."

    dar = float(raw_feat_dict.get('DAR', 0.0))
    dar_status = "Elevated (>3.70 - Ischemia Marker)" if dar > 3.70 else "Normal (<=3.70 - Stable)"

    return PredictResponse(
        prediction=pred,
        stroke_probability=round(stroke_prob, 4),
        control_probability=round(control_prob, 4),
        risk_level=risk_level,
        confidence=round(max(stroke_prob, control_prob), 4),
        is_acute=(stroke_prob >= 0.50),
        clinical_alert=alert,
        dar_value=round(dar, 3),
        dar_reference_status=dar_status,
        features_used={k: round(v, 4) for k, v in raw_feat_dict.items()}
    )
