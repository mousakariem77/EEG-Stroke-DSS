"""
Module 3 Explainable AI (XAI) Endpoints.
Provides SHAP (KernelSHAP Shapley values) and LIME local linear explanations
with plain-English clinical reasoning for medical decision makers.
"""

from fastapi import APIRouter, HTTPException
import numpy as np
from typing import Dict, List
from api.schemas import PredictRequest, SHAPResponse, SHAPContribution, LIMEResponse, LIMEContribution
from api.state import state

router = APIRouter(prefix="/api/explain", tags=["XAI Module 3"])


@router.post("/shap", response_model=SHAPResponse)
def get_shap_explanation(req: PredictRequest):
    """Compute local SHAP feature attributions for a given patient or feature set."""
    if not state.initialized:
        state.initialize()

    if (
        state.raw_df is None
        or state.X is None
        or state.X_norm is None
        or state.feature_names is None
        or state.model is None
        or state.normalizer is None
        or state.shap_explainer is None
    ):
        raise HTTPException(status_code=500, detail="XAI SHAP explainer not initialized.")

    raw_df = state.raw_df
    X = state.X
    X_norm = state.X_norm
    feature_names = state.feature_names
    model = state.model
    normalizer = state.normalizer
    shap_explainer = state.shap_explainer

    if req.patient_id is not None:
        p_id = req.patient_id
        if p_id < 0 or p_id >= len(raw_df):
            raise HTTPException(status_code=404, detail=f"Patient ID {p_id} not found.")
        x_norm = X_norm[p_id:p_id+1]
        raw_vals = X[p_id]
    elif req.features is not None:
        feat_vector = np.zeros((1, len(feature_names)))
        for i, fname in enumerate(feature_names):
            feat_vector[0, i] = float(req.features.get(fname, 0.0))
        x_norm = normalizer.transform(feat_vector)
        raw_vals = feat_vector[0]
    else:
        raise HTTPException(status_code=400, detail="Must provide either patient_id or features dictionary.")

    # Compute SHAP
    shap_vals = shap_explainer.shap_values(x_norm, nsamples=80)
    shap_vals = np.array(shap_vals)
    if shap_vals.ndim == 3:
        vals = shap_vals[1, 0, :] if shap_vals.shape[0] == 2 else shap_vals[0, :, 1]
    elif shap_vals.ndim == 2:
        vals = shap_vals[0]
    else:
        vals = shap_vals

    proba = float(model.predict_proba(x_norm)[0, 1])
    abs_sum = float(np.sum(np.abs(vals))) + 1e-9

    contributions = []
    for i, fname in enumerate(feature_names):
        sv = float(vals[i])
        raw_v = float(raw_vals[i])
        direction = "risk_increasing" if sv > 0 else "protective"
        impact_pct = float(round((abs(sv) / abs_sum) * 100.0, 1))

        contributions.append(SHAPContribution(
            feature=fname,
            feature_value=round(raw_v, 3),
            shap_value=round(sv, 4),
            direction=direction,
            relative_impact_pct=impact_pct
        ))

    # Sort descending by absolute SHAP impact
    contributions.sort(key=lambda x: abs(x.shap_value), reverse=True)

    # Formulate plain-English clinical summary
    top_pos = [c.feature for c in contributions if c.direction == "risk_increasing"][:2]
    top_neg = [c.feature for c in contributions if c.direction == "protective"][:2]

    reasoning_parts = []
    if top_pos:
        reasoning_parts.append(f"Elevated risk is primarily driven by abnormal electro-physiological slowing in: {', '.join(top_pos)}.")
    if top_neg:
        reasoning_parts.append(f"Preserved frequency patterns in: {', '.join(top_neg)} provide moderate protective stability.")
    if not reasoning_parts:
        reasoning_parts.append("Spectral band powers align closely with average baseline cohort distributions.")

    summary_text = " ".join(reasoning_parts)

    return SHAPResponse(
        base_value=round(state.shap_base_value, 4),
        output_probability=round(proba, 4),
        contributions=contributions,
        clinical_summary=summary_text
    )


@router.post("/lime", response_model=LIMEResponse)
def get_lime_explanation(req: PredictRequest):
    """Compute local LIME linear explanation for a given patient or feature set."""
    if not state.initialized:
        state.initialize()

    if (
        state.raw_df is None
        or state.X_norm is None
        or state.feature_names is None
        or state.model is None
        or state.normalizer is None
        or state.lime_explainer is None
    ):
        raise HTTPException(status_code=500, detail="XAI LIME explainer not initialized.")

    raw_df = state.raw_df
    X_norm = state.X_norm
    feature_names = state.feature_names
    model = state.model
    normalizer = state.normalizer
    lime_explainer = state.lime_explainer

    if req.patient_id is not None:
        p_id = req.patient_id
        if p_id < 0 or p_id >= len(raw_df):
            raise HTTPException(status_code=404, detail=f"Patient ID {p_id} not found.")
        x_sample = X_norm[p_id]
    elif req.features is not None:
        feat_vector = np.zeros((1, len(feature_names)))
        for i, fname in enumerate(feature_names):
            feat_vector[0, i] = float(req.features.get(fname, 0.0))
        x_sample = normalizer.transform(feat_vector)[0]
    else:
        raise HTTPException(status_code=400, detail="Must provide either patient_id or features dictionary.")

    exp = lime_explainer.explain_instance(
        data_row=x_sample,
        predict_fn=model.predict_proba,
        num_features=8,
        num_samples=250
    )

    contributions = []
    for rule, weight in exp.as_list():
        w = float(weight)
        contributions.append(LIMEContribution(
            rule=rule,
            weight=round(w, 4),
            direction="risk_increasing" if w > 0 else "protective"
        ))

    score = float(exp.score) if hasattr(exp, 'score') else 0.85
    intercept = float(exp.intercept[1]) if hasattr(exp, 'intercept') and len(exp.intercept) > 1 else 0.5

    return LIMEResponse(
        score=round(score, 3),
        intercept=round(intercept, 4),
        contributions=contributions
    )
