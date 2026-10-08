"""
Module 4 Decision Support System (DSS) & Clinical Action Center.
Manages clinical test recommendations, physician actions (Confirm/Cancel),
audit logging for human-in-the-loop accountability, and PDF generation.
"""

from fastapi import APIRouter, HTTPException, Response, Query
from typing import List, Optional, Dict
from pydantic import BaseModel
import numpy as np
from api.schemas import (
    ClinicalTestRecommendation,
    PhysicianActionRequest,
    AuditLogItem,
    SessionStateResponse
)
from api.state import state
from src.dss.recommendation import DecisionStatus
from src.reports.pdf_generator import generate_report

router = APIRouter(prefix="/api/dss", tags=["DSS Module 4"])


class PDFDownloadRequest(BaseModel):
    patient_id: int
    session_id: Optional[str] = None
    stroke_probability: Optional[float] = None
    risk_level: Optional[str] = None
    prediction: Optional[str] = None
    features: Optional[Dict[str, float]] = None


@router.get("/recommendations", response_model=List[ClinicalTestRecommendation])
def get_recommendations(
    stroke_probability: Optional[float] = Query(None, ge=0.0, le=1.0),
    stroke_prob: Optional[float] = Query(None, ge=0.0, le=1.0),
    session_id: Optional[str] = None,
    patient_id: int = 0
):
    """Retrieve actionable diagnostic tests based on AI acute stroke probability."""
    if not state.initialized:
        state.initialize()

    prob = stroke_probability if stroke_probability is not None else (stroke_prob if stroke_prob is not None else 0.5)
    pred_class = "Stroke" if prob >= 0.50 else "Control"
    recs = state.recommendation_engine.get_recommendations(prob, pred_class)
    tests = recs.get('recommended_tests', [])

    session = None
    if session_id:
        session = state.get_or_create_session(session_id, patient_id)

    response_tests = []
    for t in tests:
        t_id = t.get('name', '').lower().replace(' ', '_')
        status = "PENDING"
        notes = ""
        if session:
            match = session.decisions.get(t.get('name', '')) or session.decisions.get(t_id)
            if match:
                st = match.get('status', 'PENDING')
                status = getattr(st, 'value', str(st))
                notes = match.get('physician_notes', match.get('notes', ''))

        response_tests.append(ClinicalTestRecommendation(
            id=t_id,
            name=t.get('full_name', t.get('name', '')),
            urgency=recs.get('urgency', 'ROUTINE'),
            target_time="Within 3 hours" if recs.get('urgency') == 'CRITICAL' else "Within 24 hours",
            rationale=t.get('description', ''),
            key_findings="Lesion detection & penumbra" if 'MRI' in t.get('name') else "Systemic & vascular markers",
            contraindications="Pacemaker, metal implants" if 'MRI' in t.get('name') else "None",
            category=t.get('category', 'Diagnostic').capitalize(),
            status=status,
            notes=notes
        ))

    return response_tests


@router.post("/action", response_model=SessionStateResponse)
def record_physician_action(req: PhysicianActionRequest):
    """
    Record physician Human-in-the-Loop decision (CONFIRMED or CANCELLED) with audit trail.
    Ensures that AI only recommends, while the certified medical professional retains final authority.
    """
    if not state.initialized:
        state.initialize()

    session = state.get_or_create_session(req.session_id, patient_id=0)

    action_str = req.action.upper()
    if action_str not in ("CONFIRMED", "CANCELLED"):
        raise HTTPException(status_code=400, detail=f"Invalid action '{req.action}'. Must be 'CONFIRMED' or 'CANCELLED'.")

    # Find matching test in session.decisions (case-insensitive or normalized)
    target_key = None
    for k in session.decisions.keys():
        if k.lower() == req.test_id.lower() or k.lower().replace(' ', '_') == req.test_id.lower():
            target_key = k
            break
    if not target_key:
        target_key = list(session.decisions.keys())[0] if session.decisions else req.test_id

    if action_str == "CONFIRMED":
        session.confirm_test(target_key, physician_notes=req.notes)
    else:
        session.cancel_test(target_key, reason=req.notes)

    state.save_sessions()

    return get_session_state(req.session_id)


@router.get("/session/{session_id}", response_model=SessionStateResponse)
def get_session_state(session_id: str):
    """Get current physician review session state, decision counts, and audit trail."""
    if session_id not in state.sessions:
        return SessionStateResponse(
            session_id=session_id,
            patient_id=0,
            recommendations=[],
            audit_log=[],
            confirmed_count=0,
            cancelled_count=0,
            pending_count=0
        )

    session = state.sessions[session_id]
    summary = session.get_summary()

    audit_items = [
        AuditLogItem(
            id=f"audit-{i}",
            timestamp=item['timestamp'],
            test_id=item.get('test_name', item.get('test_id', '')),
            test_name=session.decisions.get(item.get('test_name', ''), {}).get('full_name', item.get('test_name', '')),
            action=item['action'],
            physician_id="Dr. Attending Neurologist",
            notes=item.get('notes', item.get('reason', ''))
        )
        for i, item in enumerate(session.audit_log)
    ]

    return SessionStateResponse(
        session_id=session_id,
        patient_id=session.patient_id,
        recommendations=[],
        audit_log=audit_items,
        confirmed_count=summary['confirmed'],
        cancelled_count=summary['cancelled'],
        pending_count=summary['pending']
    )


@router.post("/pdf")
def generate_pdf_report(req: PDFDownloadRequest):
    """Generate and stream signed clinical PDF report with full decision audit log."""
    if not state.initialized:
        state.initialize()

    if (
        state.raw_df is None
        or state.X is None
        or state.feature_names is None
        or state.normalizer is None
        or state.model is None
    ):
        raise HTTPException(status_code=500, detail="Clinical data and models not initialized.")

    raw_df = state.raw_df
    X = state.X
    feature_names = state.feature_names
    normalizer = state.normalizer
    model = state.model

    p_id = req.patient_id
    if p_id < 0 or p_id >= len(raw_df):
        p_id = 0

    row = raw_df.iloc[p_id]
    true_label = str(row.get('Participant', 'Unknown')).capitalize()

    features = req.features or {f: float(X[p_id, i]) for i, f in enumerate(feature_names)}
    feat_vector = np.zeros((1, len(feature_names)))
    for i, fname in enumerate(feature_names):
        feat_vector[0, i] = float(features.get(fname, 0.0))
    norm_features = {f: float(normalizer.transform(feat_vector)[0, i]) for i, f in enumerate(feature_names)}

    prob = req.stroke_probability
    if prob is None:
        prob = float(model.predict_proba(normalizer.transform(feat_vector))[0, 1])

    risk = req.risk_level or ("high" if prob >= 0.7 else "medium" if prob >= 0.4 else "low")
    pred = req.prediction or ("Stroke" if prob >= 0.5 else "Control")

    recs = state.recommendation_engine.get_recommendations(prob, pred)

    physician_decisions = None
    if req.session_id and req.session_id in state.sessions:
        session = state.sessions[req.session_id]
        physician_decisions = session.get_summary()

    pdf_bytes = generate_report(
        patient_id=p_id,
        prediction=pred,
        stroke_probability=prob,
        risk_level=risk.lower(),
        features=features,
        normalized_features=norm_features,
        recommendations=recs,
        lime_contributions=[],
        true_label=true_label,
        physician_decisions=physician_decisions
    )

    filename = f"Stroke_DSS_Report_P{p_id + 1:03d}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Access-Control-Expose-Headers": "Content-Disposition"
        }
    )
