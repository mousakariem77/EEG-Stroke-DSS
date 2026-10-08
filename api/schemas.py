"""
Pydantic data schemas for FastAPI REST API endpoints.
All schemas adhere to strict clinical data typing and security best practices.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class PatientSummary(BaseModel):
    id: int
    participant_id: str
    diagnosis: str
    age: str
    gender: str
    dar: float
    dtr: float
    total_power: float
    is_stroke: bool
    handedness: Optional[str] = "Right"
    education: Optional[int] = None
    epoch: Optional[int] = None


class PatientDetail(BaseModel):
    id: int
    participant_id: str
    diagnosis: str
    age: str
    gender: str
    features: Dict[str, float]
    normalized_features: Dict[str, float]
    clinical_profile: Optional[Dict[str, Any]] = None


class RawEEGSimulateRequest(BaseModel):
    pattern: str = Field(default="stroke", description="'stroke' or 'control'")
    duration_sec: float = Field(default=8.0, ge=4.0, le=60.0)
    sampling_rate: float = Field(default=128.0, ge=64.0, le=512.0)
    seed: Optional[int] = None


class RawEEGSimulateResponse(BaseModel):
    signal: List[float]
    time_axis: List[float]
    sampling_rate: float
    duration_sec: float
    pattern: str


class RawEEGProcessRequest(BaseModel):
    signal: List[float]
    sampling_rate: float = 128.0
    epoch_duration_sec: float = 4.0
    patient_id: Optional[int] = 0


class RawEEGProcessResponse(BaseModel):
    filtered_signal: List[float]
    time_axis: List[float]
    total_epochs: int
    clean_epochs: int
    rejected_epochs: int
    frequency_resolution_hz: float
    features: Dict[str, float]
    normalized_features: Dict[str, float]


class PredictRequest(BaseModel):
    patient_id: Optional[int] = None
    features: Optional[Dict[str, float]] = None


class PredictResponse(BaseModel):
    prediction: str
    stroke_probability: float
    control_probability: float
    risk_level: str  # "High", "Medium", "Low"
    confidence: float
    is_acute: bool
    clinical_alert: str
    dar_value: float
    dar_reference_status: str  # "Elevated (>3.7)" or "Normal (<=3.7)"
    features_used: Dict[str, float]


class SHAPContribution(BaseModel):
    feature: str
    feature_value: float
    shap_value: float
    direction: str  # "risk_increasing" or "protective"
    relative_impact_pct: float


class SHAPResponse(BaseModel):
    base_value: float
    output_probability: float
    contributions: List[SHAPContribution]
    clinical_summary: str


class LIMEContribution(BaseModel):
    rule: str
    weight: float
    direction: str


class LIMEResponse(BaseModel):
    score: float
    intercept: float
    contributions: List[LIMEContribution]


class ClinicalTestRecommendation(BaseModel):
    id: str
    name: str
    urgency: str
    target_time: str
    rationale: str
    key_findings: str
    contraindications: str
    category: str
    status: str  # "PENDING", "CONFIRMED", "CANCELLED"
    notes: Optional[str] = ""


class PhysicianActionRequest(BaseModel):
    session_id: str
    test_id: str
    action: str  # "CONFIRMED" or "CANCELLED"
    notes: str = ""
    physician_id: str = "Dr. Attending Neurologist"


class AuditLogItem(BaseModel):
    id: str
    timestamp: str
    test_id: str
    test_name: str
    action: str
    physician_id: str
    notes: str


class SessionStateResponse(BaseModel):
    session_id: str
    patient_id: int
    recommendations: List[ClinicalTestRecommendation]
    audit_log: List[AuditLogItem]
    confirmed_count: int
    cancelled_count: int
    pending_count: int
