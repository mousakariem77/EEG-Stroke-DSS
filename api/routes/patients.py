"""
Patient data retrieval endpoints.
Provides demographic and clinical EEG feature summaries from the EPoC clinical study.
"""

from fastapi import APIRouter, HTTPException
from typing import List
from api.state import state
from api.schemas import PatientSummary, PatientDetail
from src.data.loader import LABEL_MAP_INV

router = APIRouter(prefix="/api/patients", tags=["Patients"])


@router.get("", response_model=List[PatientSummary])
def get_all_patients():
    """Retrieve list of all 38 clinical study patients with key EEG metrics."""
    if not state.initialized:
        state.initialize()

    if state.raw_df is None:
        raise HTTPException(status_code=500, detail="Patient data not initialized.")

    results = []
    df = state.raw_df

    for idx, row in df.iterrows():
        p_id = int(idx)
        raw_label = row.get('Participant', 'control')
        diag = str(raw_label).lower()
        is_stroke = (diag == 'stroke')

        results.append(PatientSummary(
            id=p_id,
            participant_id=f"P_{p_id + 1:03d}",
            diagnosis="Acute Stroke" if is_stroke else "Healthy Control",
            age=str(row.get('Age', 'N/A')),
            gender=str(row.get('Gender', 'Unknown')).capitalize(),
            dar=float(round(row.get('DAR', 0.0), 3)),
            dtr=float(round(row.get('DTR', 0.0), 3)),
            total_power=float(round(row.get('Total Power', 0.0), 3)),
            is_stroke=is_stroke,
            handedness=str(row.get('Handedness', 'N/A')).capitalize(),
            education=int(row.get('Years of education', 0)) if str(row.get('Years of education', '')).isdigit() else None,
            epoch=int(row.get('Epoch', 0)) if str(row.get('Epoch', '')).isdigit() else None
        ))

    return results


@router.get("/{patient_id}", response_model=PatientDetail)
def get_patient_detail(patient_id: int):
    """Retrieve full clinical features and normalized representations for a patient."""
    import pandas as pd
    import numpy as np

    if not state.initialized:
        state.initialize()

    if state.raw_df is None or state.X is None or state.X_norm is None or state.feature_names is None:
        raise HTTPException(status_code=500, detail="Patient data not initialized.")

    raw_df = state.raw_df
    X = state.X
    X_norm = state.X_norm
    feature_names = state.feature_names

    if patient_id < 0 or patient_id >= len(raw_df):
        raise HTTPException(status_code=404, detail=f"Patient ID {patient_id} not found.")

    row = raw_df.iloc[patient_id]
    features_dict = {f: float(round(X[patient_id, i], 4)) for i, f in enumerate(feature_names)}
    norm_dict = {f: float(round(X_norm[patient_id, i], 4)) for i, f in enumerate(feature_names)}

    diag = str(row.get('Participant', 'control')).lower()

    # Build clean clinical profile mapping
    clinical_data = {}
    for col, val in row.items():
        clean_col = str(col).strip()
        if pd.isna(val):
            clinical_data[clean_col] = "N/A"
        elif isinstance(val, (int, np.integer)):
            clinical_data[clean_col] = int(val)
        elif isinstance(val, (float, np.floating)):
            clinical_data[clean_col] = float(round(val, 4))
        else:
            clinical_data[clean_col] = str(val)

    return PatientDetail(
        id=patient_id,
        participant_id=f"P_{patient_id + 1:03d}",
        diagnosis="Acute Stroke" if diag == 'stroke' else "Healthy Control",
        age=str(row.get('Age', 'N/A')),
        gender=str(row.get('Gender', 'Unknown')).capitalize(),
        features=features_dict,
        normalized_features=norm_dict,
        clinical_profile=clinical_data
    )
