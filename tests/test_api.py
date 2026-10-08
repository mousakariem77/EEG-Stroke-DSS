"""
Unit and integration tests for FastAPI REST API endpoints.
Validates all endpoints across data loading, EEG processing, model inference, XAI, and clinical DSS.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_check():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "Ensemble Echo State Network" in data["model"]


def test_get_patients():
    res = client.get("/api/patients")
    assert res.status_code == 200
    patients = res.json()
    assert len(patients) == 38
    assert patients[0]["participant_id"] == "P_001"
    assert "dar" in patients[0]


def test_get_patient_detail():
    res = client.get("/api/patients/0")
    assert res.status_code == 200
    detail = res.json()
    assert detail["id"] == 0
    assert "Delta" in detail["features"]
    assert "DAR" in detail["features"]


def test_eeg_simulation_and_processing():
    # Simulate
    sim_res = client.post("/api/eeg/simulate", json={"pattern": "stroke", "duration_sec": 8.0, "sampling_rate": 128.0})
    assert sim_res.status_code == 200
    sim_data = sim_res.json()
    assert len(sim_data["signal"]) == 8 * 128

    # Process
    proc_res = client.post("/api/eeg/process", json={"signal": sim_data["signal"], "sampling_rate": 128.0})
    assert proc_res.status_code == 200
    proc_data = proc_res.json()
    assert proc_data["clean_epochs"] >= 1
    assert "DAR" in proc_data["features"]
    assert proc_data["frequency_resolution_hz"] == 0.25


def test_model_prediction():
    res = client.post("/api/predict", json={"patient_id": 0})
    assert res.status_code == 200
    data = res.json()
    assert "prediction" in data
    assert 0.0 <= data["stroke_probability"] <= 1.0
    assert data["risk_level"] in ("High", "Medium", "Low")
    assert "dar_value" in data


def test_xai_shap_and_lime():
    # SHAP
    shap_res = client.post("/api/explain/shap", json={"patient_id": 0})
    assert shap_res.status_code == 200
    shap_data = shap_res.json()
    assert len(shap_data["contributions"]) > 0
    assert "clinical_summary" in shap_data

    # LIME
    lime_res = client.post("/api/explain/lime", json={"patient_id": 0})
    assert lime_res.status_code == 200
    lime_data = lime_res.json()
    assert len(lime_data["contributions"]) > 0


def test_dss_recommendations_and_actions():
    sess_id = "test_unit_session"
    rec_res = client.get(f"/api/dss/recommendations?stroke_probability=0.85&session_id={sess_id}")
    assert rec_res.status_code == 200
    recs = rec_res.json()
    assert len(recs) >= 3

    # Record action
    act_res = client.post("/api/dss/action", json={
        "session_id": sess_id,
        "test_id": recs[0]["id"],
        "action": "CONFIRMED",
        "notes": "Urgent verification confirmed."
    })
    assert act_res.status_code == 200
    session_data = act_res.json()
    assert session_data["confirmed_count"] >= 1


def test_pdf_generation():
    res = client.post("/api/dss/pdf", json={"patient_id": 0, "session_id": "test_unit_session"})
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"
    assert len(res.content) > 1000


def test_frontend_serving():
    res = client.get("/")
    assert res.status_code == 200
    assert "Explainable EEG Stroke" in res.text
