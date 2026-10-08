"""
Unit tests for Module 4: DSS Recommendation Engine & Physician Action Control
"""
import sys
import pytest

from src.dss.recommendation import (
    RecommendationEngine, DecisionStatus, PhysicianDecisionSession
)
from src.reports.pdf_generator import generate_report


class TestPhysicianActions:
    def setup_method(self):
        self.engine = RecommendationEngine()
        self.rec = self.engine.get_recommendations(0.92, 'Stroke')
        self.session = PhysicianDecisionSession(patient_id=1, recommendations=self.rec)

    def test_initial_session_status(self):
        summary = self.session.get_summary()
        assert summary['total_recommended'] == 6
        assert summary['confirmed'] == 0
        assert summary['cancelled'] == 0
        assert summary['pending'] == 6

    def test_confirm_test(self):
        self.session.confirm_test('MRI', physician_notes="Urgent brain MRI ordered to locate infarct.")
        summary = self.session.get_summary()
        assert summary['confirmed'] == 1
        assert summary['pending'] == 5
        assert self.session.decisions['MRI']['status'] == DecisionStatus.CONFIRMED
        assert "Urgent brain MRI" in self.session.decisions['MRI']['physician_notes']

    def test_cancel_test(self):
        self.session.cancel_test('Lipid Profile', reason="Patient completed fasting lipid panel 48 hours ago.")
        summary = self.session.get_summary()
        assert summary['cancelled'] == 1
        assert self.session.decisions['Lipid Profile']['status'] == DecisionStatus.CANCELLED
        assert "fasting lipid panel" in self.session.decisions['Lipid Profile']['reason']

    def test_audit_log_tracking(self):
        self.session.confirm_test('MRI', physician_notes="Approved")
        self.session.cancel_test('D-Dimer', reason="Low clinical suspicion of pulmonary embolism")
        
        log = self.session.audit_log
        assert len(log) == 2
        assert log[0]['action'] == 'CONFIRM'
        assert log[0]['test_name'] == 'MRI'
        assert log[1]['action'] == 'CANCEL'
        assert log[1]['test_name'] == 'D-Dimer'

    def test_pdf_report_with_physician_decisions(self):
        self.session.confirm_test('MRI', physician_notes="Stat MRI scan")
        self.session.cancel_test('CBC', reason="Recent complete blood count available in EMR")
        summary = self.session.get_summary()

        dummy_features = {'Delta': 12.0, 'Theta': 8.0, 'Alpha': 4.0, 'Beta': 6.0}
        pdf_bytes = generate_report(
            patient_id=1,
            prediction='Stroke',
            stroke_probability=0.92,
            risk_level='high',
            features=dummy_features,
            normalized_features=dummy_features,
            recommendations=self.rec,
            physician_decisions=summary
        )

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 1000
        # Verify PDF header
        assert pdf_bytes.startswith(b'%PDF')
