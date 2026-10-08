"""
Module 4 (Decision Support System): Recommendation Engine & Physician Action Control

Bouazizi & Ltifi (2024), Section 4.6 & Fig. 8:
"Based on the case of this patient, the recommendation to the physician suggests
specific diagnostic tests, such as Magnetic Resonance Imaging (MRI), Complete Blood
Count (CBC) test, Basic Metabolic Panel (BMP) test, Coagulation Profile test,
Lipid Profile test, and D-Dimer test... The physician can choose between these
recommendations, confirm or cancel them."

Includes human-in-the-loop audit session:
- Diagnostic test suggestions based on risk classification.
- Physician review actions: Confirm, Cancel, Notes.
- Immutable audit log for clinical governance.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np


# Diagnostic tests and clinical metadata (Section 4.6)
DIAGNOSTIC_TESTS = {
    'MRI': {
        'full_name': 'Magnetic Resonance Imaging (MRI)',
        'description': 'Brain neuroimaging to detect stroke lesion location, vascular territory, and infarct extent',
        'priority': 1,
        'category': 'imaging'
    },
    'CBC': {
        'full_name': 'Complete Blood Count (CBC)',
        'description': 'Screening for systemic infection, severe anemia, and thrombocytosis/thrombocytopenia',
        'priority': 2,
        'category': 'blood'
    },
    'BMP': {
        'full_name': 'Basic Metabolic Panel (BMP)',
        'description': 'Electrolytes, fasting blood glucose, and renal function assessment (BUN/Creatinine)',
        'priority': 3,
        'category': 'blood'
    },
    'Coagulation Profile': {
        'full_name': 'Coagulation Profile (PT/INR, aPTT)',
        'description': 'Assessment of intrinsic/extrinsic coagulation pathways to guide acute thrombolysis or anticoagulation',
        'priority': 2,
        'category': 'blood'
    },
    'Lipid Profile': {
        'full_name': 'Lipid Profile',
        'description': 'Total cholesterol, HDL, LDL, and triglyceride levels for atherothrombotic risk assessment',
        'priority': 4,
        'category': 'blood'
    },
    'D-Dimer': {
        'full_name': 'D-Dimer Test',
        'description': 'Fibrin degradation product assay to detect active intravascular thrombosis or thromboembolism',
        'priority': 3,
        'category': 'blood'
    }
}


class DecisionStatus:
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    CANCELLED = "CANCELLED"


class RecommendationEngine:
    """
    Rule-based clinical recommendation engine.
    
    Categorizes stroke risk and suggests evidence-based diagnostic tests.
    """
    
    def __init__(
        self,
        high_risk_threshold: float = 0.7,
        medium_risk_threshold: float = 0.4
    ):
        self.high_risk_threshold = high_risk_threshold
        self.medium_risk_threshold = medium_risk_threshold
    
    def classify_risk(self, stroke_probability: float) -> str:
        """
        Classify clinical risk level based on stroke probability.
        
        Returns: 'high', 'medium', or 'low'
        """
        if stroke_probability >= self.high_risk_threshold:
            return 'high'
        elif stroke_probability >= self.medium_risk_threshold:
            return 'medium'
        else:
            return 'low'
    
    def get_recommendations(
        self,
        stroke_probability: float,
        predicted_class: str,
        feature_contributions: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Generate diagnostic test recommendations.
        
        Args:
            stroke_probability: Probability of stroke [0.0, 1.0] from E-ESN.
            predicted_class: 'Stroke' or 'Control'.
            feature_contributions: Optional SHAP/LIME feature impact list.
        
        Returns:
            Dictionary containing risk level, tests, urgency, action, and reasoning.
        """
        risk_level = self.classify_risk(stroke_probability)
        
        # Determine candidate tests by risk level
        if risk_level == 'high':
            test_names = list(DIAGNOSTIC_TESTS.keys())  # All 6 tests
            urgency = 'URGENT'
            action = 'Immediate comprehensive acute stroke workup recommended'
        elif risk_level == 'medium':
            test_names = ['MRI', 'CBC', 'BMP', 'Coagulation Profile']
            urgency = 'PRIORITY'
            action = 'Secondary evaluation recommended within 24 hours'
        else:
            test_names = ['CBC', 'BMP']  # Baseline routine screening
            urgency = 'ROUTINE'
            action = 'Routine clinical follow-up and monitoring recommended'
        
        # Build test details
        recommended_tests = []
        for name in test_names:
            test = DIAGNOSTIC_TESTS[name].copy()
            test['name'] = name
            test['initial_status'] = DecisionStatus.PENDING
            recommended_tests.append(test)
        
        recommended_tests.sort(key=lambda t: t['priority'])
        
        reasoning = self._build_reasoning(
            stroke_probability, predicted_class, risk_level, feature_contributions
        )
        
        return {
            'risk_level': risk_level,
            'urgency': urgency,
            'stroke_probability': float(stroke_probability),
            'predicted_class': predicted_class,
            'recommended_tests': recommended_tests,
            'action': action,
            'reasoning': reasoning,
            'n_tests': len(recommended_tests)
        }
    
    def _build_reasoning(
        self,
        stroke_probability: float,
        predicted_class: str,
        risk_level: str,
        feature_contributions: Optional[List]
    ) -> List[str]:
        """Construct human-readable physiological rationale."""
        reasons = [
            f"EEG E-ESN prediction: {predicted_class} (Probability: {stroke_probability:.1%})",
            f"Stratified risk tier: {risk_level.upper()}"
        ]
        
        if feature_contributions:
            top_features = sorted(
                feature_contributions, 
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:3]
            
            for feat_name, contrib in top_features:
                direction = "elevates" if contrib > 0 else "reduces"
                reasons.append(
                    f"Biomarker influence: {feat_name} ({direction} acute stroke probability, impact: {abs(contrib):.3f})"
                )
        
        return reasons
    
    def format_report(self, recommendation: Dict[str, Any]) -> str:
        """Format recommendation as plain text clinical summary."""
        r = recommendation
        lines = [
            "=" * 60,
            "DIAGNOSTIC RECOMMENDATION REPORT",
            "=" * 60,
            f"Predicted Diagnosis: {r['predicted_class']}",
            f"Stroke Risk Probability: {r['stroke_probability']:.1%}",
            f"Clinical Risk Tier: {r['risk_level'].upper()}",
            f"Triage Urgency: {r['urgency']}",
            f"Recommended Clinical Action: {r['action']}",
            "",
            "Electrophysiological Rationale:"
        ]
        for reason in r['reasoning']:
            lines.append(f"  - {reason}")
        
        lines.append("")
        lines.append(f"Suggested Diagnostic Interventions ({r['n_tests']}):")
        for test in r['recommended_tests']:
            lines.append(f"  [{test['priority']}] {test['full_name']}")
            lines.append(f"      Indication: {test['description']}")
        lines.append("=" * 60)
        
        return "\n".join(lines)


class PhysicianDecisionSession:
    """
    Manages interactive physician reviews (Confirm / Cancel) with audit logging.
    Preserves human-in-the-loop oversight as required by Medical DSS standards.
    """

    def __init__(self, patient_id: int, recommendations: Dict[str, Any]):
        self.patient_id = patient_id
        self.recommendations = recommendations
        self.decisions: Dict[str, Dict[str, Any]] = {}
        self.audit_log: List[Dict[str, Any]] = []

        # Initialize pending decisions for all recommended tests
        for test in recommendations.get('recommended_tests', []):
            name = test['name']
            self.decisions[name] = {
                'name': name,
                'full_name': test['full_name'],
                'status': DecisionStatus.PENDING,
                'timestamp': None,
                'physician_notes': "",
                'reason': ""
            }

    def confirm_test(self, test_name: str, physician_notes: str = "") -> None:
        """Physician confirms order for a recommended diagnostic test."""
        if test_name not in self.decisions:
            raise KeyError(f"Test '{test_name}' is not in the recommended test set.")

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.decisions[test_name]['status'] = DecisionStatus.CONFIRMED
        self.decisions[test_name]['timestamp'] = now_str
        self.decisions[test_name]['physician_notes'] = physician_notes
        self.decisions[test_name]['reason'] = ""

        self.audit_log.append({
            'action': 'CONFIRM',
            'test_name': test_name,
            'timestamp': now_str,
            'notes': physician_notes
        })

    def cancel_test(self, test_name: str, reason: str = "") -> None:
        """Physician cancels or overrides a recommended diagnostic test."""
        if test_name not in self.decisions:
            raise KeyError(f"Test '{test_name}' is not in the recommended test set.")

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.decisions[test_name]['status'] = DecisionStatus.CANCELLED
        self.decisions[test_name]['timestamp'] = now_str
        self.decisions[test_name]['reason'] = reason

        self.audit_log.append({
            'action': 'CANCEL',
            'test_name': test_name,
            'timestamp': now_str,
            'reason': reason
        })

    def reset_test(self, test_name: str) -> None:
        """Reset test status back to PENDING."""
        if test_name in self.decisions:
            self.decisions[test_name]['status'] = DecisionStatus.PENDING
            self.decisions[test_name]['timestamp'] = None
            self.decisions[test_name]['reason'] = ""
            self.decisions[test_name]['physician_notes'] = ""

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of physician decisions."""
        total = len(self.decisions)
        confirmed = sum(1 for d in self.decisions.values() if d['status'] == DecisionStatus.CONFIRMED)
        cancelled = sum(1 for d in self.decisions.values() if d['status'] == DecisionStatus.CANCELLED)
        pending = sum(1 for d in self.decisions.values() if d['status'] == DecisionStatus.PENDING)

        return {
            'patient_id': self.patient_id,
            'total_recommended': total,
            'confirmed': confirmed,
            'cancelled': cancelled,
            'pending': pending,
            'decisions': list(self.decisions.values()),
            'audit_log': self.audit_log
        }
