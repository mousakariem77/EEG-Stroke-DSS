"""
Module M10: Recommendation Engine

Rule-based recommendation system for diagnostic tests.
Generates recommendations based on stroke prediction probability.

FACT FROM PAPER:
    - Recommends: MRI, CBC, BMP, Coagulation Profile, Lipid Profile, D-Dimer
    - No detailed logic provided (must implement ourselves)

IMPLEMENTATION:
    - Risk levels based on stroke probability
    - Tests prioritized by clinical relevance
    - Reasoning generated from SHAP/LIME feature contributions
"""
from typing import Dict, List, Optional
import numpy as np


# Diagnostic tests and their descriptions
DIAGNOSTIC_TESTS = {
    'MRI': {
        'full_name': 'Magnetic Resonance Imaging (MRI)',
        'description': 'Brain imaging to detect stroke location, type, and extent',
        'priority': 1,
        'category': 'imaging'
    },
    'CBC': {
        'full_name': 'Complete Blood Count (CBC)',
        'description': 'Blood cell counts to detect infection, anemia, or clotting disorders',
        'priority': 2,
        'category': 'blood'
    },
    'BMP': {
        'full_name': 'Basic Metabolic Panel (BMP)',
        'description': 'Electrolytes, glucose, kidney function assessment',
        'priority': 3,
        'category': 'blood'
    },
    'Coagulation Profile': {
        'full_name': 'Coagulation Profile (PT/INR, aPTT)',
        'description': 'Blood clotting function to guide anticoagulation therapy',
        'priority': 2,
        'category': 'blood'
    },
    'Lipid Profile': {
        'full_name': 'Lipid Profile',
        'description': 'Cholesterol and triglyceride levels for cardiovascular risk',
        'priority': 4,
        'category': 'blood'
    },
    'D-Dimer': {
        'full_name': 'D-Dimer Test',
        'description': 'Fibrin degradation product to detect active blood clotting',
        'priority': 3,
        'category': 'blood'
    }
}


class RecommendationEngine:
    """
    Rule-based diagnostic recommendation engine.
    
    Generates test recommendations based on:
    - Stroke probability from E-ESN
    - Risk level classification
    - Top contributing features (from SHAP/LIME)
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
        Classify risk level based on stroke probability.
        
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
    ) -> Dict:
        """
        Generate diagnostic recommendations.
        
        Args:
            stroke_probability: Probability of stroke (0-1)
            predicted_class: 'Stroke' or 'Control'
            feature_contributions: Optional LIME/SHAP feature contributions
        
        Returns:
            dict with risk_level, recommended_tests, reasoning, and actions
        """
        risk_level = self.classify_risk(stroke_probability)
        
        # Select tests based on risk level
        if risk_level == 'high':
            test_names = list(DIAGNOSTIC_TESTS.keys())  # All tests
            urgency = 'URGENT'
            action = 'Immediate comprehensive stroke workup recommended'
        elif risk_level == 'medium':
            test_names = ['MRI', 'CBC', 'BMP', 'Coagulation Profile']
            urgency = 'PRIORITY'
            action = 'Further evaluation recommended within 24 hours'
        else:
            test_names = ['CBC', 'BMP']  # Basic screening
            urgency = 'ROUTINE'
            action = 'Routine follow-up recommended'
        
        # Build test details
        recommended_tests = []
        for name in test_names:
            test = DIAGNOSTIC_TESTS[name].copy()
            test['name'] = name
            recommended_tests.append(test)
        
        # Sort by priority
        recommended_tests.sort(key=lambda t: t['priority'])
        
        # Build reasoning
        reasoning = self._build_reasoning(
            stroke_probability, predicted_class, risk_level, feature_contributions
        )
        
        return {
            'risk_level': risk_level,
            'urgency': urgency,
            'stroke_probability': stroke_probability,
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
        """Build human-readable reasoning for recommendations."""
        reasons = []
        
        reasons.append(
            f"EEG-based prediction: {predicted_class} "
            f"(probability: {stroke_probability:.1%})"
        )
        reasons.append(f"Risk classification: {risk_level.upper()}")
        
        if feature_contributions:
            # Top 3 contributing features
            top_features = sorted(
                feature_contributions, 
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:3]
            
            for feat_name, contrib in top_features:
                direction = "increases" if contrib > 0 else "decreases"
                reasons.append(
                    f"Key factor: {feat_name} ({direction} stroke risk, "
                    f"impact: {abs(contrib):.3f})"
                )
        
        return reasons
    
    def format_report(self, recommendation: Dict) -> str:
        """Format recommendation as readable text report."""
        r = recommendation
        lines = []
        
        lines.append("=" * 50)
        lines.append("DIAGNOSTIC RECOMMENDATION REPORT")
        lines.append("=" * 50)
        lines.append(f"Prediction: {r['predicted_class']}")
        lines.append(f"Stroke Probability: {r['stroke_probability']:.1%}")
        lines.append(f"Risk Level: {r['risk_level'].upper()}")
        lines.append(f"Urgency: {r['urgency']}")
        lines.append(f"Action: {r['action']}")
        lines.append("")
        lines.append("Reasoning:")
        for reason in r['reasoning']:
            lines.append(f"  - {reason}")
        lines.append("")
        lines.append(f"Recommended Tests ({r['n_tests']}):")
        for test in r['recommended_tests']:
            lines.append(f"  [{test['priority']}] {test['full_name']}")
            lines.append(f"      {test['description']}")
        lines.append("=" * 50)
        
        return "\n".join(lines)
