"""
Global State & Model Registry for the FastAPI backend.
Manages lazy-loaded dataset, normalizer, Ensemble ESN models, explainers,
and in-memory physician decision sessions.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_and_prepare, load_dataset, EEG_FEATURES, LABEL_MAP_INV
from src.features.normalization import FeatureNormalizer
from src.models.ensemble import EnsembleESN
from src.dss.recommendation import RecommendationEngine, PhysicianDecisionSession, DecisionStatus
from src.utils.seed import set_global_seed
import shap
from lime.lime_tabular import LimeTabularExplainer


class AppState:
    _instance = None

    def __init__(self):
        set_global_seed(42)
        self.raw_df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.normalizer = None
        self.X_norm = None
        self.model: Optional[EnsembleESN] = None
        self.recommendation_engine = RecommendationEngine()
        self.shap_explainer = None
        self.shap_base_value = 0.5
        self.lime_explainer = None
        self.sessions: Dict[str, PhysicianDecisionSession] = {}
        self.initialized = False

    def initialize(self):
        if self.initialized:
            return

        print("[STATE] Loading clinical dataset...")
        self.X, self.y, self.feature_names, self.raw_df = load_and_prepare(feature_set='eeg_only')
        self.normalizer = FeatureNormalizer()
        self.X_norm = self.normalizer.fit_transform(self.X)

        print("[STATE] Initializing Ensemble ESN (7 estimators, 200 neurons, SR=0.95)...")
        self.model = EnsembleESN(
            n_estimators=7,
            n_neurons=200,
            spectral_radius=0.95,
            sparsity=0.0,
            noise=0.01,
            leaking_rate=0.7,
            input_scaling=0.5,
            aggregation='soft',
            bootstrap=False,
            random_state=42
        )
        self.model.fit(self.X_norm, self.y)

        print("[STATE] Pre-initializing LIME tabular explainer...")
        self.lime_explainer = LimeTabularExplainer(
            training_data=self.X_norm,
            feature_names=self.feature_names,
            class_names=['Control', 'Stroke'],
            mode='classification',
            random_state=42,
            kernel_width=0.75,
            verbose=False
        )

        print("[STATE] Pre-initializing SHAP KernelExplainer...")
        def predict_stroke_fn(x):
            return self.model.predict_proba(x)[:, 1]

        # Use 5 kmeans medoids for fast, lightweight inference
        background = shap.kmeans(self.X_norm, 5)
        self.shap_explainer = shap.KernelExplainer(predict_stroke_fn, background)
        base_val = self.shap_explainer.expected_value
        if hasattr(base_val, '__len__'):
            self.shap_base_value = float(np.array(base_val).flat[0])
        else:
            self.shap_base_value = float(base_val)

        self.initialized = True
        self.load_sessions()
        print("[STATE] All clinical models, data loaders, and XAI explainers ready.")

    def save_sessions(self):
        """Persist physician decision sessions to disk for continuity across restarts."""
        try:
            import json
            sessions_file = PROJECT_ROOT / "data" / "metadata" / "dss_sessions.json"
            sessions_file.parent.mkdir(parents=True, exist_ok=True)
            data_to_save = {}
            for sid, sess in self.sessions.items():
                data_to_save[sid] = sess.get_summary()
            with open(sessions_file, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2)
        except Exception as e:
            print(f"[STATE] Warning: Could not save sessions to file: {e}")

    def load_sessions(self):
        """Restore persisted physician decision sessions from disk."""
        try:
            import json
            sessions_file = PROJECT_ROOT / "data" / "metadata" / "dss_sessions.json"
            if not sessions_file.exists():
                return
            with open(sessions_file, 'r', encoding='utf-8') as f:
                data_loaded = json.load(f)
            for sid, sdata in data_loaded.items():
                pid = sdata.get('patient_id', 0)
                sess = self.get_or_create_session(sid, patient_id=pid)
                for d in sdata.get('decisions', []):
                    name = d.get('name')
                    if name in sess.decisions:
                        st = d.get('status', DecisionStatus.PENDING)
                        sess.decisions[name]['status'] = getattr(st, 'value', str(st))
                        sess.decisions[name]['timestamp'] = d.get('timestamp')
                        sess.decisions[name]['physician_notes'] = d.get('physician_notes', '')
                        sess.decisions[name]['reason'] = d.get('reason', '')
                sess.audit_log = sdata.get('audit_log', [])
            print(f"[STATE] Restored {len(data_loaded)} persisted physician decision sessions.")
        except Exception as e:
            print(f"[STATE] Warning: Could not load sessions from file: {e}")

    def get_or_create_session(self, session_id: str, patient_id: int = 0, stroke_prob: float = 0.8) -> PhysicianDecisionSession:
        if session_id not in self.sessions:
            pred_class = "Stroke" if stroke_prob >= 0.50 else "Control"
            recs = self.recommendation_engine.get_recommendations(stroke_prob, pred_class)
            self.sessions[session_id] = PhysicianDecisionSession(patient_id=patient_id, recommendations=recs)
        return self.sessions[session_id]


# Singleton instance
state = AppState()
