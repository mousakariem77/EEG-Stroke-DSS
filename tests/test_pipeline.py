"""
Unit Tests for EEG Stroke DSS Pipeline

Tests all core modules:
- Data Loader
- Feature Normalization  
- ESN Model
- Ensemble ESN
- SHAP Explainer
- LIME Explainer
- Recommendation Engine
"""
import sys
sys.path.insert(0, r"d:\Personal\Explainable EEG Stroke DSS")

import numpy as np
import pytest
from pathlib import Path

from src.data.loader import load_and_prepare, load_dataset, encode_labels
from src.features.normalization import FeatureNormalizer
from src.data.splitter import get_cv_folds, get_train_test_split
from src.models.esn import EchoStateNetwork
from src.models.ensemble import EnsembleESN
from src.models.baselines import compute_metrics
from src.dss.recommendation import RecommendationEngine
from src.utils.seed import set_global_seed


# ============================================================
# TEST: Data Loading
# ============================================================
class TestDataLoader:
    def test_load_dataset(self):
        df = load_dataset()
        assert len(df) == 38, f"Expected 38 rows, got {len(df)}"
        assert 'Participant' in df.columns  # label column is 'Participant' (stroke/control)
    
    def test_encode_labels(self):
        df = load_dataset()
        df_enc = encode_labels(df)
        assert 'label' in df_enc.columns
        assert set(df_enc['label'].unique()) == {0, 1}
    
    def test_load_and_prepare_all(self):
        X, y, fnames, df = load_and_prepare(feature_set='all')
        assert X.shape[0] == 38
        assert X.shape[1] == len(fnames)
        assert len(y) == 38
        assert np.sum(y == 0) == 19  # balanced
        assert np.sum(y == 1) == 19
    
    def test_load_and_prepare_eeg_only(self):
        X, y, fnames, _ = load_and_prepare(feature_set='eeg_only')
        assert X.shape[1] == 12  # 12 EEG features
        assert 'Age' not in fnames
        assert 'Gender' not in fnames
    
    def test_load_and_prepare_paper_top(self):
        X, y, fnames, _ = load_and_prepare(feature_set='paper_top')
        assert 'DTR' in fnames
        assert 'RP Delta' in fnames


# ============================================================
# TEST: Normalization
# ============================================================
class TestNormalization:
    def test_fit_transform(self):
        X = np.random.rand(20, 5) * 100
        norm = FeatureNormalizer()
        X_n = norm.fit_transform(X)
        
        assert X_n.shape == X.shape
        assert X_n.min() >= 0.0 - 1e-10
        assert X_n.max() <= 1.0 + 1e-10
    
    def test_transform_consistency(self):
        X_train = np.random.rand(20, 5) * 100
        X_test = np.random.rand(5, 5) * 100
        
        norm = FeatureNormalizer()
        norm.fit(X_train)
        
        X_train_n = norm.transform(X_train)
        X_test_n = norm.transform(X_test)
        
        assert X_train_n.shape == X_train.shape
        assert X_test_n.shape == X_test.shape


# ============================================================
# TEST: CV Splitter
# ============================================================
class TestSplitter:
    def test_loocv(self):
        X = np.random.rand(10, 3)
        y = np.array([0]*5 + [1]*5)
        
        folds = list(get_cv_folds(X, y, method='loocv'))
        assert len(folds) == 10
        
        for X_tr, X_te, y_tr, y_te, idx in folds:
            assert len(X_te) == 1
            assert len(X_tr) == 9
    
    def test_kfold(self):
        X = np.random.rand(20, 3)
        y = np.array([0]*10 + [1]*10)
        
        folds = list(get_cv_folds(X, y, n_splits=5))  # default method is stratified k-fold
        assert len(folds) == 5
    
    def test_train_test_split(self):
        X = np.random.rand(20, 3)
        y = np.array([0]*10 + [1]*10)
        
        X_tr, X_te, y_tr, y_te = get_train_test_split(X, y, test_size=0.2)
        assert len(X_tr) + len(X_te) == 20


# ============================================================
# TEST: ESN Model
# ============================================================
class TestESN:
    def setup_method(self):
        set_global_seed(42)
        self.X = np.random.rand(30, 5)
        self.y = np.array([0]*15 + [1]*15)
    
    def test_fit_predict(self):
        esn = EchoStateNetwork(n_neurons=50, random_state=42)
        esn.fit(self.X[:20], self.y[:20])
        
        y_pred = esn.predict(self.X[20:])
        assert len(y_pred) == 10
        assert set(y_pred).issubset({0, 1})
    
    def test_predict_proba(self):
        esn = EchoStateNetwork(n_neurons=50, random_state=42)
        esn.fit(self.X[:20], self.y[:20])
        
        proba = esn.predict_proba(self.X[20:])
        assert proba.shape == (10, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    
    def test_reservoir_shape(self):
        esn = EchoStateNetwork(n_neurons=100, random_state=42)
        esn.fit(self.X[:20], self.y[:20])
        
        assert esn.W_.shape == (100, 100)
        assert esn.Win_.shape == (100, 6)  # 5 features + 1 bias
    
    def test_spectral_radius(self):
        esn = EchoStateNetwork(n_neurons=100, spectral_radius=0.95, random_state=42)
        esn.fit(self.X[:20], self.y[:20])
        
        eigenvalues = np.abs(np.linalg.eigvals(esn.W_))
        actual_sr = np.max(eigenvalues)
        np.testing.assert_allclose(actual_sr, 0.95, atol=0.01)
    
    def test_deterministic_with_seed(self):
        esn1 = EchoStateNetwork(n_neurons=50, random_state=42)
        esn1.fit(self.X[:20], self.y[:20])
        p1 = esn1.predict(self.X[20:])
        
        esn2 = EchoStateNetwork(n_neurons=50, random_state=42)
        esn2.fit(self.X[:20], self.y[:20])
        p2 = esn2.predict(self.X[20:])
        
        np.testing.assert_array_equal(p1, p2)


# ============================================================
# TEST: Ensemble ESN
# ============================================================
class TestEnsembleESN:
    def setup_method(self):
        set_global_seed(42)
        self.X = np.random.rand(30, 5)
        self.y = np.array([0]*15 + [1]*15)
    
    def test_fit_predict(self):
        eesn = EnsembleESN(n_estimators=3, n_neurons=50, random_state=42)
        eesn.fit(self.X[:20], self.y[:20])
        
        y_pred = eesn.predict(self.X[20:])
        assert len(y_pred) == 10
    
    def test_predict_proba(self):
        eesn = EnsembleESN(n_estimators=3, n_neurons=50, random_state=42)
        eesn.fit(self.X[:20], self.y[:20])
        
        proba = eesn.predict_proba(self.X[20:])
        assert proba.shape == (10, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    
    def test_n_estimators(self):
        eesn = EnsembleESN(n_estimators=7, n_neurons=50, random_state=42)
        eesn.fit(self.X[:20], self.y[:20])
        
        assert len(eesn.estimators_) == 7
    
    def test_bootstrap_creates_different_models(self):
        eesn = EnsembleESN(
            n_estimators=3, n_neurons=50, bootstrap=True, random_state=42
        )
        eesn.fit(self.X[:20], self.y[:20])
        
        # Each estimator should have different weights
        w1 = eesn.estimators_[0].Wout_
        w2 = eesn.estimators_[1].Wout_
        assert not np.allclose(w1, w2)


# ============================================================
# TEST: Metrics
# ============================================================
class TestMetrics:
    def test_perfect_prediction(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        
        m = compute_metrics(y_true, y_pred)
        assert m['accuracy'] == 1.0
        assert m['f1'] == 1.0
        assert m['sensitivity'] == 1.0
        assert m['specificity'] == 1.0
    
    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        
        m = compute_metrics(y_true, y_pred)
        assert m['accuracy'] == 0.0


# ============================================================
# TEST: Recommendation Engine
# ============================================================
class TestRecommendationEngine:
    def setup_method(self):
        self.engine = RecommendationEngine()
    
    def test_high_risk(self):
        rec = self.engine.get_recommendations(0.85, 'Stroke')
        assert rec['risk_level'] == 'high'
        assert rec['urgency'] == 'URGENT'
        assert rec['n_tests'] == 6  # all tests
    
    def test_medium_risk(self):
        rec = self.engine.get_recommendations(0.55, 'Stroke')
        assert rec['risk_level'] == 'medium'
        assert rec['urgency'] == 'PRIORITY'
    
    def test_low_risk(self):
        rec = self.engine.get_recommendations(0.2, 'Control')
        assert rec['risk_level'] == 'low'
        assert rec['urgency'] == 'ROUTINE'
        assert rec['n_tests'] == 2
    
    def test_report_format(self):
        rec = self.engine.get_recommendations(0.85, 'Stroke')
        report = self.engine.format_report(rec)
        assert 'DIAGNOSTIC RECOMMENDATION REPORT' in report
        assert 'MRI' in report


# ============================================================
# INTEGRATION TEST: Full Pipeline
# ============================================================
class TestIntegration:
    def test_full_pipeline(self):
        """End-to-end test: load → normalize → train → predict."""
        set_global_seed(42)
        
        X, y, fnames, _ = load_and_prepare(feature_set='eeg_only')
        norm = FeatureNormalizer()
        X_n = norm.fit_transform(X)
        
        X_tr, X_te, y_tr, y_te = get_train_test_split(
            X_n, y, test_size=0.2, random_state=42
        )
        
        eesn = EnsembleESN(
            n_estimators=3, n_neurons=100,
            spectral_radius=0.95, noise=0.01,
            leaking_rate=0.7, input_scaling=0.5,
            random_state=42
        )
        eesn.fit(X_tr, y_tr)
        
        y_pred = eesn.predict(X_te)
        proba = eesn.predict_proba(X_te)
        
        assert len(y_pred) == len(y_te)
        assert proba.shape[1] == 2
        
        metrics = compute_metrics(y_te, y_pred)
        assert 'accuracy' in metrics
        assert 0.0 <= metrics['accuracy'] <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
