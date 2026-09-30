"""
Full Pipeline Runner — End-to-End: Data → ESN → E-ESN → SHAP → LIME → DSS

This is the main entry point for the complete pipeline as described in the paper.
"""
import sys
sys.path.insert(0, r"d:\Personal\Explainable EEG Stroke DSS")

import numpy as np
import time
import os
from pathlib import Path

from src.data.loader import load_and_prepare, LABEL_MAP_INV
from src.features.normalization import FeatureNormalizer
from src.data.splitter import get_cv_folds, get_train_test_split
from src.models.esn import EchoStateNetwork
from src.models.ensemble import EnsembleESN
from src.models.baselines import compute_metrics, get_baseline_classifiers
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.lime_explainer import LIMEExplainer
from src.dss.recommendation import RecommendationEngine
from src.utils.seed import set_global_seed

# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42
RESULTS_DIR = Path(r"d:\Personal\Explainable EEG Stroke DSS\experiments\results")
FIGURES_DIR = Path(r"d:\Personal\Explainable EEG Stroke DSS\experiments\figures")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Optimal params from hyperparameter sweep
ESN_PARAMS = {
    'n_neurons': 200,
    'spectral_radius': 0.95,
    'sparsity': 0.0,
    'noise': 0.01,       # tuned
    'leaking_rate': 0.7,  # tuned
    'input_scaling': 0.5, # tuned
}

set_global_seed(SEED)

# ============================================================
# PHASE 1: DATA LOADING
# ============================================================
print("=" * 70)
print("PHASE 1: DATA LOADING")
print("=" * 70)

X, y, feature_names, raw_df = load_and_prepare(feature_set='eeg_only')
normalizer = FeatureNormalizer()
X_norm = normalizer.fit_transform(X)

# Hold-out split for XAI (need a trained model on full-ish data)
X_train, X_test, y_train, y_test = get_train_test_split(
    X_norm, y, test_size=0.2, random_state=SEED
)

# ============================================================
# PHASE 2: BASELINE COMPARISON (5-Fold CV)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 2: BASELINE COMPARISON (5-Fold CV)")
print("=" * 70)

baselines = get_baseline_classifiers(SEED)
baseline_results = {}

for name, clf in baselines.items():
    fold_accs = []
    for X_tr, X_te, y_tr, y_te, fi in get_cv_folds(X_norm, y, n_splits=5, random_state=SEED):
        from sklearn.base import clone
        c = clone(clf)
        c.fit(X_tr, y_tr)
        y_pred = c.predict(X_te)
        m = compute_metrics(y_te, y_pred)
        fold_accs.append(m['accuracy'])
    
    mean_acc = np.mean(fold_accs)
    baseline_results[name] = mean_acc
    print(f"  {name:25s}: {mean_acc:.2%}")

# ============================================================
# PHASE 3: SINGLE ESN (LOOCV)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 3: SINGLE ESN (LOOCV)")
print("=" * 70)

y_pred_esn = np.zeros_like(y)
for X_tr, X_te, y_tr, y_te, fi in get_cv_folds(X_norm, y, method='loocv'):
    esn = EchoStateNetwork(**ESN_PARAMS, random_state=SEED)
    esn.fit(X_tr, y_tr)
    y_pred_esn[fi] = esn.predict(X_te)[0]

m_esn = compute_metrics(y, y_pred_esn)
print(f"Single ESN: Acc={m_esn['accuracy']:.2%}, F1={m_esn['f1']:.2%}, "
      f"Sens={m_esn['sensitivity']:.2%}, Spec={m_esn['specificity']:.2%}")

# ============================================================
# PHASE 4: ENSEMBLE ESN (LOOCV)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 4: ENSEMBLE ESN (LOOCV)")
print("=" * 70)

y_pred_eesn = np.zeros_like(y)
y_proba_eesn = np.zeros((len(y), 2))

start_time = time.time()
for X_tr, X_te, y_tr, y_te, fi in get_cv_folds(X_norm, y, method='loocv'):
    eesn = EnsembleESN(
        n_estimators=7, **ESN_PARAMS,
        aggregation='soft', bootstrap=False,  # no-bootstrap better for small data
        random_state=SEED
    )
    eesn.fit(X_tr, y_tr)
    y_pred_eesn[fi] = eesn.predict(X_te)[0]
    y_proba_eesn[fi] = eesn.predict_proba(X_te)[0]

elapsed = time.time() - start_time
m_eesn = compute_metrics(y, y_pred_eesn)

print(f"E-ESN (7, no-bootstrap):")
print(f"  Accuracy:    {m_eesn['accuracy']:.2%}")
print(f"  PPV:         {m_eesn['ppv']:.2%}")
print(f"  NPV:         {m_eesn['npv']:.2%}")
print(f"  Sensitivity: {m_eesn['sensitivity']:.2%}")
print(f"  Specificity: {m_eesn['specificity']:.2%}")
print(f"  F1:          {m_eesn['f1']:.2%}")
print(f"  Time:        {elapsed:.2f}s")
print(f"  CM:          {m_eesn['confusion_matrix'].tolist()}")

# ============================================================
# PHASE 5: XAI — SHAP GLOBAL EXPLANATION
# ============================================================
print("\n" + "=" * 70)
print("PHASE 5: SHAP GLOBAL EXPLANATION")
print("=" * 70)

# Train final E-ESN on ALL data for XAI
eesn_final = EnsembleESN(
    n_estimators=7, **ESN_PARAMS,
    aggregation='soft', bootstrap=False,
    random_state=SEED
)
eesn_final.fit(X_norm, y)

# SHAP analysis
shap_explainer = SHAPExplainer(eesn_final, X_norm, feature_names)
shap_values = shap_explainer.compute_shap_values(X_norm, nsamples=100)

# Feature importance
importance = shap_explainer.get_feature_importance()
print("\nSHAP Feature Importance (stroke class):")
for i, (name, val) in enumerate(importance.items()):
    print(f"  {i+1}. {name:20s}: {val:.4f}")

# Save plots
shap_explainer.plot_summary(X_norm, save_path=str(FIGURES_DIR / "shap_summary.png"))
shap_explainer.plot_bar(save_path=str(FIGURES_DIR / "shap_bar.png"))

# ============================================================
# PHASE 6: XAI — LIME LOCAL EXPLANATION
# ============================================================
print("\n" + "=" * 70)
print("PHASE 6: LIME LOCAL EXPLANATION")
print("=" * 70)

lime_explainer = LIMEExplainer(eesn_final, X_norm, feature_names)
lime_summaries = lime_explainer.explain_multiple(
    X_norm, y, save_dir=str(FIGURES_DIR / "lime"),
    n_per_class=2, num_features=8
)

for s in lime_summaries:
    print(f"  True: {s['true_label']:10s} -> Predicted: {s['predicted_class']:10s} "
          f"P(stroke)={s['predicted_proba'][1]:.3f}")

# ============================================================
# PHASE 7: DSS RECOMMENDATIONS
# ============================================================
print("\n" + "=" * 70)
print("PHASE 7: DSS RECOMMENDATIONS")
print("=" * 70)

rec_engine = RecommendationEngine()

# Show recommendations for 2 example patients
for idx in [0, 19]:  # first control, first stroke
    proba = eesn_final.predict_proba(X_norm[idx:idx+1])[0]
    stroke_prob = proba[1]
    pred_class = 'Stroke' if np.argmax(proba) == 1 else 'Control'
    
    rec = rec_engine.get_recommendations(stroke_prob, pred_class)
    report = rec_engine.format_report(rec)
    print(f"\n--- Patient #{idx} (True: {LABEL_MAP_INV[y[idx]]}) ---")
    print(report)

# ============================================================
# FINAL COMPARISON TABLE
# ============================================================
print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(f"\n{'Model':25s} | {'Accuracy':>10s} | {'F1':>10s}")
print("-" * 50)
for name, acc in sorted(baseline_results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:25s} | {acc:>9.2%} | {'N/A':>10s}")
print("-" * 50)
print(f"{'Single ESN':25s} | {m_esn['accuracy']:>9.2%} | {m_esn['f1']:>9.2%}")
print(f"{'E-ESN (7)':25s} | {m_eesn['accuracy']:>9.2%} | {m_eesn['f1']:>9.2%}")
print("-" * 50)
print(f"{'Paper Target':25s} | {'96.50%':>10s} | {'94.73%':>10s}")

print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
