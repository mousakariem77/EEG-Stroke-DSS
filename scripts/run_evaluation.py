"""
Comprehensive Evaluation: Feature Selection + ROC/AUC + Model Comparison

Runs the full evaluation pipeline:
1. Feature selection (ANOVA + MI)
2. Models with selected features
3. ROC/AUC curves
4. Full comparison chart
"""
import sys
sys.path.insert(0, r"d:\Personal\Explainable EEG Stroke DSS")

import numpy as np
from pathlib import Path
from sklearn.base import clone

from src.data.loader import load_and_prepare
from src.features.normalization import FeatureNormalizer
from src.features.selection import FeatureSelector
from src.data.splitter import get_cv_folds
from src.models.esn import EchoStateNetwork
from src.models.ensemble import EnsembleESN
from src.models.baselines import compute_metrics, get_baseline_classifiers
from src.evaluation.visualization import (
    plot_roc_curves, plot_precision_recall, 
    plot_model_comparison, plot_confusion_matrix
)
from src.utils.seed import set_global_seed

set_global_seed(42)

FIGURES_DIR = Path(r"d:\Personal\Explainable EEG Stroke DSS\experiments\figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ESN_PARAMS = {
    'n_neurons': 200, 'spectral_radius': 0.95, 'sparsity': 0.0,
    'noise': 0.01, 'leaking_rate': 0.7, 'input_scaling': 0.5,
}

# ============================================================
# PHASE A: Feature Selection
# ============================================================
print("=" * 70)
print("PHASE A: FEATURE SELECTION")
print("=" * 70)

X, y, feature_names, _ = load_and_prepare(feature_set='eeg_only')
norm = FeatureNormalizer()
X_norm = norm.fit_transform(X)

# Run feature selection
for k in [4, 6, 8]:
    fs = FeatureSelector(method='combined', k=k)
    fs.fit(X_norm, y, feature_names)
    
    table = fs.get_ranking_table()
    print(f"\n  Top {k} features (combined ANOVA + MI):")
    for row in table[:k]:
        marker = "***" if row['selected'] else ""
        print(f"    {row['rank']:2d}. {row['feature']:15s}  "
              f"ANOVA={row['anova_f']:7.2f} (p={row['anova_p']:.4f})  "
              f"MI={row['mi']:.4f}  Combined={row['combined']:.4f} {marker}")

# Save feature selection plot (k=4 as paper suggests)
fs4 = FeatureSelector(method='combined', k=4)
fs4.fit(X_norm, y, feature_names)
fs4.plot_scores(save_path=str(FIGURES_DIR / "feature_selection.png"))

# ============================================================
# PHASE B: Model Evaluation with LOOCV + ROC
# ============================================================
print("\n" + "=" * 70)
print("PHASE B: MODEL EVALUATION (LOOCV)")
print("=" * 70)

all_metrics = {}
roc_data = {}

# --- Baselines ---
baselines = get_baseline_classifiers(42)
for name, clf in baselines.items():
    y_pred = np.zeros_like(y)
    y_proba = np.zeros(len(y))
    
    for X_tr, X_te, y_tr, y_te, fi in get_cv_folds(X_norm, y, method='loocv'):
        c = clone(clf)
        c.fit(X_tr, y_tr)
        y_pred[fi] = c.predict(X_te)[0]
        
        if hasattr(c, 'predict_proba'):
            y_proba[fi] = c.predict_proba(X_te)[0, 1]
        elif hasattr(c, 'decision_function'):
            y_proba[fi] = c.decision_function(X_te)[0]
    
    m = compute_metrics(y, y_pred)
    all_metrics[name] = m
    
    if hasattr(clf, 'predict_proba') or hasattr(clf, 'decision_function'):
        roc_data[name] = (y, y_proba)
    
    print(f"  {name:25s}: Acc={m['accuracy']:.2%}, F1={m['f1']:.2%}")

# --- Single ESN ---
y_pred_esn = np.zeros_like(y)
y_proba_esn = np.zeros(len(y))

for X_tr, X_te, y_tr, y_te, fi in get_cv_folds(X_norm, y, method='loocv'):
    esn = EchoStateNetwork(**ESN_PARAMS, random_state=42)
    esn.fit(X_tr, y_tr)
    y_pred_esn[fi] = esn.predict(X_te)[0]
    y_proba_esn[fi] = esn.predict_proba(X_te)[0, 1]

m_esn = compute_metrics(y, y_pred_esn)
all_metrics['Single ESN'] = m_esn
roc_data['Single ESN'] = (y, y_proba_esn)
print(f"  {'Single ESN':25s}: Acc={m_esn['accuracy']:.2%}, F1={m_esn['f1']:.2%}")

# --- Ensemble ESN ---
y_pred_eesn = np.zeros_like(y)
y_proba_eesn = np.zeros(len(y))

for X_tr, X_te, y_tr, y_te, fi in get_cv_folds(X_norm, y, method='loocv'):
    eesn = EnsembleESN(
        n_estimators=7, **ESN_PARAMS,
        aggregation='soft', bootstrap=False, random_state=42
    )
    eesn.fit(X_tr, y_tr)
    y_pred_eesn[fi] = eesn.predict(X_te)[0]
    y_proba_eesn[fi] = eesn.predict_proba(X_te)[0, 1]

m_eesn = compute_metrics(y, y_pred_eesn)
all_metrics['E-ESN (7)'] = m_eesn
roc_data['E-ESN (7)'] = (y, y_proba_eesn)
print(f"  {'E-ESN (7)':25s}: Acc={m_eesn['accuracy']:.2%}, F1={m_eesn['f1']:.2%}")

# --- ESN with Feature Selection (k=4) ---
X_fs = fs4.transform(X_norm)
y_pred_fs = np.zeros_like(y)
y_proba_fs = np.zeros(len(y))

for X_tr, X_te, y_tr, y_te, fi in get_cv_folds(X_fs, y, method='loocv'):
    esn_fs = EchoStateNetwork(**ESN_PARAMS, random_state=42)
    esn_fs.fit(X_tr, y_tr)
    y_pred_fs[fi] = esn_fs.predict(X_te)[0]
    y_proba_fs[fi] = esn_fs.predict_proba(X_te)[0, 1]

m_fs = compute_metrics(y, y_pred_fs)
all_metrics['ESN (4 features)'] = m_fs
roc_data['ESN (4 features)'] = (y, y_proba_fs)
print(f"  {'ESN (4 features)':25s}: Acc={m_fs['accuracy']:.2%}, F1={m_fs['f1']:.2%}")

# ============================================================
# PHASE C: VISUALIZATIONS
# ============================================================
print("\n" + "=" * 70)
print("PHASE C: GENERATING VISUALIZATIONS")
print("=" * 70)

# ROC curves — select key models
key_models = {k: v for k, v in roc_data.items() 
              if k in ['Random Forest', 'SVM (RBF)', 'KNN (k=3)',
                       'Single ESN', 'E-ESN (7)', 'ESN (4 features)']}

auc_scores = plot_roc_curves(
    key_models,
    save_path=str(FIGURES_DIR / "roc_curves.png")
)

print("\nAUC Scores:")
for name, score in sorted(auc_scores.items(), key=lambda x: -x[1]):
    print(f"  {name:25s}: {score:.4f}")

# Precision-Recall curves
plot_precision_recall(
    key_models,
    save_path=str(FIGURES_DIR / "pr_curves.png")
)

# Model comparison chart
comparison_metrics = {k: v for k, v in all_metrics.items()
                     if k in ['Random Forest', 'KNN (k=3)', 
                              'Single ESN', 'E-ESN (7)', 'ESN (4 features)']}

plot_model_comparison(
    comparison_metrics,
    save_path=str(FIGURES_DIR / "model_comparison.png")
)

# Confusion matrices for key models
plot_confusion_matrix(
    m_esn['confusion_matrix'],
    save_path=str(FIGURES_DIR / "cm_single_esn.png"),
    title="Confusion Matrix — Single ESN (86.84%)"
)

plot_confusion_matrix(
    m_eesn['confusion_matrix'],
    save_path=str(FIGURES_DIR / "cm_eesn.png"),
    title="Confusion Matrix — E-ESN (84.21%)"
)

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("FULL RESULTS TABLE")
print("=" * 70)
print(f"\n{'Model':25s} | {'Acc':>7s} | {'F1':>7s} | {'Sens':>7s} | {'Spec':>7s} | {'AUC':>7s}")
print("-" * 75)

for name in ['KNN (k=3)', 'Random Forest', 'SVM (RBF)', 'Single ESN', 
             'E-ESN (7)', 'ESN (4 features)']:
    if name in all_metrics:
        m = all_metrics[name]
        a = auc_scores.get(name, 0)
        print(f"{name:25s} | {m['accuracy']:>6.1%} | {m['f1']:>6.1%} | "
              f"{m['sensitivity']:>6.1%} | {m['specificity']:>6.1%} | {a:>6.3f}")

print("-" * 75)
print(f"{'Paper Target':25s} | {'96.5%':>7s} | {'94.7%':>7s} | {'96.4%':>7s} | {'81.8%':>7s} |   N/A")
print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
