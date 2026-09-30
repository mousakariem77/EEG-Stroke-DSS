#!/usr/bin/env python
"""
scripts/run_full_evaluation.py

Comprehensive evaluation replicating ALL experiments from the paper:
  1. Feature distribution analysis (Fig. 3 from paper)
  2. All models × ALL CV methods (LOOCV, 5-Fold, 10-Fold)
  3. Feature set comparison (EEG-only 12 feat vs All 15 feat)
  4. Feature selection experiments (k = 4, 8, all)
  5. Generate comparison tables and figures

Paper: Bouazizi & Ltifi (2024), Decision Support Systems 178
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, r"d:\Personal\Explainable EEG Stroke DSS")

import numpy as np
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.base import clone

from src.data.loader import (
    load_and_prepare, EEG_FEATURES, DEMOGRAPHIC_FEATURES
)
from src.features.normalization import FeatureNormalizer
from src.features.selection import FeatureSelector
from src.data.splitter import get_cv_folds
from src.models.esn import EchoStateNetwork
from src.models.ensemble import EnsembleESN
from src.models.baselines import get_baseline_classifiers, compute_metrics
from src.utils.seed import set_global_seed

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════
SEED = 42
RESULTS_DIR = Path(r"d:\Personal\Explainable EEG Stroke DSS\experiments\results")
FIGURES_DIR = Path(r"d:\Personal\Explainable EEG Stroke DSS\experiments\figures")

ESN_PARAMS = dict(
    n_neurons=200, spectral_radius=0.95, sparsity=0.0,
    noise=0.01, leaking_rate=0.7, input_scaling=0.5,
    random_state=SEED
)
EESN_PARAMS = dict(
    n_estimators=7, aggregation='soft', bootstrap=False,
    **ESN_PARAMS
)


# ═══════════════════════════════════════════════════════════
# EVALUATION CORE
# ═══════════════════════════════════════════════════════════
def evaluate_model_cv(model_factory, X_norm, y, cv_method, n_splits=None):
    """
    Evaluate a model with cross-validation.
    Returns aggregate metrics over all folds.
    """
    y_true_all = []
    y_pred_all = []
    y_proba_all = []

    for X_tr, X_te, y_tr, y_te, _ in get_cv_folds(
        X_norm, y, method=cv_method, n_splits=n_splits, random_state=SEED
    ):
        model = model_factory()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_te)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                y_proba_pos = proba[:, 1]
            else:
                y_proba_pos = proba.ravel()
        else:
            y_proba_pos = y_pred.astype(float)

        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(y_pred.tolist())
        y_proba_all.extend(y_proba_pos.tolist())

    metrics = compute_metrics(np.array(y_true_all), np.array(y_pred_all))
    return {
        'accuracy': metrics['accuracy'],
        'f1': metrics['f1'],
        'sensitivity': metrics['sensitivity'],
        'specificity': metrics['specificity'],
        'ppv': metrics['ppv'],
        'npv': metrics['npv'],
    }


# ═══════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════
def plot_feature_distributions(X, y, feature_names, save_path):
    """Generate boxplots of features by class — Paper Fig. 3."""
    n = len(feature_names)
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
    axes = axes.flatten()

    for i, fname in enumerate(feature_names):
        ax = axes[i]
        ctrl = X[y == 0, i]
        strk = X[y == 1, i]

        bp = ax.boxplot(
            [ctrl, strk], tick_labels=['Control', 'Stroke'],
            patch_artist=True, widths=0.6
        )
        bp['boxes'][0].set(facecolor='#667eea', alpha=0.7)
        bp['boxes'][1].set(facecolor='#ff6b6b', alpha=0.7)
        for m in bp['medians']:
            m.set(color='black', linewidth=2)

        ax.set_title(fname, fontsize=10, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(
        'Feature Distribution by Class (Control vs Stroke)',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [FIG] Feature distributions → {save_path.name}")


def plot_comprehensive_comparison(results, fig_dir):
    """Full model comparison bar chart — main paper figure."""
    metrics_keys = ['accuracy', 'f1', 'sensitivity', 'specificity', 'ppv', 'npv']
    metrics_labels = ['Accuracy', 'F1', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
    colors = [
        '#667eea', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
        '#1abc9c', '#e67e22', '#3498db', '#e91e63', '#00bcd4'
    ]

    for fs_name in results['models']:
        loocv = results['models'][fs_name].get('LOOCV', {})
        if not loocv:
            continue

        model_names = list(loocv.keys())
        n_m = len(model_names)
        n_met = len(metrics_keys)

        fig, ax = plt.subplots(figsize=(18, 7))
        x = np.arange(n_met)
        w = 0.8 / n_m

        for i, mname in enumerate(model_names):
            vals = [loocv[mname].get(k, 0) for k in metrics_keys]
            offset = (i - n_m / 2 + 0.5) * w
            ax.bar(x + offset, vals, w * 0.9,
                   label=mname, color=colors[i % len(colors)], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(metrics_labels, fontsize=11)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(0, 1.15)
        ax.set_title(
            f'Model Comparison — {fs_name}, LOOCV',
            fontsize=14, fontweight='bold'
        )
        ax.legend(loc='upper right', fontsize=7, ncol=2)
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()

        tag = 'eeg' if '12' in fs_name else 'all'
        path = fig_dir / f"full_comparison_{tag}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [FIG] Full comparison ({tag}) → {path.name}")


def plot_cv_comparison(results, fig_dir):
    """Compare accuracy across CV methods for key models."""
    key_models = ['ESN', 'E-ESN (7)', 'SVM (RBF)', 'Random Forest', 'KNN (k=3)']
    cv_names = ['LOOCV', '5-Fold', '10-Fold']
    cv_colors = ['#667eea', '#2ecc71', '#f39c12']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, fs_name in enumerate(results['models']):
        ax = axes[ax_idx]
        x = np.arange(len(key_models))
        w = 0.25

        for i, cv in enumerate(cv_names):
            data = results['models'][fs_name].get(cv, {})
            vals = [data.get(m, {}).get('accuracy', 0) for m in key_models]
            ax.bar(x + (i - 1) * w, vals, w * 0.9,
                   label=cv, color=cv_colors[i], alpha=0.85)

            for j, v in enumerate(vals):
                if v > 0:
                    ax.text(x[j] + (i - 1) * w, v + 0.01, f'{v:.0%}',
                           ha='center', fontsize=7, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(key_models, fontsize=9, rotation=12)
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1.15)
        ax.set_title(fs_name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle(
        'Cross-Validation Method Comparison',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    path = fig_dir / "cv_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [FIG] CV comparison → {path.name}")


def plot_feature_selection_comparison(fs_results, fig_dir):
    """Plot feature selection experiment results."""
    k_labels = list(fs_results.keys())
    esn_acc = [fs_results[k]['ESN']['accuracy'] for k in k_labels]
    eesn_acc = [fs_results[k]['E-ESN']['accuracy'] for k in k_labels]
    esn_f1 = [fs_results[k]['ESN']['f1'] for k in k_labels]
    eesn_f1 = [fs_results[k]['E-ESN']['f1'] for k in k_labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(k_labels))
    w = 0.35

    # Accuracy
    ax1.bar(x - w / 2, esn_acc, w, label='ESN', color='#667eea', alpha=0.85)
    ax1.bar(x + w / 2, eesn_acc, w, label='E-ESN', color='#ff6b6b', alpha=0.85)
    for i in range(len(k_labels)):
        ax1.text(x[i] - w / 2, esn_acc[i] + 0.01, f'{esn_acc[i]:.1%}',
                ha='center', fontsize=9, fontweight='bold')
        ax1.text(x[i] + w / 2, eesn_acc[i] + 0.01, f'{eesn_acc[i]:.1%}',
                ha='center', fontsize=9, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(k_labels, fontsize=10)
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1.15)
    ax1.set_title('Accuracy by Feature Count', fontweight='bold')
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)

    # F1
    ax2.bar(x - w / 2, esn_f1, w, label='ESN', color='#667eea', alpha=0.85)
    ax2.bar(x + w / 2, eesn_f1, w, label='E-ESN', color='#ff6b6b', alpha=0.85)
    for i in range(len(k_labels)):
        ax2.text(x[i] - w / 2, esn_f1[i] + 0.01, f'{esn_f1[i]:.1%}',
                ha='center', fontsize=9, fontweight='bold')
        ax2.text(x[i] + w / 2, eesn_f1[i] + 0.01, f'{eesn_f1[i]:.1%}',
                ha='center', fontsize=9, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(k_labels, fontsize=10)
    ax2.set_ylabel('F1 Score')
    ax2.set_ylim(0, 1.15)
    ax2.set_title('F1 Score by Feature Count', fontweight='bold')
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)

    plt.suptitle(
        'Feature Selection Experiment (LOOCV)',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    path = fig_dir / "feature_selection_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [FIG] Feature selection comparison → {path.name}")


def print_summary_table(results):
    """Print a comprehensive summary table matching paper format."""
    SEP = "─" * 120

    print(f"\n{'═' * 120}")
    print("COMPREHENSIVE RESULTS — Paper Table Replication")
    print(f"{'═' * 120}")

    for fs_name in results['models']:
        print(f"\n{SEP}")
        print(f"  Feature Set: {fs_name}")
        print(SEP)
        hdr = (f"{'Model':25s} │ {'CV':8s} │ {'Accuracy':>8s} │ "
               f"{'F1':>8s} │ {'Sens':>8s} │ {'Spec':>8s} │ "
               f"{'PPV':>8s} │ {'NPV':>8s}")
        print(hdr)
        print(SEP)

        for cv_name in ['LOOCV', '5-Fold', '10-Fold']:
            cv_data = results['models'][fs_name].get(cv_name, {})
            for mname, m in cv_data.items():
                print(
                    f"{mname:25s} │ {cv_name:8s} │ "
                    f"{m['accuracy']:>7.2%} │ {m['f1']:>7.2%} │ "
                    f"{m['sensitivity']:>7.2%} │ {m['specificity']:>7.2%} │ "
                    f"{m['ppv']:>7.2%} │ {m['npv']:>7.2%}"
                )
            if cv_data:
                print(f"{'':25s} │ {'':8s} │")

    if 'feature_selection' in results:
        print(f"\n{SEP}")
        print("  Feature Selection (ESN + E-ESN, LOOCV)")
        print(SEP)
        for k_label, data in results['feature_selection'].items():
            feats = data.get('features', [])
            esn = data['ESN']
            eesn = data['E-ESN']
            print(f"  {k_label}")
            print(f"    ESN  : Acc={esn['accuracy']:.2%}  F1={esn['f1']:.2%}  "
                  f"Sens={esn['sensitivity']:.2%}  Spec={esn['specificity']:.2%}")
            print(f"    E-ESN: Acc={eesn['accuracy']:.2%}  F1={eesn['f1']:.2%}  "
                  f"Sens={eesn['sensitivity']:.2%}  Spec={eesn['specificity']:.2%}")
            print(f"    Features: {', '.join(feats[:10])}")

    # Paper target
    print(f"\n{SEP}")
    print(f"  Paper Target (E-ESN, LOOCV, All features)")
    print(f"  Acc=96.50%  F1=94.73%  Sens=96.42%  Spec=81.81%  PPV=93.10%  NPV=90.00%")
    print(f"{'═' * 120}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    set_global_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {'models': {}, 'feature_selection': {}}
    t_start = time.time()

    # ──────────────────────────────────────────────────────
    # PART 1: Load data — two feature sets
    # ──────────────────────────────────────────────────────
    print("=" * 70)
    print("PART 1: Loading Data")
    print("=" * 70)

    X_eeg, y, feat_eeg, df_raw = load_and_prepare(
        feature_set='eeg_only', include_demographics=False
    )
    X_all, _, feat_all, _ = load_and_prepare(
        feature_set='all', include_demographics=True
    )

    print(f"\n  EEG-only: {X_eeg.shape[1]} features  → {feat_eeg}")
    print(f"  All:      {X_all.shape[1]} features  → {feat_all}")

    # ──────────────────────────────────────────────────────
    # PART 2: Feature Distribution Boxplots (Paper Fig. 3)
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PART 2: Feature Distribution Analysis (Paper Fig. 3)")
    print("=" * 70)

    plot_feature_distributions(
        X_all, y, feat_all,
        FIGURES_DIR / "feature_distributions.png"
    )

    # ──────────────────────────────────────────────────────
    # PART 3: All Models × All CV × All Feature Sets
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PART 3: Comprehensive Model Evaluation")
    print("=" * 70)

    feature_sets = {
        'EEG Only (12)': (X_eeg, feat_eeg),
        'All Features (15)': (X_all, feat_all),
    }
    cv_methods = [
        ('LOOCV', 'loocv', None),
        ('5-Fold', 'stratified_kfold', 5),
        ('10-Fold', 'stratified_kfold', 10),
    ]

    # Build model factory dict
    baselines = get_baseline_classifiers(SEED)
    model_factories = {}
    for name, clf in baselines.items():
        model_factories[name] = lambda c=clf: clone(c)
    model_factories['ESN'] = lambda: EchoStateNetwork(**ESN_PARAMS)
    model_factories['E-ESN (7)'] = lambda: EnsembleESN(**EESN_PARAMS)

    for fs_name, (X_raw, feat_names) in feature_sets.items():
        normalizer = FeatureNormalizer()
        X_norm = normalizer.fit_transform(X_raw)

        all_results['models'][fs_name] = {}

        for cv_label, cv_method, n_splits in cv_methods:
            print(f"\n── {fs_name} │ {cv_label} {'─' * 30}")
            all_results['models'][fs_name][cv_label] = {}

            for model_name, factory in model_factories.items():
                t0 = time.time()
                result = evaluate_model_cv(
                    factory, X_norm, y, cv_method, n_splits
                )
                dt = time.time() - t0

                all_results['models'][fs_name][cv_label][model_name] = result
                print(
                    f"  {model_name:25s}  "
                    f"Acc={result['accuracy']:.2%}  "
                    f"F1={result['f1']:.2%}  "
                    f"Sens={result['sensitivity']:.2%}  "
                    f"Spec={result['specificity']:.2%}  "
                    f"({dt:.1f}s)"
                )

    # ──────────────────────────────────────────────────────
    # PART 4: Feature Selection Experiments
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PART 4: Feature Selection Experiments (ANOVA + MI)")
    print("=" * 70)

    # Use ALL features, then select subsets
    norm_all = FeatureNormalizer()
    X_all_norm = norm_all.fit_transform(X_all)

    k_values = [4, 8, len(feat_all)]

    for k in k_values:
        k_actual = min(k, len(feat_all))
        k_label = f"k={k_actual}" if k_actual < len(feat_all) else f"k=all ({k_actual})"

        selector = FeatureSelector(method='combined', k=k_actual)
        selector.fit(X_all_norm, y, feature_names=feat_all)
        X_sel = selector.transform(X_all_norm)
        sel_names = selector.selected_names_

        # Ranking table
        ranking = selector.get_ranking_table()

        print(f"\n  {k_label}: {sel_names}")

        # Run ESN and E-ESN with LOOCV
        esn_r = evaluate_model_cv(
            lambda: EchoStateNetwork(**ESN_PARAMS),
            X_sel, y, 'loocv'
        )
        eesn_r = evaluate_model_cv(
            lambda: EnsembleESN(**EESN_PARAMS),
            X_sel, y, 'loocv'
        )

        all_results['feature_selection'][k_label] = {
            'features': sel_names,
            'k': k_actual,
            'ESN': esn_r,
            'E-ESN': eesn_r,
            'ranking': [
                {'feature': r['feature'], 'rank': r['rank'],
                 'anova_p': r['anova_p'], 'combined': r['combined']}
                for r in ranking
            ]
        }

        print(f"    ESN  : Acc={esn_r['accuracy']:.2%}  F1={esn_r['f1']:.2%}")
        print(f"    E-ESN: Acc={eesn_r['accuracy']:.2%}  F1={eesn_r['f1']:.2%}")

    # ──────────────────────────────────────────────────────
    # SAVE RESULTS
    # ──────────────────────────────────────────────────────
    results_path = RESULTS_DIR / "full_evaluation.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved → {results_path}")

    # ──────────────────────────────────────────────────────
    # GENERATE FIGURES
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PART 5: Generating Figures")
    print("=" * 70)

    plot_comprehensive_comparison(all_results, FIGURES_DIR)
    plot_cv_comparison(all_results, FIGURES_DIR)
    plot_feature_selection_comparison(
        all_results['feature_selection'], FIGURES_DIR
    )

    # ──────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────
    print_summary_table(all_results)

    elapsed = time.time() - t_start
    print(f"\n⏱  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("✅ Full evaluation complete!")


if __name__ == '__main__':
    main()
