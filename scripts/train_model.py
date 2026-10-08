"""
Model Training CLI for Explainable EEG Stroke DSS
Trains the Ensemble Echo State Network (E-ESN) model on EEG clinical features.
"""

import os
import sys
import argparse
import time
from pathlib import Path
import joblib
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_and_prepare
from src.features.normalization import FeatureNormalizer
from src.data.splitter import get_cv_folds
from src.models.ensemble import EnsembleESN
from src.models.baselines import compute_metrics
from src.utils.seed import set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train Ensemble ESN for EEG Stroke Prediction")
    parser.add_argument("--feature_set", type=str, default="eeg_only", choices=["all", "eeg_only", "paper_top"],
                        help="Feature subset to use ('eeg_only' matches the paper's 12 EEG features)")
    parser.add_argument("--n_estimators", type=int, default=7,
                        help="Number of ESN reservoir estimators (paper optimal: 7)")
    parser.add_argument("--n_neurons", type=int, default=200,
                        help="Number of reservoir neurons per estimator (paper optimal: 200)")
    parser.add_argument("--spectral_radius", type=float, default=0.95,
                        help="Reservoir spectral radius (paper optimal: 0.95)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--eval_loocv", action="store_true", default=True,
                        help="Evaluate performance with Leave-One-Out Cross-Validation")
    parser.add_argument("--output_dir", type=str, default="experiments/models",
                        help="Directory to save the trained model artifact")
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)

    print("=" * 70)
    print("      EXPLAINABLE EEG STROKE DSS - MODEL TRAINING CLI")
    print("=" * 70)
    print(f"Features:        {args.feature_set}")
    print(f"Estimators:      {args.n_estimators} ESNs (Reservoir size: {args.n_neurons}, SR: {args.spectral_radius})")
    print(f"Random Seed:     {args.seed}")

    # 1. Load Data
    X, y, feature_names, _ = load_and_prepare(feature_set=args.feature_set)
    print(f"\n[1/4] Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features.")

    # 2. Normalize
    normalizer = FeatureNormalizer()
    X_norm = normalizer.fit_transform(X)
    print("[2/4] Normalized features to [0, 1] range using MinMaxScaler.")

    # 3. LOOCV Evaluation
    if args.eval_loocv:
        print("\n[3/4] Running Leave-One-Out Cross-Validation (LOOCV)...")
        y_pred = np.zeros_like(y)
        y_proba = np.zeros((len(y), 2))
        start_time = time.time()

        for X_tr, X_te, y_tr, y_te, fi in get_cv_folds(X_norm, y, method='loocv'):
            model_fold = EnsembleESN(
                n_estimators=args.n_estimators,
                n_neurons=args.n_neurons,
                spectral_radius=args.spectral_radius,
                aggregation='soft',
                bootstrap=False,
                random_state=args.seed + fi,
            )
            model_fold.fit(X_tr, y_tr)
            y_pred[fi] = model_fold.predict(X_te)[0]
            y_proba[fi] = model_fold.predict_proba(X_te)[0]

        eval_duration = time.time() - start_time
        metrics = compute_metrics(y, y_pred)

        print("-" * 50)
        print("  LOOCV Validation Results (Paper Target Reproduction):")
        print(f"    - Accuracy:    {metrics['accuracy']:.2%}")
        print(f"    - Sensitivity: {metrics['sensitivity']:.2%}")
        print(f"    - Specificity: {metrics['specificity']:.2%}")
        print(f"    - PPV:         {metrics['ppv']:.2%}")
        print(f"    - NPV:         {metrics['npv']:.2%}")
        print(f"    - F1-Score:    {metrics['f1']:.2%}")
        print(f"    - Time:        {eval_duration:.2f}s")
        print("-" * 50)

    # 4. Train Final Model on All Data & Save
    print("\n[4/4] Fitting final production model on all samples...")
    final_model = EnsembleESN(
        n_estimators=args.n_estimators,
        n_neurons=args.n_neurons,
        spectral_radius=args.spectral_radius,
        aggregation='soft',
        bootstrap=False,
        random_state=args.seed,
    )
    final_model.fit(X_norm, y)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_artifact = {
        'model': final_model,
        'normalizer': normalizer,
        'feature_names': feature_names,
        'params': {
            'n_estimators': args.n_estimators,
            'n_neurons': args.n_neurons,
            'spectral_radius': args.spectral_radius,
            'feature_set': args.feature_set,
            'seed': args.seed,
        }
    }

    model_path = output_dir / "eesn_production_model.joblib"
    joblib.dump(model_artifact, model_path)
    print(f" Saved trained model bundle to: {model_path}")
    print("=" * 70)
    print("Training successfully completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
