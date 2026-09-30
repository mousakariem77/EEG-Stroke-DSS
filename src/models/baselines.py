"""
Module M5 (part): Baseline classifiers.

Train simple classifiers first to:
1. Validate the data pipeline produces discriminative features
2. Establish a performance baseline for ESN comparison
3. Cross-check with related works in paper's Table 3
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from typing import Dict, Tuple, Optional


def get_baseline_classifiers(random_state: int = 42) -> Dict:
    """
    Get a dictionary of baseline classifiers for comparison.
    
    Returns:
        dict: {name: classifier} mapping
    """
    return {
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=random_state),
        'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=random_state),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=random_state
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'Naive Bayes': GaussianNB(),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute classification metrics matching paper's Table 2.
    
    Paper reports: Accuracy, PPV, NPV, Sensitivity, Specificity, F1-score
    
    For binary classification (0=control, 1=stroke):
        - TP = stroke correctly predicted as stroke
        - TN = control correctly predicted as control
        - FP = control incorrectly predicted as stroke
        - FN = stroke incorrectly predicted as control
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        dict: All metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # For binary: cm[0,0]=TN, cm[0,1]=FP, cm[1,0]=FN, cm[1,1]=TP
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Edge case: only one class predicted
        tn = fp = fn = tp = 0
        for i in range(len(y_true)):
            if y_true[i] == 1 and y_pred[i] == 1:
                tp += 1
            elif y_true[i] == 0 and y_pred[i] == 0:
                tn += 1
            elif y_true[i] == 0 and y_pred[i] == 1:
                fp += 1
            elif y_true[i] == 1 and y_pred[i] == 0:
                fn += 1
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    # PPV (Positive Predictive Value) = Precision
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # NPV (Negative Predictive Value)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # Sensitivity = Recall = TPR
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Specificity = TNR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # F1-score
    f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'ppv': ppv,
        'npv': npv,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'confusion_matrix': cm
    }


def evaluate_baselines_cv(
    X: np.ndarray,
    y: np.ndarray,
    cv_generator,
    classifiers: Dict = None,
    random_state: int = 42
) -> Dict:
    """
    Evaluate all baseline classifiers using cross-validation.
    
    Args:
        X: Feature matrix (normalized)
        y: Label vector
        cv_generator: Generator yielding (X_train, X_test, y_train, y_test, fold_idx)
        classifiers: Dict of classifiers. If None, use defaults.
        random_state: Random seed
    
    Returns:
        dict: {model_name: {metric: [values_per_fold]}}
    """
    if classifiers is None:
        classifiers = get_baseline_classifiers(random_state)
    
    # Collect all folds first
    folds = list(cv_generator)
    n_folds = len(folds)
    
    results = {}
    
    for name, clf in classifiers.items():
        fold_metrics = []
        
        for X_train, X_test, y_train, y_test, fold_idx in folds:
            # Clone the classifier for each fold
            from sklearn.base import clone
            clf_fold = clone(clf)
            
            # Train
            clf_fold.fit(X_train, y_train)
            
            # Predict
            y_pred = clf_fold.predict(X_test)
            
            # Compute metrics
            metrics = compute_metrics(y_test, y_pred)
            fold_metrics.append(metrics)
        
        # Aggregate across folds
        metric_names = ['accuracy', 'ppv', 'npv', 'sensitivity', 'specificity', 'f1']
        results[name] = {}
        for m in metric_names:
            values = [fm[m] for fm in fold_metrics]
            results[name][m] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
        
        acc_mean = results[name]['accuracy']['mean']
        f1_mean = results[name]['f1']['mean']
        print(f"  {name:25s}: Accuracy={acc_mean:.4f} (+/-{results[name]['accuracy']['std']:.4f}), "
              f"F1={f1_mean:.4f}")
    
    return results


def print_results_table(results: Dict) -> None:
    """Print results in a formatted table."""
    print(f"\n{'Model':25s} | {'Accuracy':>10s} | {'PPV':>10s} | {'NPV':>10s} | "
          f"{'Sensitivity':>12s} | {'Specificity':>12s} | {'F1':>10s}")
    print("-" * 100)
    
    for name, metrics in results.items():
        acc = metrics['accuracy']['mean']
        ppv = metrics['ppv']['mean']
        npv = metrics['npv']['mean']
        sens = metrics['sensitivity']['mean']
        spec = metrics['specificity']['mean']
        f1 = metrics['f1']['mean']
        
        print(f"{name:25s} | {acc:>9.2%} | {ppv:>9.2%} | {npv:>9.2%} | "
              f"{sens:>11.2%} | {spec:>11.2%} | {f1:>9.2%}")
    
    # Paper's target
    print("-" * 100)
    print(f"{'Paper (E-ESN) target':25s} | {'96.50%':>10s} | {'93.10%':>10s} | {'90.00%':>10s} | "
          f"{'96.42%':>12s} | {'81.81%':>12s} | {'94.73%':>10s}")
