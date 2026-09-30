"""
Module M1 (part): Data splitting strategies.

Paper does NOT specify the exact train/test split strategy.
We implement multiple approaches for comparison.
"""
import numpy as np
from sklearn.model_selection import (
    StratifiedKFold, 
    LeaveOneOut, 
    train_test_split,
    RepeatedStratifiedKFold
)
from typing import Tuple, Generator, Optional


def get_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple stratified train/test split.
    
    Args:
        X: Feature matrix
        y: Label vector
        test_size: Fraction for test set
        random_state: Random seed
        stratify: Whether to stratify by class
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )
    
    print(f"[SPLIT] Hold-out: train={len(y_train)} (stroke={np.sum(y_train==1)}), "
          f"test={len(y_test)} (stroke={np.sum(y_test==1)})")
    
    return X_train, X_test, y_train, y_test


def get_cv_folds(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    method: str = 'stratified_kfold'
) -> Generator:
    """
    Generate cross-validation folds.
    
    Args:
        X: Feature matrix
        y: Label vector
        n_splits: Number of folds (for k-fold methods)
        random_state: Random seed
        method: 'stratified_kfold', 'loocv', 'repeated_stratified_kfold'
    
    Yields:
        Tuples of (X_train, X_test, y_train, y_test, fold_idx)
    """
    if method == 'stratified_kfold':
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        print(f"[CV] Stratified {n_splits}-Fold CV")
    elif method == 'loocv':
        cv = LeaveOneOut()
        print(f"[CV] Leave-One-Out CV ({len(y)} folds)")
    elif method == 'repeated_stratified_kfold':
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=3, random_state=random_state)
        print(f"[CV] Repeated Stratified {n_splits}-Fold CV (3 repeats)")
    else:
        raise ValueError(f"Unknown CV method: {method}")
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        yield X_train, X_test, y_train, y_test, fold_idx
