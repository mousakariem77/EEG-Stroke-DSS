"""
Module M4: Feature Normalization

Normalizes features to [0, 1] range using MinMaxScaler.
Paper specifies: "normalize to [0, 1]"
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional


class FeatureNormalizer:
    """
    MinMax feature normalizer — scales all features to [0, 1].
    
    FACT FROM PAPER: Features are normalized to [0, 1] range.
    """
    
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'FeatureNormalizer':
        """
        Fit the scaler on training data.
        
        Args:
            X: Training feature matrix (n_samples, n_features)
        
        Returns:
            self
        """
        self.scaler.fit(X)
        self.is_fitted = True
        print(f"[NORM] Fitted MinMaxScaler on {X.shape[0]} samples, {X.shape[1]} features")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features to [0, 1] range.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            np.ndarray: Normalized features
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        
        X_norm = self.scaler.transform(X)
        return X_norm
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_norm: np.ndarray) -> np.ndarray:
        """Convert normalized features back to original scale."""
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        return self.scaler.inverse_transform(X_norm)
