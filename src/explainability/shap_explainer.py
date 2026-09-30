"""
Module M8: SHAP Global Explanation

Provides global model explanation using SHAP (SHapley Additive exPlanations).
Uses KernelSHAP because ESN is not tree-based (model-agnostic).

FACT FROM PAPER:
    - SHAP used for global feature importance
    - Top 8 features: DTR, Theta, RP Delta, RP Alpha, age, beta, education, gender
    - Fig. 6a: SHAP summary plot (dot plot)
    - Fig. 6b: SHAP bar plot (average impact)
"""
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from pathlib import Path


class SHAPExplainer:
    """
    SHAP-based global model explainer.
    
    Uses KernelSHAP (model-agnostic) to compute Shapley values
    for the E-ESN classifier.
    """
    
    def __init__(self, model, X_background: np.ndarray, feature_names: List[str]):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model with predict_proba method
            X_background: Background dataset for SHAP (typically training data)
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        
        # Use a summary of the background data to speed up KernelSHAP
        if len(X_background) > 50:
            self.X_background = shap.kmeans(X_background, 10)
        else:
            self.X_background = X_background
        
        # Create KernelSHAP explainer — predict stroke probability (class 1)
        def predict_stroke_proba(X):
            return self.model.predict_proba(X)[:, 1]
        
        self.explainer = shap.KernelExplainer(
            predict_stroke_proba,
            self.X_background
        )
        
        self.shap_values_ = None  # (n_samples, n_features)
    
    def compute_shap_values(
        self, X: np.ndarray, nsamples: int = 100
    ) -> np.ndarray:
        """
        Compute SHAP values for given instances.
        
        Args:
            X: Feature matrix to explain (n_samples, n_features)
            nsamples: Number of samples for KernelSHAP estimation
        
        Returns:
            shap_values: (n_samples, n_features) array
        """
        print(f"[SHAP] Computing SHAP values for {X.shape[0]} instances "
              f"(nsamples={nsamples})...")
        
        self.shap_values_ = self.explainer.shap_values(X, nsamples=nsamples)
        
        # Ensure shape is (n_samples, n_features)
        self.shap_values_ = np.array(self.shap_values_)
        if self.shap_values_.ndim == 3:
            # If shape is (n_samples, n_features, n_classes), take class 1
            self.shap_values_ = self.shap_values_[:, :, 1]
        
        print(f"[SHAP] Done. Shape: {self.shap_values_.shape}")
        return self.shap_values_
    
    def get_feature_importance(self) -> dict:
        """
        Get feature importance ranking (mean absolute SHAP value).
        
        Returns:
            dict: {feature_name: mean_abs_shap_value}, sorted descending
        """
        if self.shap_values_ is None:
            raise RuntimeError("Call compute_shap_values() first")
        
        mean_abs = np.abs(self.shap_values_).mean(axis=0)
        importance = dict(zip(self.feature_names, mean_abs))
        
        # Sort descending
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return importance
    
    def plot_summary(
        self, X: np.ndarray,
        save_path: Optional[str] = None, max_display: int = 15
    ) -> None:
        """
        Generate SHAP summary plot (beeswarm/dot plot).
        Matches paper's Fig. 6a.
        """
        if self.shap_values_ is None:
            raise RuntimeError("Call compute_shap_values() first")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values_, X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.title("SHAP Summary Plot - Feature Impact on Stroke Prediction")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[SHAP] Summary plot saved to {save_path}")
        plt.close()
    
    def plot_bar(
        self,
        save_path: Optional[str] = None, max_display: int = 15
    ) -> None:
        """
        Generate SHAP bar plot (mean absolute impact).
        Matches paper's Fig. 6b.
        """
        if self.shap_values_ is None:
            raise RuntimeError("Call compute_shap_values() first")
        
        importance = self.get_feature_importance()
        names = list(importance.keys())[:max_display]
        values = list(importance.values())[:max_display]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(names)), values[::-1], color='#1f77b4')
        plt.yticks(range(len(names)), names[::-1])
        plt.xlabel('Mean |SHAP value|')
        plt.title('SHAP Feature Importance - Average Impact on Stroke Prediction')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[SHAP] Bar plot saved to {save_path}")
        plt.close()
