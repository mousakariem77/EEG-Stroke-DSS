"""
Module M5: Feature Selection

Implements feature selection methods as described in the paper:
- ANOVA F-test (filter method)
- Mutual Information (filter method)
- Combined ranking

FACT FROM PAPER:
    - Uses filter-based feature selection
    - Selects top-k features based on statistical tests
    - 4 optimal features identified: DTR, Theta, RP Delta, RP Alpha
    - Feature selection applied BEFORE model training
"""
import numpy as np
from typing import List, Tuple, Optional
from sklearn.feature_selection import (
    f_classif, mutual_info_classif, SelectKBest
)
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class FeatureSelector:
    """
    Feature selection using multiple filter methods.
    
    Supports:
    - ANOVA F-test: measures linear separability between classes
    - Mutual Information: captures nonlinear dependencies
    - Combined rank: average of normalized scores
    """
    
    def __init__(self, method: str = 'combined', k: int = 4):
        """
        Args:
            method: 'anova', 'mi', or 'combined'
            k: Number of top features to select
        """
        self.method = method
        self.k = k
        
        self.scores_ = None
        self.rankings_ = None
        self.selected_indices_ = None
        self.selected_names_ = None
        self.feature_names_ = None
    
    def fit(
        self, X: np.ndarray, y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> 'FeatureSelector':
        """
        Compute feature scores and rankings.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            feature_names: Optional list of feature names
        
        Returns:
            self
        """
        n_features = X.shape[1]
        self.feature_names_ = feature_names or [f"F{i}" for i in range(n_features)]
        
        # ANOVA F-test
        f_scores, f_pvalues = f_classif(X, y)
        
        # Mutual Information
        mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=3)
        
        # Store raw scores
        self.anova_scores_ = f_scores
        self.anova_pvalues_ = f_pvalues
        self.mi_scores_ = mi_scores
        
        # Normalize scores to [0, 1] for comparison
        scaler = MinMaxScaler()
        f_norm = scaler.fit_transform(f_scores.reshape(-1, 1)).ravel()
        mi_norm = scaler.fit_transform(mi_scores.reshape(-1, 1)).ravel()
        
        # Compute final scores based on method
        if self.method == 'anova':
            self.scores_ = f_scores
        elif self.method == 'mi':
            self.scores_ = mi_scores
        elif self.method == 'combined':
            self.scores_ = (f_norm + mi_norm) / 2
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Rankings (higher score = lower rank number)
        self.rankings_ = np.argsort(np.argsort(-self.scores_)) + 1
        
        # Select top-k
        self.selected_indices_ = np.argsort(-self.scores_)[:self.k]
        self.selected_names_ = [self.feature_names_[i] for i in self.selected_indices_]
        
        print(f"[FS] Method: {self.method}, k={self.k}")
        print(f"[FS] Selected features: {self.selected_names_}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Select top-k features from X."""
        if self.selected_indices_ is None:
            raise RuntimeError("Call fit() first")
        return X[:, self.selected_indices_]
    
    def fit_transform(
        self, X: np.ndarray, y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y, feature_names)
        return self.transform(X)
    
    def get_ranking_table(self) -> list:
        """
        Get feature ranking as sorted list of tuples.
        
        Returns:
            List of (rank, name, anova_score, mi_score, combined_score, p_value)
        """
        if self.scores_ is None:
            raise RuntimeError("Call fit() first")
        
        table = []
        for i in range(len(self.feature_names_)):
            table.append({
                'rank': int(self.rankings_[i]),
                'feature': self.feature_names_[i],
                'anova_f': float(self.anova_scores_[i]),
                'anova_p': float(self.anova_pvalues_[i]),
                'mi': float(self.mi_scores_[i]),
                'combined': float(self.scores_[i]),
                'selected': i in self.selected_indices_
            })
        
        table.sort(key=lambda x: x['rank'])
        return table
    
    def plot_scores(
        self, save_path: Optional[str] = None, max_display: int = 15
    ) -> None:
        """
        Plot feature selection scores comparison.
        
        Args:
            save_path: Path to save figure
            max_display: Max features to show
        """
        if self.scores_ is None:
            raise RuntimeError("Call fit() first")
        
        # Sort by combined score
        sorted_idx = np.argsort(-self.scores_)[:max_display]
        names = [self.feature_names_[i] for i in sorted_idx]
        
        # Normalize for comparison
        scaler = MinMaxScaler()
        anova_norm = scaler.fit_transform(self.anova_scores_.reshape(-1, 1)).ravel()
        mi_norm = scaler.fit_transform(self.mi_scores_.reshape(-1, 1)).ravel()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # ANOVA
        ax = axes[0]
        vals = [anova_norm[i] for i in sorted_idx]
        colors = ['#667eea' if i in self.selected_indices_ else '#c3cfe2' for i in sorted_idx]
        ax.barh(range(len(names)), vals[::-1], color=colors[::-1])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names[::-1])
        ax.set_xlabel('ANOVA F-score (normalized)')
        ax.set_title('ANOVA F-test')
        
        # MI
        ax = axes[1]
        vals = [mi_norm[i] for i in sorted_idx]
        ax.barh(range(len(names)), vals[::-1], color=colors[::-1])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names[::-1])
        ax.set_xlabel('Mutual Information (normalized)')
        ax.set_title('Mutual Information')
        
        # Combined
        ax = axes[2]
        vals = [self.scores_[i] for i in sorted_idx]
        ax.barh(range(len(names)), vals[::-1], color=colors[::-1])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names[::-1])
        ax.set_xlabel('Combined Score')
        ax.set_title(f'Combined (Top {self.k} highlighted)')
        
        plt.suptitle('Feature Selection Scores', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[FS] Plot saved to {save_path}")
        plt.close()
