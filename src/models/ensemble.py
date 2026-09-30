"""
Module M6: Ensemble Echo State Network (E-ESN)

Combines multiple ESN classifiers using bagging (bootstrap aggregating).

FACT FROM PAPER:
    - 7 ESNs in ensemble
    - Same hyperparameters, different random initializations
    - Bagging with bootstrap sampling
    - Aggregation: averaging predictions
    - Running time: 9.35 minutes

Architecture:
    1. Create N ESN models with different random seeds
    2. For each ESN:
       a. Create bootstrap sample from training data
       b. Train ESN on bootstrap sample
    3. For prediction:
       a. Each ESN predicts probabilities
       b. Average all probabilities
       c. Argmax for final class
"""
import numpy as np
from typing import Optional, List, Dict
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import resample

from src.models.esn import EchoStateNetwork


class EnsembleESN(BaseEstimator, ClassifierMixin):
    """
    Ensemble of Echo State Networks with Bagging.
    
    Parameters
    ----------
    n_estimators : int
        Number of ESN models in ensemble. Paper: 7.
    n_neurons : int
        Neurons per ESN. Paper: 200.
    spectral_radius : float
        Spectral radius. Paper: 0.95.
    sparsity : float
        Reservoir sparsity. Paper: 0.0 (fully connected).
    noise : float
        Ridge regularization. Paper: 0.001.
    leaking_rate : float
        Leaking rate. Not in paper, default: 1.0.
    input_scaling : float
        Input scaling. Not in paper, default: 1.0.
    aggregation : str
        'soft' (average probabilities) or 'hard' (majority vote).
    bootstrap : bool
        Whether to use bootstrap sampling. Paper: True (bagging).
    bootstrap_ratio : float
        Fraction of training data for each bootstrap sample. Default: 1.0.
    random_state : int
        Base random seed. Each ESN gets seed = random_state + i.
    """
    
    def __init__(
        self,
        n_estimators: int = 7,
        n_neurons: int = 200,
        spectral_radius: float = 0.95,
        sparsity: float = 0.0,
        noise: float = 0.001,
        leaking_rate: float = 1.0,
        input_scaling: float = 1.0,
        aggregation: str = 'soft',
        bootstrap: bool = True,
        bootstrap_ratio: float = 1.0,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.n_neurons = n_neurons
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.leaking_rate = leaking_rate
        self.input_scaling = input_scaling
        self.aggregation = aggregation
        self.bootstrap = bootstrap
        self.bootstrap_ratio = bootstrap_ratio
        self.random_state = random_state
        
        self.estimators_: List[EchoStateNetwork] = []
        self.classes_ = None
        self.is_fitted_ = False
    
    def _create_esn(self, seed: int) -> EchoStateNetwork:
        """Create a single ESN with specific random seed."""
        return EchoStateNetwork(
            n_neurons=self.n_neurons,
            spectral_radius=self.spectral_radius,
            sparsity=self.sparsity,
            noise=self.noise,
            leaking_rate=self.leaking_rate,
            input_scaling=self.input_scaling,
            random_state=seed,
            washout=0
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleESN':
        """
        Train the E-ESN ensemble.
        
        For each ESN:
        1. Create bootstrap sample (if bootstrap=True)
        2. Train ESN on the sample
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        
        Returns:
            self
        """
        self.classes_ = np.unique(y)
        self.estimators_ = []
        
        n_samples = X.shape[0]
        bootstrap_size = int(n_samples * self.bootstrap_ratio)
        
        rng = np.random.RandomState(self.random_state)
        
        for i in range(self.n_estimators):
            # Create ESN with unique seed
            esn_seed = self.random_state + i + 1
            esn = self._create_esn(esn_seed)
            
            # Bootstrap sampling
            if self.bootstrap:
                boot_indices = rng.choice(n_samples, size=bootstrap_size, replace=True)
                X_boot = X[boot_indices]
                y_boot = y[boot_indices]
            else:
                X_boot = X
                y_boot = y
            
            # Train
            esn.fit(X_boot, y_boot)
            self.estimators_.append(esn)
        
        self.is_fitted_ = True
        print(f"[E-ESN] Trained {self.n_estimators} ESN models "
              f"(neurons={self.n_neurons}, sr={self.spectral_radius}, "
              f"bootstrap={'on' if self.bootstrap else 'off'})")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities by averaging across all ESNs.
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            proba: Averaged class probabilities (n_samples, n_classes)
        """
        if not self.is_fitted_:
            raise RuntimeError("E-ESN not fitted. Call fit() first.")
        
        # Collect predictions from all ESNs
        all_proba = []
        for esn in self.estimators_:
            proba = esn.predict_proba(X)
            all_proba.append(proba)
        
        # Stack: (n_estimators, n_samples, n_classes)
        all_proba = np.array(all_proba)
        
        if self.aggregation == 'soft':
            # Soft voting: average probabilities
            avg_proba = np.mean(all_proba, axis=0)
        else:
            # Hard voting: majority vote (less common)
            predictions = np.argmax(all_proba, axis=2)  # (n_estimators, n_samples)
            avg_proba = np.zeros((X.shape[0], len(self.classes_)))
            for i in range(X.shape[0]):
                for cls_idx in range(len(self.classes_)):
                    avg_proba[i, cls_idx] = np.mean(predictions[:, i] == cls_idx)
        
        return avg_proba
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            y_pred: Predicted labels (n_samples,)
        """
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]
    
    def get_individual_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Get predictions from each individual ESN (for analysis).
        
        Returns:
            predictions: (n_estimators, n_samples) array of predicted classes
        """
        predictions = []
        for esn in self.estimators_:
            y_pred = esn.predict(X)
            predictions.append(y_pred)
        return np.array(predictions)
    
    def get_individual_accuracies(self, X: np.ndarray, y: np.ndarray) -> List[float]:
        """
        Get accuracy of each individual ESN (for analysis).
        
        Returns:
            List of accuracies per ESN
        """
        from sklearn.metrics import accuracy_score
        accuracies = []
        for esn in self.estimators_:
            y_pred = esn.predict(X)
            acc = accuracy_score(y, y_pred)
            accuracies.append(acc)
        return accuracies
    
    def __repr__(self) -> str:
        return (
            f"EnsembleESN(n_estimators={self.n_estimators}, "
            f"n_neurons={self.n_neurons}, "
            f"sr={self.spectral_radius}, "
            f"noise={self.noise})"
        )
