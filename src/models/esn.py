"""
Module M5: Echo State Network (ESN) Implementation

Custom ESN classifier built from scratch for full control over all parameters.

FACT FROM PAPER (Table 1):
    - Neurons: 200
    - Spectral radius: 0.95
    - Sparsity: 0 (fully connected)
    - Noise: 0.001 (regularization for ridge regression)
    
NOT SPECIFIED IN PAPER (need tuning):
    - Leaking rate (default: 1.0)
    - Input scaling (default: 1.0)
    - Activation function (inference: tanh)
    - Wout solver (inference: ridge regression)

ESN Theory:
    State update:  x(t) = (1-lr) * x(t-1) + lr * tanh(Win @ u(t) + W @ x(t-1))
    Output:        y(t) = Wout @ [x(t); 1]   (with bias)
    Training:      Wout = Y @ X_ext^T @ (X_ext @ X_ext^T + alpha*I)^(-1)
"""
import numpy as np
from typing import Optional, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin


class EchoStateNetwork(BaseEstimator, ClassifierMixin):
    """
    Echo State Network (ESN) classifier.
    
    Implements sklearn interface (fit/predict/predict_proba) for compatibility
    with sklearn tools (cross-validation, bagging, etc.).
    
    Parameters
    ----------
    n_neurons : int
        Number of reservoir neurons. Paper: 200.
    spectral_radius : float
        Spectral radius of reservoir weight matrix. Paper: 0.95.
        Must be < 1 for echo state property.
    sparsity : float
        Fraction of zero connections in reservoir. Paper: 0 (fully connected).
        0 = fully connected, 0.9 = 90% sparse.
    noise : float
        Ridge regression regularization parameter. Paper: 0.001.
    leaking_rate : float
        Leaking rate for state update. Not in paper, default: 1.0.
        1.0 = no leaking (standard ESN), <1.0 = leaky integrator ESN.
    input_scaling : float
        Scaling factor for input weights. Not in paper, default: 1.0.
    random_state : int or None
        Random seed for reproducibility.
    washout : int
        Number of initial time steps to discard (transient). Default: 0.
        For tabular data (1 sample = 1 time step), usually 0.
    """
    
    def __init__(
        self,
        n_neurons: int = 200,
        spectral_radius: float = 0.95,
        sparsity: float = 0.0,
        noise: float = 0.001,
        leaking_rate: float = 1.0,
        input_scaling: float = 1.0,
        random_state: Optional[int] = None,
        washout: int = 0
    ):
        self.n_neurons = n_neurons
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.leaking_rate = leaking_rate
        self.input_scaling = input_scaling
        self.random_state = random_state
        self.washout = washout
        
        # Will be initialized in _initialize_weights
        self.Win_ = None    # Input weights (n_neurons, n_features+1)
        self.W_ = None      # Reservoir weights (n_neurons, n_neurons)
        self.Wout_ = None   # Output weights (n_classes, n_neurons+1)
        self.classes_ = None
        self.n_features_ = None
        self.is_fitted_ = False
    
    def _initialize_weights(self, n_features: int) -> None:
        """
        Initialize input weights (Win) and reservoir weights (W).
        
        Win: random uniform [-input_scaling, input_scaling]
        W: random, scaled to desired spectral radius
        """
        rng = np.random.RandomState(self.random_state)
        
        self.n_features_ = n_features
        
        # Input weights: (n_neurons, n_features + 1 bias)
        self.Win_ = rng.uniform(
            -self.input_scaling, self.input_scaling,
            size=(self.n_neurons, n_features + 1)
        )
        
        # Reservoir weights: (n_neurons, n_neurons)
        W = rng.randn(self.n_neurons, self.n_neurons)
        
        # Apply sparsity mask
        if self.sparsity > 0:
            mask = rng.rand(self.n_neurons, self.n_neurons) > self.sparsity
            W *= mask
        
        # Scale to desired spectral radius
        # Spectral radius = max absolute eigenvalue
        eigenvalues = np.abs(np.linalg.eigvals(W))
        current_sr = np.max(eigenvalues)
        
        if current_sr > 0:
            W = W * (self.spectral_radius / current_sr)
        
        self.W_ = W
    
    def _compute_reservoir_states(self, X: np.ndarray) -> np.ndarray:
        """
        Run input through the reservoir and collect states.
        
        For TABULAR data: each sample is INDEPENDENT.
        We reset the reservoir state for each sample and run multiple
        iterations to create a rich nonlinear projection.
        
        State update equation (per iteration):
            x(t) = (1 - lr) * x(t-1) + lr * tanh(Win @ [u; 1] + W @ x(t-1))
        
        The input u is the same for all iterations (it's the feature vector).
        Multiple iterations let the reservoir settle into a fixed point that
        depends nonlinearly on the input — this is the key to ESN for tabular data.
        
        Args:
            X: Input data (n_samples, n_features)
        
        Returns:
            states: Reservoir states (n_samples, n_neurons)
        """
        n_samples = X.shape[0]
        states = np.zeros((n_samples, self.n_neurons))
        n_iterations = 10  # iterations per sample for reservoir to settle
        
        for i in range(n_samples):
            # Reset state for each independent sample
            x = np.zeros(self.n_neurons)
            
            # Input with bias: [u, 1]
            u = np.concatenate([X[i], [1.0]])
            
            # Run multiple iterations with the SAME input
            # to let the reservoir settle into a nonlinear fixed point
            for _ in range(n_iterations):
                x_new = np.tanh(self.Win_ @ u + self.W_ @ x)
                x = (1 - self.leaking_rate) * x + self.leaking_rate * x_new
            
            states[i] = x
        
        return states
    
    def _solve_readout(
        self, states: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """
        Solve for output weights using ridge regression.
        
        Wout = Y @ S_ext^T @ (S_ext @ S_ext^T + alpha*I)^(-1)
        
        where S_ext = [states, 1] (states with bias column)
        
        Args:
            states: Reservoir states (n_samples, n_neurons) — after washout
            y: Target labels (n_samples,) or one-hot (n_samples, n_classes)
        
        Returns:
            Wout: Output weights (n_classes, n_neurons + 1)
        """
        # Add bias column to states
        n_samples = states.shape[0]
        S_ext = np.hstack([states, np.ones((n_samples, 1))])
        
        # One-hot encode targets if needed
        if y.ndim == 1:
            n_classes = len(self.classes_)
            Y = np.zeros((n_samples, n_classes))
            for i, cls in enumerate(self.classes_):
                Y[y == cls, i] = 1.0
        else:
            Y = y
        
        # Ridge regression: Wout = (S^T S + alpha*I)^(-1) S^T Y
        n_ext = S_ext.shape[1]
        reg = self.noise * np.eye(n_ext)
        
        Wout = np.linalg.solve(
            S_ext.T @ S_ext + reg,
            S_ext.T @ Y
        ).T  # (n_classes, n_neurons + 1)
        
        return Wout
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EchoStateNetwork':
        """
        Train the ESN classifier.
        
        1. Initialize reservoir weights (if not already done)
        2. Run data through reservoir to collect states
        3. Solve for output weights via ridge regression
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        
        Returns:
            self
        """
        # Store classes
        self.classes_ = np.unique(y)
        
        # Initialize weights
        self._initialize_weights(X.shape[1])
        
        # Compute reservoir states
        states = self._compute_reservoir_states(X)
        
        # Remove washout period
        if self.washout > 0:
            states = states[self.washout:]
            y = y[self.washout:]
        
        # Solve for output weights
        self.Wout_ = self._solve_readout(states, y)
        
        self.is_fitted_ = True
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Uses softmax on raw output to get probabilities.
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            proba: Class probabilities (n_samples, n_classes)
        """
        if not self.is_fitted_:
            raise RuntimeError("ESN not fitted. Call fit() first.")
        
        # Compute reservoir states
        states = self._compute_reservoir_states(X)
        
        # Add bias
        n_samples = states.shape[0]
        S_ext = np.hstack([states, np.ones((n_samples, 1))])
        
        # Raw output
        raw_output = S_ext @ self.Wout_.T  # (n_samples, n_classes)
        
        # Softmax for probabilities
        exp_output = np.exp(raw_output - np.max(raw_output, axis=1, keepdims=True))
        proba = exp_output / np.sum(exp_output, axis=1, keepdims=True)
        
        return proba
    
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
    
    def get_reservoir_states(self, X: np.ndarray) -> np.ndarray:
        """
        Get reservoir states for analysis/visualization.
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            states: Reservoir states (n_samples, n_neurons)
        """
        if not self.is_fitted_:
            raise RuntimeError("ESN not fitted. Call fit() first.")
        return self._compute_reservoir_states(X)
    
    def __repr__(self) -> str:
        return (
            f"EchoStateNetwork(n_neurons={self.n_neurons}, "
            f"sr={self.spectral_radius}, "
            f"sparsity={self.sparsity}, "
            f"noise={self.noise}, "
            f"lr={self.leaking_rate}, "
            f"is={self.input_scaling})"
        )
