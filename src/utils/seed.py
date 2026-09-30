"""
Random seed management for reproducibility.
Centralizes all random seed settings across libraries.
"""
import random
import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """
    Set random seed for all libraries to ensure reproducibility.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Try to set sklearn seed (if available)
    try:
        import sklearn
        # sklearn uses numpy's random state
    except ImportError:
        pass
    
    print(f"[SEED] Global random seed set to {seed}")


def get_rng(seed: int) -> np.random.RandomState:
    """
    Get a numpy RandomState instance with a specific seed.
    Useful for creating independent random streams (e.g., for each ESN in ensemble).
    
    Args:
        seed: Seed for this specific random stream
    
    Returns:
        np.random.RandomState instance
    """
    return np.random.RandomState(seed)
