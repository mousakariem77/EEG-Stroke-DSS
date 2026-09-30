"""
Module M9: LIME Local Explanation

Provides local model explanation using LIME (Local Interpretable Model-Agnostic Explanations).
Explains individual predictions by fitting a local linear model.

FACT FROM PAPER:
    - LIME used for local (per-instance) explanations
    - Fig. 7: Example explanations for 1 control + 1 stroke patient
    - Shows prediction probabilities + feature contributions
"""
import numpy as np
import lime
import lime.lime_tabular
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, List, Dict
from pathlib import Path


class LIMEExplainer:
    """
    LIME-based local model explainer.
    
    Explains individual predictions by fitting interpretable
    surrogate models in the neighborhood of each instance.
    """
    
    def __init__(
        self,
        model,
        X_train: np.ndarray,
        feature_names: List[str],
        class_names: List[str] = None
    ):
        """
        Initialize LIME explainer.
        
        Args:
            model: Trained model with predict_proba method
            X_train: Training data (for distribution reference)
            feature_names: List of feature names
            class_names: List of class names (default: ['Control', 'Stroke'])
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names or ['Control', 'Stroke']
        
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=self.class_names,
            mode='classification',
            discretize_continuous=True,
            random_state=42
        )
    
    def explain_instance(
        self,
        instance: np.ndarray,
        num_features: int = 8,
        num_samples: int = 5000
    ) -> lime.explanation.Explanation:
        """
        Generate LIME explanation for a single instance.
        
        Args:
            instance: Single feature vector (n_features,)
            num_features: Number of features to include in explanation
            num_samples: Number of perturbation samples
        
        Returns:
            lime.explanation.Explanation object
        """
        explanation = self.explainer.explain_instance(
            instance,
            self.model.predict_proba,
            num_features=num_features,
            num_samples=num_samples
        )
        
        return explanation
    
    def explain_and_save(
        self,
        instance: np.ndarray,
        save_path: str,
        true_label: int = None,
        num_features: int = 8,
        num_samples: int = 5000
    ) -> Dict:
        """
        Explain an instance and save the visualization.
        
        Args:
            instance: Single feature vector
            save_path: Path to save the figure
            true_label: True class label (for display)
            num_features: Features in explanation
            num_samples: Perturbation samples
        
        Returns:
            dict: Explanation summary
        """
        exp = self.explain_instance(instance, num_features, num_samples)
        
        # Get prediction probabilities
        proba = self.model.predict_proba(instance.reshape(1, -1))[0]
        predicted_class = self.class_names[np.argmax(proba)]
        
        # Save figure
        fig = exp.as_pyplot_figure()
        fig.set_size_inches(12, 6)
        
        title = f"LIME Explanation — Predicted: {predicted_class} "
        title += f"(P={proba[np.argmax(proba)]:.3f})"
        if true_label is not None:
            title += f" | True: {self.class_names[true_label]}"
        fig.suptitle(title, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[LIME] Saved explanation to {save_path}")
        
        # Build summary
        summary = {
            'predicted_class': predicted_class,
            'predicted_proba': proba.tolist(),
            'true_label': self.class_names[true_label] if true_label is not None else None,
            'feature_contributions': exp.as_list(),
            'intercept': exp.intercept,
        }
        
        return summary
    
    def explain_multiple(
        self,
        X: np.ndarray,
        y: np.ndarray,
        save_dir: str,
        n_per_class: int = 2,
        num_features: int = 8
    ) -> List[Dict]:
        """
        Explain multiple instances (n per class) and save visualizations.
        
        Args:
            X: Feature matrix
            y: True labels
            save_dir: Directory to save figures
            n_per_class: Number of instances per class to explain
            num_features: Features per explanation
        
        Returns:
            List of explanation summaries
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        summaries = []
        
        for class_idx, class_name in enumerate(self.class_names):
            # Get indices of this class
            indices = np.where(y == class_idx)[0]
            selected = indices[:n_per_class]
            
            for i, idx in enumerate(selected):
                save_path = save_dir / f"lime_{class_name.lower()}_{i+1}.png"
                summary = self.explain_and_save(
                    X[idx], str(save_path),
                    true_label=class_idx,
                    num_features=num_features
                )
                summaries.append(summary)
        
        return summaries
