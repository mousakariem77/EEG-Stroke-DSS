"""
Module M11: Evaluation & Visualization

Comprehensive model evaluation:
- ROC curves and AUC for all models
- Performance comparison charts
- Per-fold metrics with confidence intervals
- Confusion matrix visualization
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def plot_roc_curves(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Optional[str] = None,
    title: str = "ROC Curves — Model Comparison"
) -> Dict[str, float]:
    """
    Plot ROC curves for multiple models.
    
    Args:
        results: {model_name: (y_true, y_proba_positive)}
        save_path: Path to save figure
        title: Plot title
    
    Returns:
        {model_name: auc_score}
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    
    colors = ['#667eea', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
              '#1abc9c', '#e67e22', '#3498db', '#e91e63', '#00bcd4']
    
    auc_scores = {}
    
    for i, (name, (y_true, y_proba)) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        auc_scores[name] = roc_auc
        
        color = colors[i % len(colors)]
        lw = 3 if 'ESN' in name or 'E-ESN' in name else 1.8
        
        ax.plot(fpr, tpr, color=color, lw=lw,
                label=f'{name} (AUC = {roc_auc:.3f})')
    
    # Diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.500)')
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Shade optimal region
    ax.fill_between([0, 0, 1], [0, 1, 1], alpha=0.03, color='green')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[EVAL] ROC curves saved to {save_path}")
    plt.close()
    
    return auc_scores


def plot_precision_recall(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Optional[str] = None
) -> None:
    """Plot Precision-Recall curves for multiple models."""
    fig, ax = plt.subplots(figsize=(9, 7))
    
    colors = ['#667eea', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    for i, (name, (y_true, y_proba)) in enumerate(results.items()):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        
        color = colors[i % len(colors)]
        ax.plot(recall, precision, color=color, lw=2,
                label=f'{name} (AP = {pr_auc:.3f})')
    
    ax.set_xlabel('Recall', fontsize=13)
    ax.set_ylabel('Precision', fontsize=13)
    ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[EVAL] PR curves saved to {save_path}")
    plt.close()


def plot_model_comparison(
    metrics: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> None:
    """
    Bar chart comparing all models across multiple metrics.
    
    Args:
        metrics: {model_name: {metric_name: value}}
    """
    metric_names = ['accuracy', 'f1', 'sensitivity', 'specificity', 'ppv', 'npv']
    metric_labels = ['Accuracy', 'F1', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
    
    models = list(metrics.keys())
    n_metrics = len(metric_names)
    n_models = len(models)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(n_metrics)
    width = 0.8 / n_models
    
    colors = ['#667eea', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
              '#1abc9c', '#e67e22', '#3498db']
    
    for i, model_name in enumerate(models):
        values = [metrics[model_name].get(m, 0) for m in metric_names]
        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width * 0.9,
                     label=model_name, color=colors[i % len(colors)], alpha=0.85)
        
        # Value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.0%}', ha='center', va='bottom', fontsize=7,
                       fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title('Model Comparison — All Metrics', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[EVAL] Model comparison saved to {save_path}")
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix"
) -> None:
    """Plot a styled confusion matrix."""
    class_names = class_names or ['Control', 'Stroke']
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() * 0.6 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                   fontsize=28, fontweight='bold', color=color)
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12)
    ax.set_xlabel('Predicted', fontsize=13, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[EVAL] Confusion matrix saved to {save_path}")
    plt.close()
