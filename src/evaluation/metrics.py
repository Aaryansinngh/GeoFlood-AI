"""
src/evaluation/metrics.py

Comprehensive evaluation metrics and threshold optimization.
Reports:
- Precision, Recall, F1 score
- ROC-AUC, PR-AUC (Average Precision)
- Brier Score (probability calibration error)
- False Negative Rate (FNR = FN / (TP + FN))
- Confusion Matrix (TP, FP, TN, FN)
"""

from typing import Dict, Tuple, Any
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


def find_optimal_threshold(y_val: np.ndarray, y_val_probs: np.ndarray, metric: str = "f1") -> Tuple[float, float]:
    """
    Find optimal decision threshold solely on the VALIDATION set.
    Never evaluates or leaks the test set.
    Returns (optimal_threshold, best_metric_val).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_probs)
    
    # Avoid zero division
    denominator = precisions + recalls
    f1_scores = np.divide(2 * (precisions * recalls), denominator, out=np.zeros_like(precisions), where=denominator > 0)
    
    # Precision_recall_curve returns thresholds of length len(precisions) - 1
    if len(thresholds) == 0:
        return 0.5, 0.0

    best_idx = np.argmax(f1_scores[:-1])
    best_thresh = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    return best_thresh, best_f1


def evaluate_predictions(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Compute full evaluation metrics on predicted probabilities and binary thresholded outputs.
    """
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs, dtype=float)
    y_pred = (y_probs >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Core classification metrics
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Discrimination & probability calibration
    try:
        roc_auc = float(roc_auc_score(y_true, y_probs))
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = float(average_precision_score(y_true, y_probs))
    except ValueError:
        pr_auc = float("nan")

    brier = float(brier_score_loss(y_true, y_probs))

    # False negative rate (critical for disaster risk: missed flooded areas)
    total_positives = tp + fn
    fnr = float(fn / total_positives) if total_positives > 0 else 0.0

    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "false_negative_rate": fnr,
        "confusion_matrix": {
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn),
        },
        "sample_count": len(y_true),
        "positive_count": int(total_positives),
        "prevalence": float(total_positives / len(y_true)) if len(y_true) > 0 else 0.0,
    }


def compute_roc_pr_curves(y_true: np.ndarray, y_probs: np.ndarray, max_points: int = 100) -> Dict[str, Any]:
    """
    Compute downsampled ROC and PR curves for JSON export and Plotly rendering.
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    precision, recall, _ = precision_recall_curve(y_true, y_probs)

    # Downsample points for lightweight web rendering
    def downsample(x, y, n):
        if len(x) <= n:
            return [round(float(v), 4) for v in x], [round(float(v), 4) for v in y]
        idx = np.round(np.linspace(0, len(x) - 1, n)).astype(int)
        return [round(float(x[i]), 4) for i in idx], [round(float(y[i]), 4) for i in idx]

    roc_fpr, roc_tpr = downsample(fpr, tpr, max_points)
    pr_recall, pr_precision = downsample(recall, precision, max_points)

    return {
        "roc": {"fpr": roc_fpr, "tpr": roc_tpr},
        "pr": {"recall": pr_recall, "precision": pr_precision},
    }
