"""Comprehensive evaluation metrics for audio deepfake detection."""
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from typing import Any, Dict, Optional, Tuple


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER).

    EER is the point where FAR == FRR on the ROC curve.

    Args:
        y_true: Ground truth binary labels (0 = bonafide, 1 = spoof).
        y_scores: Predicted scores (higher = more likely spoof).

    Returns:
        Tuple of (EER, threshold at EER).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr

    # Find the threshold where FPR == FNR
    try:
        eer_fn = interp1d(fpr, fnr)
        eer_threshold_fn = interp1d(fpr, thresholds)
        eer = brentq(lambda x: eer_fn(x) - x, 0.0, 1.0)
        eer_threshold = float(eer_threshold_fn(eer))
    except ValueError:
        # Fallback: find closest point
        abs_diff = np.abs(fpr - fnr)
        idx = np.argmin(abs_diff)
        eer = float(np.mean([fpr[idx], fnr[idx]]))
        eer_threshold = float(thresholds[idx])

    return eer, eer_threshold


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute all classification metrics.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        y_scores: Predicted scores/probabilities (for ROC-AUC and EER).

    Returns:
        Dictionary of metric name -> value.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }

    if y_scores is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_scores))
        except ValueError:
            metrics["roc_auc"] = 0.0

        eer, eer_threshold = compute_eer(y_true, y_scores)
        metrics["eer"] = eer
        metrics["eer_threshold"] = eer_threshold

    return metrics


def compute_confusion_matrix_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Compute confusion matrix and derived metrics.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Dictionary containing confusion matrix and derived rates.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "confusion_matrix": cm.tolist(),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
    }


def compute_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute ROC curve data.

    Args:
        y_true: Ground truth binary labels.
        y_scores: Predicted scores.

    Returns:
        Dictionary with fpr, tpr, thresholds arrays.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}


def compute_pr_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute Precision-Recall curve data.

    Args:
        y_true: Ground truth binary labels.
        y_scores: Predicted scores.

    Returns:
        Dictionary with precision, recall, thresholds arrays.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    return {"precision": precision, "recall": recall, "thresholds": thresholds}


def compute_robustness_score(
    clean_accuracy: float,
    perturbed_accuracies: Dict[str, float],
) -> Dict[str, float]:
    """Compute robustness score from clean and perturbed accuracies.

    Args:
        clean_accuracy: Model accuracy on clean data.
        perturbed_accuracies: Dict mapping perturbation name to accuracy.

    Returns:
        Dictionary with robustness metrics.
    """
    if not perturbed_accuracies:
        return {"robustness_score": 1.0, "mean_degradation": 0.0}

    degradations = {}
    for name, acc in perturbed_accuracies.items():
        if clean_accuracy > 0:
            degradations[name] = (clean_accuracy - acc) / clean_accuracy
        else:
            degradations[name] = 0.0

    mean_perturbed_acc = float(np.mean(list(perturbed_accuracies.values())))
    mean_degradation = float(np.mean(list(degradations.values())))

    return {
        "robustness_score": mean_perturbed_acc,
        "mean_degradation": mean_degradation,
        "per_perturbation_degradation": degradations,
    }


def compute_deployment_score(
    accuracy: float,
    robustness: float,
    generalization: float,
    computational_cost: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Compute composite deployment score.

    D = w_a * accuracy + w_r * robustness + w_g * generalization + w_c * (1 - cost)

    All inputs should be normalized to [0, 1].

    Args:
        accuracy: Normalized accuracy score.
        robustness: Normalized robustness score.
        generalization: Normalized generalization score (cross-dataset).
        computational_cost: Normalized computational cost (lower is better).
        weights: Optional weight dictionary with keys:
            accuracy_weight, robustness_weight, generalization_weight, cost_weight.

    Returns:
        Composite deployment score in [0, 1].
    """
    if weights is None:
        weights = {
            "accuracy_weight": 0.30,
            "robustness_weight": 0.25,
            "generalization_weight": 0.25,
            "cost_weight": 0.20,
        }

    score = (
        weights["accuracy_weight"] * accuracy
        + weights["robustness_weight"] * robustness
        + weights["generalization_weight"] * generalization
        + weights["cost_weight"] * (1.0 - computational_cost)
    )
    return float(np.clip(score, 0.0, 1.0))


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalize scores across models.

    Args:
        scores: Dictionary mapping model name to score value.

    Returns:
        Dictionary with normalized scores in [0, 1].
    """
    values = list(scores.values())
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val

    if range_val == 0:
        return {k: 1.0 for k in scores}

    return {k: (v - min_val) / range_val for k, v in scores.items()}
