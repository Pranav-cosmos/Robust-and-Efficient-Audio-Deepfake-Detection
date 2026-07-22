"""Cross-dataset evaluation for generalization analysis.

Evaluates models trained on ASVspoof 2019 against:
    - ASVspoof 2021 LA
    - ASVspoof 2021 DF
    - WaveFake
"""
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import compute_classification_metrics
from evaluation.evaluate import evaluate_deep_model, evaluate_classical_model


def cross_dataset_evaluation(
    model: nn.Module,
    test_loaders: Dict[str, DataLoader],
    device: torch.device,
    prepare_fn=None,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate a deep model across multiple test datasets.

    Args:
        model: Trained model.
        test_loaders: Dict mapping dataset name to DataLoader.
        device: Device.
        prepare_fn: Optional batch preparation function.

    Returns:
        Dictionary mapping dataset name to results.
    """
    model.eval()
    results = {}

    for dataset_name, loader in test_loaders.items():
        print(f"\nEvaluating on {dataset_name}...")
        result = evaluate_deep_model(model, loader, device, prepare_fn)
        results[dataset_name] = result

        metrics = result["metrics"]
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  EER: {metrics.get('eer', 'N/A')}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  ROC-AUC: {metrics.get('roc_auc', 'N/A')}")

    return results


def cross_dataset_evaluation_classical(
    model,
    test_features: Dict[str, tuple],
) -> Dict[str, Dict[str, Any]]:
    """Evaluate classical model across multiple test feature sets.

    Args:
        model: Trained classical model.
        test_features: Dict mapping dataset name to (X_test, y_test) tuples.

    Returns:
        Dictionary mapping dataset name to results.
    """
    results = {}

    for dataset_name, (X_test, y_test) in test_features.items():
        print(f"\nEvaluating on {dataset_name}...")
        result = evaluate_classical_model(model, X_test, y_test)
        results[dataset_name] = result

        metrics = result["metrics"]
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  EER: {metrics.get('eer', 'N/A')}")

    return results


def compute_generalization_gap(
    in_domain_results: Dict[str, float],
    cross_domain_results: Dict[str, Dict[str, float]],
    metric: str = "accuracy",
) -> Dict[str, Any]:
    """Compute generalization gap between in-domain and cross-domain.

    Args:
        in_domain_results: In-domain metrics.
        cross_domain_results: Dict of cross-domain results.
        metric: Metric to use for gap computation.

    Returns:
        Generalization analysis dictionary.
    """
    in_domain_score = in_domain_results.get(metric, 0.0)

    gaps = {}
    for dataset_name, results in cross_domain_results.items():
        cross_score = results.get("metrics", {}).get(metric, 0.0)
        gap = in_domain_score - cross_score
        relative_gap = gap / in_domain_score if in_domain_score > 0 else 0.0

        gaps[dataset_name] = {
            "in_domain_score": in_domain_score,
            "cross_domain_score": cross_score,
            "absolute_gap": gap,
            "relative_gap": relative_gap,
        }

    # Overall generalization score (mean cross-domain accuracy)
    mean_cross = np.mean([
        r.get("metrics", {}).get(metric, 0.0) for r in cross_domain_results.values()
    ])

    return {
        "per_dataset_gaps": gaps,
        "mean_cross_domain_score": float(mean_cross),
        "mean_generalization_gap": float(in_domain_score - mean_cross),
    }


def save_cross_dataset_results(
    results: Dict[str, Dict[str, Any]],
    save_dir: str,
    model_name: str,
    seed: int,
) -> str:
    """Save cross-dataset evaluation results."""
    save_path = Path(save_dir) / model_name / f"seed_{seed}"
    save_path.mkdir(parents=True, exist_ok=True)

    filepath = save_path / "cross_dataset_results.json"

    # Convert numpy arrays to lists for JSON serialization
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {
                "metrics": v.get("metrics", {}),
                "n_samples": v.get("n_samples", 0),
            }

    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    return str(filepath)
