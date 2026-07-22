"""Deployment score computation.

Composite score combining accuracy, robustness, generalization, and cost.
All components are min-max normalized across models.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from utils.metrics import compute_deployment_score, normalize_scores


def compute_all_deployment_scores(
    model_results: Dict[str, Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute deployment scores for all models.

    Args:
        model_results: Dictionary mapping model name to results dict.
            Each results dict should contain:
                - 'accuracy': In-domain accuracy
                - 'robustness_score': Mean accuracy under perturbations
                - 'generalization_score': Mean cross-dataset accuracy
                - 'inference_time_ms': Inference latency
        weights: Optional deployment score weights.

    Returns:
        Dictionary mapping model name to deployment analysis.
    """
    if weights is None:
        weights = {
            "accuracy_weight": 0.30,
            "robustness_weight": 0.25,
            "generalization_weight": 0.25,
            "cost_weight": 0.20,
        }

    # Extract raw scores
    accuracies = {}
    robustness = {}
    generalization = {}
    costs = {}  # Using inference time as cost proxy

    for model_name, results in model_results.items():
        accuracies[model_name] = results.get("accuracy", 0.0)
        robustness[model_name] = results.get("robustness_score", 0.0)
        generalization[model_name] = results.get("generalization_score", 0.0)
        costs[model_name] = results.get("inference_time_ms", 1.0)

    # Normalize all scores to [0, 1]
    norm_accuracy = normalize_scores(accuracies)
    norm_robustness = normalize_scores(robustness)
    norm_generalization = normalize_scores(generalization)
    norm_cost = normalize_scores(costs)

    # Compute deployment scores
    deployment_scores = {}
    for model_name in model_results:
        score = compute_deployment_score(
            accuracy=norm_accuracy[model_name],
            robustness=norm_robustness[model_name],
            generalization=norm_generalization[model_name],
            computational_cost=norm_cost[model_name],
            weights=weights,
        )

        deployment_scores[model_name] = {
            "deployment_score": score,
            "raw_accuracy": accuracies[model_name],
            "raw_robustness": robustness[model_name],
            "raw_generalization": generalization[model_name],
            "raw_cost_ms": costs[model_name],
            "norm_accuracy": norm_accuracy[model_name],
            "norm_robustness": norm_robustness[model_name],
            "norm_generalization": norm_generalization[model_name],
            "norm_cost": norm_cost[model_name],
        }

    # Rank models
    ranked = sorted(
        deployment_scores.items(),
        key=lambda x: x[1]["deployment_score"],
        reverse=True,
    )
    for rank, (model_name, _) in enumerate(ranked, 1):
        deployment_scores[model_name]["rank"] = rank

    return deployment_scores


def save_deployment_scores(
    scores: Dict[str, Dict[str, Any]],
    save_dir: str,
    weights: Dict[str, float],
) -> str:
    """Save deployment scores to JSON."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    filepath = save_path / "deployment_scores.json"
    output = {
        "weights": weights,
        "scores": scores,
    }
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)

    return str(filepath)
