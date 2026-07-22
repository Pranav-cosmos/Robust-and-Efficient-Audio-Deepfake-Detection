"""Core evaluation engine for audio deepfake detection models.

Computes all classification metrics, generates confusion matrices,
ROC/PR curves, and saves results.

Usage:
    python evaluation/evaluate.py --model resnet18 --test_set asvspoof2019_eval --seed 42
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config, resolve_model_config
from utils.seed import set_seed, get_device
from utils.logger import setup_logger
from utils.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix_metrics,
    compute_roc_curve,
    compute_pr_curve,
)


def evaluate_deep_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    prepare_fn=None,
) -> Dict[str, Any]:
    """Evaluate a PyTorch deep learning model.

    Args:
        model: Model in eval mode.
        dataloader: Test DataLoader.
        device: Device.
        prepare_fn: Optional function to prepare batch inputs.

    Returns:
        Dictionary with all metrics and curve data.
    """
    model.eval()
    all_labels = []
    all_scores = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if prepare_fn:
                inputs, labels = prepare_fn(batch)
            else:
                waveforms, labels, _ = batch
                inputs = waveforms.to(device)
                labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            scores = probs[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    y_true = np.array(all_labels)
    y_scores = np.array(all_scores)
    y_pred = np.array(all_preds)

    # Compute all metrics
    metrics = compute_classification_metrics(y_true, y_pred, y_scores)
    cm_metrics = compute_confusion_matrix_metrics(y_true, y_pred)
    roc_data = compute_roc_curve(y_true, y_scores)
    pr_data = compute_pr_curve(y_true, y_scores)

    results = {
        "metrics": metrics,
        "confusion_matrix": cm_metrics,
        "roc_curve": {
            "fpr": roc_data["fpr"].tolist(),
            "tpr": roc_data["tpr"].tolist(),
        },
        "pr_curve": {
            "precision": pr_data["precision"].tolist(),
            "recall": pr_data["recall"].tolist(),
        },
        "n_samples": len(y_true),
        "n_bonafide": int(np.sum(y_true == 0)),
        "n_spoof": int(np.sum(y_true == 1)),
    }

    return results


def evaluate_classical_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate a classical ML model.

    Args:
        model: Fitted sklearn/xgboost model with predict/get_scores methods.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dictionary with all metrics.
    """
    y_pred = model.predict(X_test)
    y_scores = model.get_scores(X_test)

    metrics = compute_classification_metrics(y_test, y_pred, y_scores)
    cm_metrics = compute_confusion_matrix_metrics(y_test, y_pred)
    roc_data = compute_roc_curve(y_test, y_scores)
    pr_data = compute_pr_curve(y_test, y_scores)

    return {
        "metrics": metrics,
        "confusion_matrix": cm_metrics,
        "roc_curve": {
            "fpr": roc_data["fpr"].tolist(),
            "tpr": roc_data["tpr"].tolist(),
        },
        "pr_curve": {
            "precision": pr_data["precision"].tolist(),
            "recall": pr_data["recall"].tolist(),
        },
        "n_samples": len(y_test),
    }


def save_results(
    results: Dict[str, Any],
    save_dir: str,
    model_name: str,
    test_set: str,
    seed: int,
) -> str:
    """Save evaluation results to JSON.

    Args:
        results: Results dictionary.
        save_dir: Output directory.
        model_name: Model name.
        test_set: Test set name.
        seed: Random seed.

    Returns:
        Path to saved results file.
    """
    save_path = Path(save_dir) / model_name / f"seed_{seed}"
    save_path.mkdir(parents=True, exist_ok=True)

    filepath = save_path / f"{test_set}_results.json"
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return str(filepath)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test_set", type=str, required=True,
                        choices=["asvspoof2019_eval", "asvspoof2021_la",
                                 "asvspoof2021_df", "wavefake"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)
    device = get_device(args.device)

    logger = setup_logger("evaluate", config.get("paths", {}).get("logs", "./logs"))
    logger.info(f"Evaluating {args.model} on {args.test_set} (seed={args.seed})")

    # Results would be saved here
    results_dir = config.get("paths", {}).get("results", "./results")

    logger.info("Evaluation script ready. Call evaluate_deep_model() or "
                "evaluate_classical_model() programmatically.")


if __name__ == "__main__":
    main()
