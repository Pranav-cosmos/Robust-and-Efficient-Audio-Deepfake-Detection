"""Robustness evaluation under audio perturbations.

Evaluates each model under all configured perturbation types and levels.
Computes robustness scores and performance degradation metrics.
"""
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import compute_classification_metrics, compute_robustness_score
from preprocessing.perturbations import PerturbationPipeline


def evaluate_robustness_deep(
    model: nn.Module,
    dataloader: DataLoader,
    perturbation_pipeline: PerturbationPipeline,
    device: torch.device,
    prepare_fn=None,
) -> Dict[str, Any]:
    """Evaluate deep model robustness under all perturbations.

    Args:
        model: Trained model in eval mode.
        dataloader: Test DataLoader.
        perturbation_pipeline: Configured perturbation pipeline.
        device: Device.
        prepare_fn: Optional batch preparation function.

    Returns:
        Dictionary with clean and perturbed metrics.
    """
    model.eval()
    perturbation_list = perturbation_pipeline.get_perturbation_list()

    # First pass: collect clean results
    clean_labels = []
    clean_scores = []
    clean_preds = []

    # Store all waveforms for perturbation
    all_waveforms = []
    all_labels_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting data"):
            waveforms, labels, _ = batch
            all_waveforms.append(waveforms)
            all_labels_list.append(labels)

            if prepare_fn:
                inputs, labels_d = prepare_fn(batch)
            else:
                inputs = waveforms.to(device)
                labels_d = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            clean_scores.extend(probs[:, 1].cpu().numpy())
            clean_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            clean_labels.extend(labels.numpy())

    y_true = np.array(clean_labels)
    clean_metrics = compute_classification_metrics(
        y_true, np.array(clean_preds), np.array(clean_scores)
    )

    # Evaluate under each perturbation
    perturbed_results = {}

    for pert_name, pert_params in tqdm(perturbation_list, desc="Perturbations"):
        pert_scores = []
        pert_preds = []

        with torch.no_grad():
            for waveforms, labels, _ in tqdm(
                dataloader, desc=f"  {pert_name}", leave=False
            ):
                # Apply perturbation
                perturbed_batch = []
                for i in range(waveforms.shape[0]):
                    perturbed = perturbation_pipeline.apply_single(
                        waveforms[i], pert_params
                    )
                    perturbed_batch.append(perturbed)

                perturbed_waveforms = torch.stack(perturbed_batch)

                if prepare_fn:
                    inputs, _ = prepare_fn(
                        (perturbed_waveforms, labels, [{}] * len(labels))
                    )
                else:
                    inputs = perturbed_waveforms.to(device)

                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                pert_scores.extend(probs[:, 1].cpu().numpy())
                pert_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())

        pert_metrics = compute_classification_metrics(
            y_true, np.array(pert_preds), np.array(pert_scores)
        )
        perturbed_results[pert_name] = pert_metrics

    # Compute robustness score
    perturbed_accuracies = {
        name: metrics["accuracy"] for name, metrics in perturbed_results.items()
    }
    robustness = compute_robustness_score(clean_metrics["accuracy"], perturbed_accuracies)

    return {
        "clean_metrics": clean_metrics,
        "perturbed_results": perturbed_results,
        "robustness": robustness,
    }


def evaluate_robustness_classical(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_extractor,
    dataloader: DataLoader,
    perturbation_pipeline: PerturbationPipeline,
    scaler=None,
) -> Dict[str, Any]:
    """Evaluate classical model robustness.

    For classical models, we need to re-extract features from perturbed audio.

    Args:
        model: Trained classical model.
        X_test: Clean test features.
        y_test: Test labels.
        feature_extractor: Feature extractor (MFCC).
        dataloader: DataLoader for raw audio.
        perturbation_pipeline: Perturbation pipeline.
        scaler: Feature scaler.

    Returns:
        Robustness results dictionary.
    """
    # Clean metrics
    y_pred = model.predict(X_test)
    y_scores = model.get_scores(X_test)
    clean_metrics = compute_classification_metrics(y_test, y_pred, y_scores)

    perturbation_list = perturbation_pipeline.get_perturbation_list()
    perturbed_results = {}

    for pert_name, pert_params in tqdm(perturbation_list, desc="Perturbations"):
        # Re-extract features from perturbed audio
        pert_features = []
        pert_labels = []

        for waveforms, labels, _ in dataloader:
            for i in range(waveforms.shape[0]):
                perturbed = perturbation_pipeline.apply_single(
                    waveforms[i], pert_params
                )
                feat = feature_extractor.extract_utterance_level(perturbed)
                pert_features.append(feat)
                pert_labels.append(labels[i].item())

        X_pert = np.array(pert_features)
        y_pert = np.array(pert_labels)

        if scaler is not None:
            X_pert = scaler.transform(X_pert)

        y_pred_p = model.predict(X_pert)
        y_scores_p = model.get_scores(X_pert)
        perturbed_results[pert_name] = compute_classification_metrics(
            y_pert, y_pred_p, y_scores_p
        )

    perturbed_accuracies = {
        name: m["accuracy"] for name, m in perturbed_results.items()
    }
    robustness = compute_robustness_score(clean_metrics["accuracy"], perturbed_accuracies)

    return {
        "clean_metrics": clean_metrics,
        "perturbed_results": perturbed_results,
        "robustness": robustness,
    }
