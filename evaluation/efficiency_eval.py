"""Computational efficiency evaluation.

Measures inference time, training time, GPU memory, model size, and parameter count.
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.timer import measure_inference_time, count_parameters, get_model_size_mb, MemoryTracker


def evaluate_efficiency_deep(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: torch.device,
    training_time: float = 0.0,
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict[str, Any]:
    """Evaluate computational efficiency of a deep model.

    Args:
        model: PyTorch model.
        sample_input: Sample input tensor for inference timing.
        device: Device.
        training_time: Total training time in seconds.
        num_runs: Number of inference timing runs.
        warmup_runs: Warmup runs before timing.

    Returns:
        Efficiency metrics dictionary.
    """
    model.to(device)
    model.eval()

    # Parameter count
    params = count_parameters(model)

    # Model size
    model_size = get_model_size_mb(model)

    # Inference time
    timing = measure_inference_time(
        model, sample_input, device, num_runs, warmup_runs
    )

    # GPU memory
    gpu_memory = 0.0
    if torch.cuda.is_available():
        with MemoryTracker() as mem:
            with torch.no_grad():
                sample_input = sample_input.to(device)
                _ = model(sample_input)
        gpu_memory = mem.peak_memory_mb

    return {
        "total_params": params["total_params"],
        "trainable_params": params["trainable_params"],
        "frozen_params": params["frozen_params"],
        "model_size_mb": model_size,
        "inference_time_ms": timing["mean_ms"],
        "inference_time_std_ms": timing["std_ms"],
        "inference_time_median_ms": timing["median_ms"],
        "training_time_seconds": training_time,
        "gpu_memory_mb": gpu_memory,
    }


def evaluate_efficiency_classical(
    model,
    X_sample: np.ndarray,
    training_time: float = 0.0,
    num_runs: int = 100,
) -> Dict[str, Any]:
    """Evaluate computational efficiency of a classical model.

    Args:
        model: Trained classical model.
        X_sample: Sample input for inference timing.
        training_time: Total training time in seconds.
        num_runs: Number of timing runs.

    Returns:
        Efficiency metrics dictionary.
    """
    # Model size
    model_size = model.get_model_size_mb()

    # Parameter count
    params = model.get_params_count()

    # Inference time (single sample)
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model.predict(X_sample[:1])
        end = time.perf_counter()
        times.append((end - start) * 1000)

    times_arr = np.array(times[10:])  # Skip warmup

    return {
        "total_params": params.get("total_params", 0),
        "trainable_params": params.get("trainable_params", 0),
        "frozen_params": 0,
        "model_size_mb": model_size,
        "inference_time_ms": float(times_arr.mean()),
        "inference_time_std_ms": float(times_arr.std()),
        "inference_time_median_ms": float(np.median(times_arr)),
        "training_time_seconds": training_time,
        "gpu_memory_mb": 0.0,
    }


def save_efficiency_results(
    results: Dict[str, Dict[str, Any]],
    save_dir: str,
) -> str:
    """Save efficiency results for all models."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    filepath = save_path / "efficiency_results.json"
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return str(filepath)
