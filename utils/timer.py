"""Timing and memory profiling utilities."""
import time
from contextlib import contextmanager
from typing import Dict, Optional

import torch


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self):
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time


class MemoryTracker:
    """Track GPU memory consumption during a block of code."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device
        self.peak_memory_bytes: int = 0
        self.peak_memory_mb: float = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.peak_memory_bytes = torch.cuda.max_memory_allocated()
            self.peak_memory_mb = self.peak_memory_bytes / (1024 * 1024)


def measure_inference_time(
    model,
    input_data,
    device: torch.device,
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict[str, float]:
    """Measure inference time for a PyTorch model.

    Args:
        model: PyTorch model (in eval mode).
        input_data: Sample input tensor.
        device: Computation device.
        num_runs: Number of timed runs.
        warmup_runs: Number of warmup runs (not timed).

    Returns:
        Dictionary with mean, std, min, max inference times in ms.
    """
    model.eval()
    input_data = input_data.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_data)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(input_data)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    import numpy as np
    times_arr = np.array(times)
    return {
        "mean_ms": float(times_arr.mean()),
        "std_ms": float(times_arr.std()),
        "min_ms": float(times_arr.min()),
        "max_ms": float(times_arr.max()),
        "median_ms": float(np.median(times_arr)),
    }


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary with total, trainable, and frozen parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": frozen,
    }


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Estimate model size on disk in MB.

    Args:
        model: PyTorch model.

    Returns:
        Estimated model size in megabytes.
    """
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    total_bytes = param_size + buffer_size
    return total_bytes / (1024 * 1024)
