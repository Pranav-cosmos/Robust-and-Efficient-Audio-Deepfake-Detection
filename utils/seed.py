"""Seed setting for reproducibility across all random number generators."""
import os
import random
from typing import Optional

# Fix OpenMP duplicate-runtime crash on Windows (Intel MKL + PyTorch both load libiomp5)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set seed for all random number generators for reproducibility.

    Args:
        seed: Integer seed value.
        deterministic: If True, enables deterministic CUDA operations.
            May reduce performance slightly.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch >= 1.8
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Older PyTorch versions
            torch.use_deterministic_algorithms(True)


def get_device(preference: str = "cuda") -> torch.device:
    """Get the compute device based on preference and availability.

    Args:
        preference: One of 'auto', 'cuda', 'cpu'.
            - 'cuda': Forces GPU. Raises RuntimeError if CUDA is unavailable.
            - 'auto': Uses CUDA if available, falls back to CPU.
            - 'cpu' : Forces CPU.

    Returns:
        torch.device instance.

    Raises:
        RuntimeError: If preference is 'cuda' but CUDA is not available.
    """
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available but device='cuda' was requested. "
                "Please install a CUDA-enabled PyTorch build:\n"
                "  pip install torch torchvision torchaudio "
                "--index-url https://download.pytorch.org/whl/cu124"
            )
        return torch.device("cuda")
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)
