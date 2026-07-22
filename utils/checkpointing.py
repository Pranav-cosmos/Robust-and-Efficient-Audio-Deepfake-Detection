"""Model checkpointing utilities for save, load, and resume."""
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a training checkpoint.

    Args:
        model: PyTorch model.
        optimizer: Optimizer state.
        epoch: Current epoch number.
        metrics: Dictionary of current metrics.
        save_path: Path to save the checkpoint file.
        scheduler: Optional learning rate scheduler.
        scaler: Optional AMP GradScaler.
        extra: Optional extra data to save.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    if extra is not None:
        checkpoint.update(extra)

    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a training checkpoint and restore states.

    Args:
        checkpoint_path: Path to the checkpoint file.
        model: Model to load state into.
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.
        scaler: Optional AMP scaler to restore.
        device: Device to map tensors to.

    Returns:
        Checkpoint dictionary with all saved data.
    """
    map_location = device if device is not None else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint


def save_best_model(
    model: torch.nn.Module,
    metrics: Dict[str, float],
    save_dir: str,
    model_name: str,
    seed: int,
) -> str:
    """Save the best model weights.

    Args:
        model: PyTorch model.
        metrics: Metrics at the time of saving.
        save_dir: Directory to save the model.
        model_name: Name of the model.
        seed: Random seed used for training.

    Returns:
        Path to the saved model file.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{model_name}_seed{seed}_best.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        },
        save_path,
    )
    return str(save_path)


def get_checkpoint_path(
    checkpoint_dir: str,
    model_name: str,
    seed: int,
    checkpoint_type: str = "best",
) -> str:
    """Construct the checkpoint file path.

    Args:
        checkpoint_dir: Base checkpoint directory.
        model_name: Name of the model.
        seed: Random seed.
        checkpoint_type: 'best' or 'last'.

    Returns:
        Checkpoint file path.
    """
    return os.path.join(
        checkpoint_dir, model_name, f"seed_{seed}", f"{checkpoint_type}.pt"
    )
