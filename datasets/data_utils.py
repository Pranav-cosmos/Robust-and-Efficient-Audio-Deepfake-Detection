"""Data utility functions: DataLoader creation, leakage checks, collation."""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    collate_fn=None,
) -> DataLoader:
    """Create a DataLoader with standard settings.

    Args:
        dataset: PyTorch Dataset.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for GPU transfer.
        drop_last: Whether to drop the last incomplete batch.
        collate_fn: Optional custom collate function.

    Returns:
        DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn or default_collate_fn,
    )


def default_collate_fn(
    batch: List[Tuple[torch.Tensor, int, Dict]],
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """Custom collate function that handles metadata dictionaries.

    Args:
        batch: List of (waveform, label, metadata) tuples.

    Returns:
        Tuple of (batched_waveforms, batched_labels, metadata_list).
    """
    waveforms = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    metadata = [item[2] for item in batch]
    return waveforms, labels, metadata


def verify_no_speaker_leakage(
    train_dataset,
    eval_dataset,
) -> bool:
    """Verify no speaker leakage between train and eval sets.

    Args:
        train_dataset: Training dataset with get_speakers() method.
        eval_dataset: Evaluation dataset with get_speakers() method.

    Returns:
        True if no leakage, False otherwise.
    """
    train_speakers = set(train_dataset.get_speakers())
    eval_speakers = set(eval_dataset.get_speakers())
    overlap = train_speakers & eval_speakers

    if overlap:
        logger.warning(
            f"Speaker leakage detected! {len(overlap)} speakers in both sets: {overlap}"
        )
        return False

    logger.info("No speaker leakage detected.")
    return True


def verify_no_attack_leakage(
    train_dataset,
    eval_dataset,
    allow_bonafide: bool = True,
) -> Dict[str, Any]:
    """Check for attack type leakage between train and eval sets.

    Note: For ASVspoof 2019, some known attacks are in train and unknown
    attacks are in eval. This is by design, not leakage.

    Args:
        train_dataset: Training dataset.
        eval_dataset: Evaluation dataset.
        allow_bonafide: If True, bonafide ('-') overlap is acceptable.

    Returns:
        Dictionary with leakage analysis results.
    """
    train_attacks = set(train_dataset.get_attack_types())
    eval_attacks = set(eval_dataset.get_attack_types())

    if allow_bonafide:
        train_attacks.discard("-")
        eval_attacks.discard("-")

    overlap = train_attacks & eval_attacks
    eval_only = eval_attacks - train_attacks

    result = {
        "train_attacks": sorted(list(train_attacks)),
        "eval_attacks": sorted(list(eval_attacks)),
        "overlapping_attacks": sorted(list(overlap)),
        "eval_only_attacks": sorted(list(eval_only)),
        "has_novel_attacks": len(eval_only) > 0,
    }

    logger.info(f"Train attacks: {result['train_attacks']}")
    logger.info(f"Eval attacks: {result['eval_attacks']}")
    logger.info(f"Eval-only (novel) attacks: {result['eval_only_attacks']}")

    return result


def compute_class_weights(dataset) -> Tuple[float, float]:
    """Compute class weights for handling imbalance.

    Args:
        dataset: Dataset with samples attribute containing label fields.

    Returns:
        Tuple of (bonafide_weight, spoof_weight).
    """
    labels = [s["label"] for s in dataset.samples]
    n_bonafide = labels.count(0)
    n_spoof = labels.count(1)
    total = len(labels)

    if n_bonafide == 0 or n_spoof == 0:
        return 1.0, 1.0

    w_bonafide = total / (2.0 * n_bonafide)
    w_spoof = total / (2.0 * n_spoof)

    return w_bonafide, w_spoof


def make_balanced_subset(dataset: Dataset, max_samples: int, seed: int = 42) -> Dataset:
    """Return a deterministic class-balanced subset when labels are available."""
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset

    samples = getattr(dataset, "samples", None)
    if not samples:
        return Subset(dataset, list(range(max_samples)))

    rng = np.random.default_rng(seed)
    labels = np.array([sample["label"] for sample in samples])
    classes = np.unique(labels)
    per_class = max(1, max_samples // max(len(classes), 1))
    selected = []

    for cls in classes:
        cls_indices = np.flatnonzero(labels == cls)
        take = min(per_class, len(cls_indices))
        selected.extend(rng.choice(cls_indices, size=take, replace=False).tolist())

    if len(selected) < max_samples:
        remaining = np.setdiff1d(np.arange(len(labels)), np.array(selected), assume_unique=False)
        take = min(max_samples - len(selected), len(remaining))
        if take > 0:
            selected.extend(rng.choice(remaining, size=take, replace=False).tolist())

    return Subset(dataset, sorted(selected[:max_samples]))
