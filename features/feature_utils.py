"""Feature utility functions: normalization, combination, caching."""
import os
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
import joblib


def normalize_features(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    scaler: Optional[StandardScaler] = None,
) -> Tuple:
    """Normalize features using StandardScaler.

    Fits on training data, transforms both train and test.

    Args:
        X_train: Training features (n_samples, n_features).
        X_test: Optional test features.
        scaler: Optional pre-fitted scaler.

    Returns:
        Tuple of (X_train_norm, X_test_norm or None, fitted_scaler).
    """
    if scaler is None:
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train)
    else:
        X_train_norm = scaler.transform(X_train)

    X_test_norm = None
    if X_test is not None:
        X_test_norm = scaler.transform(X_test)

    return X_train_norm, X_test_norm, scaler


def combine_features(*feature_arrays: np.ndarray) -> np.ndarray:
    """Concatenate multiple feature arrays along the feature dimension.

    Args:
        *feature_arrays: Variable number of feature arrays,
            each of shape (n_samples, feature_dim_i).

    Returns:
        Combined array of shape (n_samples, sum(feature_dim_i)).
    """
    # Verify all have the same number of samples
    n_samples = feature_arrays[0].shape[0]
    for i, arr in enumerate(feature_arrays):
        if arr.shape[0] != n_samples:
            raise ValueError(
                f"Feature array {i} has {arr.shape[0]} samples, "
                f"expected {n_samples}"
            )

    return np.concatenate(feature_arrays, axis=1)


def save_features(
    features: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    metadata: Optional[Dict] = None,
) -> None:
    """Save extracted features to disk.

    Args:
        features: Feature array.
        labels: Label array.
        save_path: Path to save (will create .npz file).
        metadata: Optional metadata dictionary to save alongside.
    """
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    save_dict = {"features": features, "labels": labels}
    if metadata is not None:
        save_dict["metadata"] = np.array([metadata], dtype=object)

    np.savez_compressed(save_path, **save_dict)


def load_features(load_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load saved features from disk.

    Args:
        load_path: Path to .npz file.

    Returns:
        Tuple of (features, labels).
    """
    data = np.load(load_path, allow_pickle=True)
    return data["features"], data["labels"]


def save_scaler(scaler: StandardScaler, save_path: str) -> None:
    """Save a fitted scaler to disk."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, save_path)


def load_scaler(load_path: str) -> StandardScaler:
    """Load a fitted scaler from disk."""
    return joblib.load(load_path)


def get_feature_cache_path(
    cache_dir: str,
    feature_type: str,
    dataset_name: str,
    split: str,
) -> str:
    """Construct feature cache file path.

    Args:
        cache_dir: Base cache directory.
        feature_type: Feature type name (e.g., 'mfcc').
        dataset_name: Dataset name (e.g., 'asvspoof2019').
        split: Data split (e.g., 'train', 'dev', 'eval').

    Returns:
        Cache file path.
    """
    return os.path.join(cache_dir, feature_type, f"{dataset_name}_{split}.npz")
