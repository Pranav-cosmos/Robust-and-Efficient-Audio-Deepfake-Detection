"""Random Forest classifier wrapper for audio deepfake detection."""
import os
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from typing import Any, Dict, Optional


class RandomForestModel:
    """Random Forest wrapper with consistent API.

    Input: MFCC utterance-level features (240-dim).
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Random Forest model.

        Args:
            config: Configuration dictionary with model.params.
        """
        params = config.get("model", {}).get("params", {})

        self.model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 500),
            max_depth=params.get("max_depth", 30),
            min_samples_split=params.get("min_samples_split", 5),
            min_samples_leaf=params.get("min_samples_leaf", 2),
            max_features=params.get("max_features", "sqrt"),
            class_weight=params.get("class_weight", "balanced"),
            n_jobs=params.get("n_jobs", -1),
            random_state=params.get("random_state", 42),
            verbose=0,
        )
        self.is_fitted = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the Random Forest model.

        Args:
            X_train: Training features (n_samples, n_features).
            y_train: Training labels (n_samples,).
        """
        self.model.fit(X_train, y_train)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels.

        Args:
            X: Input features (n_samples, n_features).

        Returns:
            Predicted labels (n_samples,).
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities.

        Args:
            X: Input features (n_samples, n_features).

        Returns:
            Predicted probabilities (n_samples, 2).
        """
        return self.model.predict_proba(X)

    def get_scores(self, X: np.ndarray) -> np.ndarray:
        """Get spoof probability scores for evaluation.

        Args:
            X: Input features.

        Returns:
            Spoof probability scores (n_samples,).
        """
        return self.predict_proba(X)[:, 1]

    def save(self, save_path: str) -> None:
        """Save model to disk."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, save_path)

    def load(self, load_path: str) -> None:
        """Load model from disk."""
        self.model = joblib.load(load_path)
        self.is_fitted = True

    def get_model_size_mb(self) -> float:
        """Estimate model size."""
        import tempfile
        fd, tmp_path = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)
        try:
            joblib.dump(self.model, tmp_path)
            return os.path.getsize(tmp_path) / (1024 * 1024)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def get_params_count(self) -> Dict[str, int]:
        """Estimate number of parameters (tree nodes)."""
        if not self.is_fitted:
            return {"total_params": 0}
        total_nodes = sum(
            tree.tree_.node_count for tree in self.model.estimators_
        )
        return {"total_params": total_nodes, "trainable_params": total_nodes}
