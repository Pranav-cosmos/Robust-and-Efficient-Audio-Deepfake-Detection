"""XGBoost classifier wrapper for audio deepfake detection."""
import os
import numpy as np
import joblib
from pathlib import Path
from typing import Any, Dict, Optional
import xgboost as xgb


class XGBoostModel:
    """XGBoost wrapper with consistent API.

    Input: MFCC utterance-level features (240-dim).
    Handles class imbalance via scale_pos_weight.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize XGBoost model.

        Args:
            config: Configuration dictionary with model.params.
        """
        params = config.get("model", {}).get("params", {})

        self.scale_pos_weight_auto = params.get("scale_pos_weight", 1.0) == "auto"
        self.early_stopping_rounds = params.get("early_stopping_rounds", 20)

        model_params = {
            "n_estimators": params.get("n_estimators", 150),
            "max_depth": params.get("max_depth", 6),
            "learning_rate": params.get("learning_rate", 0.1),
            "subsample": params.get("subsample", 0.8),
            "colsample_bytree": params.get("colsample_bytree", 0.8),
            "min_child_weight": params.get("min_child_weight", 5),
            "gamma": params.get("gamma", 0.1),
            "reg_alpha": params.get("reg_alpha", 0.1),
            "reg_lambda": params.get("reg_lambda", 1.0),
            "eval_metric": params.get("eval_metric", "logloss"),
            "use_label_encoder": False,
            "tree_method": params.get("tree_method", "hist"),
            "device": params.get("device", "cpu"),
            "random_state": params.get("random_state", 42),
            "verbosity": 0,
        }

        if not self.scale_pos_weight_auto:
            model_params["scale_pos_weight"] = params.get("scale_pos_weight", 1.0)

        self.model = xgb.XGBClassifier(**model_params)
        self.is_fitted = False

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train the XGBoost model.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Optional validation features for early stopping.
            y_val: Optional validation labels.
        """
        # Auto-compute scale_pos_weight
        if self.scale_pos_weight_auto:
            n_neg = np.sum(y_train == 0)
            n_pos = np.sum(y_train == 1)
            if n_pos > 0:
                self.model.set_params(scale_pos_weight=n_neg / n_pos)

        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False
            self.model.set_params(early_stopping_rounds=self.early_stopping_rounds)

        self.model.fit(X_train, y_train, **fit_params)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_scores(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X)[:, 1]

    def save(self, save_path: str) -> None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, save_path)

    def load(self, load_path: str) -> None:
        self.model = joblib.load(load_path)
        self.is_fitted = True

    def get_model_size_mb(self) -> float:
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
        if not self.is_fitted:
            return {"total_params": 0}
        booster = self.model.get_booster()
        # Approximate: trees * avg nodes
        n_trees = int(booster.attr("best_iteration") or self.model.n_estimators)
        return {"total_params": n_trees * 100, "trainable_params": n_trees * 100}
