"""Training script for classical ML models (Random Forest, XGBoost).

Usage:
    python training/train_classical.py --model random_forest --seed 42
    python training/train_classical.py --model xgboost --seed 42
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Must be set before any torch/numpy import to fix OpenMP duplicate runtime on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config, resolve_model_config
from utils.seed import set_seed
from utils.logger import setup_logger
from utils.metrics import compute_classification_metrics
from features.feature_utils import load_features, normalize_features, save_scaler
from models.classical.random_forest import RandomForestModel
from models.classical.xgboost_model import XGBoostModel


MODEL_CLASSES = {
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
}


def select_balanced_subset(X: np.ndarray, y: np.ndarray, max_samples: int, seed: int) -> tuple:
    """Select a deterministic class-balanced subset for smoke/debug runs."""
    if max_samples <= 0 or len(y) <= max_samples:
        return X, y

    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = max(1, max_samples // max(len(classes), 1))
    selected = []

    for cls in classes:
        cls_idx = np.flatnonzero(y == cls)
        take = min(per_class, len(cls_idx))
        selected.extend(rng.choice(cls_idx, size=take, replace=False).tolist())

    if len(selected) < max_samples:
        remaining = np.setdiff1d(np.arange(len(y)), np.array(selected), assume_unique=False)
        take = min(max_samples - len(selected), len(remaining))
        if take > 0:
            selected.extend(rng.choice(remaining, size=take, replace=False).tolist())

    selected = np.array(sorted(selected[:max_samples]))
    return X[selected], y[selected]


def save_scores(result_dir: str, filename: str, y_true, y_scores, y_pred) -> None:
    np.savez(
        os.path.join(result_dir, filename),
        y_true=y_true,
        y_scores=y_scores,
        y_pred=y_pred,
    )


def main():
    parser = argparse.ArgumentParser(description="Train classical ML model")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_CLASSES.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--train_features", type=str, default=None)
    parser.add_argument("--val_features", type=str, default=None)
    parser.add_argument("--eval_features", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--use_gpu", action="store_true", default=False,
                        help="Enable GPU acceleration for XGBoost")
    parser.add_argument("--debug", action="store_true", help="Use bounded subsets for quick checks")
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--artifact_tag", type=str, default=None,
                        help="Write checkpoints/results under a tag subdirectory")
    args = parser.parse_args()

    model_config_path = resolve_model_config(args.model)
    config = load_config(args.config, model_config_path)

    if args.artifact_tag:
        paths = config.setdefault("paths", {})
        for key in ("checkpoints", "results", "logs"):
            paths[key] = os.path.join(paths.get(key, f"./{key}"), args.artifact_tag)

    if args.debug:
        args.max_train_samples = args.max_train_samples or 1000
        args.max_val_samples = args.max_val_samples or 500
        args.max_eval_samples = args.max_eval_samples or 500

    set_seed(args.seed)

    logger = setup_logger(f"train_{args.model}", config.get("paths", {}).get("logs", "./logs"))
    logger.info(f"Training {args.model} with seed {args.seed}")

    cache_dir = config.get("paths", {}).get("feature_cache", "./cache/features")
    train_path = args.train_features or os.path.join(cache_dir, "mfcc", "asvspoof2019_train.npz")
    val_path = args.val_features or os.path.join(cache_dir, "mfcc", "asvspoof2019_dev.npz")
    eval_path = args.eval_features or os.path.join(cache_dir, "mfcc", "asvspoof2019_eval.npz")

    logger.info(f"Loading training features from {train_path}")
    X_train, y_train = load_features(train_path)
    X_train, y_train = select_balanced_subset(X_train, y_train, args.max_train_samples, args.seed)
    logger.info(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    X_val, y_val = None, None
    if os.path.exists(val_path):
        X_val, y_val = load_features(val_path)
        X_val, y_val = select_balanced_subset(X_val, y_val, args.max_val_samples, args.seed + 1)
        logger.info(f"Validation set: {X_val.shape[0]} samples")

    X_eval, y_eval = None, None
    if os.path.exists(eval_path):
        X_eval, y_eval = load_features(eval_path)
        X_eval, y_eval = select_balanced_subset(X_eval, y_eval, args.max_eval_samples, args.seed + 2)
        logger.info(f"Evaluation set: {X_eval.shape[0]} samples")

    X_train_norm, X_val_norm, scaler = normalize_features(X_train, X_val)
    X_eval_norm = scaler.transform(X_eval) if X_eval is not None else None

    output_dir = args.output_dir or os.path.join(
        config.get("paths", {}).get("checkpoints", "./checkpoints"),
        args.model
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_scaler(scaler, os.path.join(output_dir, "scaler.pkl"))

    if "model" in config and "params" in config["model"]:
        config["model"]["params"]["random_state"] = args.seed

    if args.use_gpu and args.model == "xgboost":
        if "model" not in config:
            config["model"] = {"params": {}}
        if "params" not in config["model"]:
            config["model"]["params"] = {}
        if torch.cuda.is_available():
            config["model"]["params"]["device"] = "cuda"
            logger.info("XGBoost: GPU acceleration enabled (device=cuda)")
        else:
            config["model"]["params"]["device"] = "cpu"
            logger.warning("XGBoost: CUDA requested but unavailable; using CPU")

    model = MODEL_CLASSES[args.model](config)

    logger.info("Starting training...")
    start_time = time.time()

    if args.model == "xgboost" and X_val_norm is not None:
        model.train(X_train_norm, y_train, X_val_norm, y_val)
    else:
        model.train(X_train_norm, y_train)

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.1f}s")

    model_path = os.path.join(output_dir, f"{args.model}_model.pkl")
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    validation_metrics = {}
    evaluation_metrics = {}
    y_pred, y_scores = None, None
    if X_val_norm is not None and y_val is not None:
        y_pred = model.predict(X_val_norm)
        y_scores = model.get_scores(X_val_norm)
        validation_metrics = compute_classification_metrics(y_val, y_pred, y_scores)

        logger.info("Validation Results:")
        for k, v in validation_metrics.items():
            logger.info(f"  {k}: {v:.4f}")

    eval_pred, eval_scores = None, None
    if X_eval_norm is not None and y_eval is not None:
        eval_pred = model.predict(X_eval_norm)
        eval_scores = model.get_scores(X_eval_norm)
        evaluation_metrics = compute_classification_metrics(y_eval, eval_pred, eval_scores)

        logger.info("Evaluation Results:")
        for k, v in evaluation_metrics.items():
            logger.info(f"  {k}: {v:.4f}")

    model_size = model.get_model_size_mb()
    params = model.get_params_count()
    logger.info(f"Model size: {model_size:.2f} MB")
    logger.info(f"Parameters: {params}")
    logger.info(f"Training time: {training_time:.1f}s")

    info = {
        "model": args.model,
        "seed": args.seed,
        "training_time_seconds": training_time,
        "model_size_mb": model_size,
        "parameters": params,
        "validation_metrics": validation_metrics,
        "evaluation_metrics": evaluation_metrics,
    }
    with open(os.path.join(output_dir, "training_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    result_dir = os.path.join(
        config.get("paths", {}).get("results", "./results"),
        args.model
    )
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    primary_metrics = evaluation_metrics or validation_metrics
    result_metrics = {**info, **primary_metrics}
    if evaluation_metrics:
        result_metrics["reported_split"] = "eval"
        result_metrics.update({f"validation_{k}": v for k, v in validation_metrics.items()})
    else:
        result_metrics["reported_split"] = "dev"
    with open(os.path.join(result_dir, "metrics.json"), "w") as f:
        json.dump(result_metrics, f, indent=2)

    if X_val_norm is not None and y_val is not None:
        save_scores(result_dir, "val_scores.npz", y_val, y_scores, y_pred)
    if X_eval_norm is not None and y_eval is not None:
        save_scores(result_dir, "eval_scores.npz", y_eval, eval_scores, eval_pred)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        cm_true = y_eval if eval_pred is not None else y_val
        cm_pred = eval_pred if eval_pred is not None else y_pred
        if cm_true is not None:
            cm = confusion_matrix(cm_true, cm_pred)
            disp = ConfusionMatrixDisplay(cm, display_labels=["Bonafide", "Spoof"])
            fig, ax = plt.subplots(figsize=(5, 4))
            disp.plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(f"{args.model} — Confusion Matrix")
            fig.savefig(os.path.join(result_dir, "confusion_matrix.png"), dpi=120, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        logger.warning(f"Could not save confusion matrix: {e}")

    logger.info(f"All outputs saved to: {result_dir}")


if __name__ == "__main__":
    main()
