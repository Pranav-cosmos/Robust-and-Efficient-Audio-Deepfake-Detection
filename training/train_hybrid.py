"""Train hybrid deep-embedding + XGBoost models.

Hybrid models freeze a neural feature extractor, cache embeddings once, and
train a lightweight XGBoost classifier on the cached representation.
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.asvspoof2019 import ASVspoof2019LA
from datasets.data_utils import get_dataloader
from features.feature_utils import load_features, normalize_features, save_scaler
from features.mel_spectrogram_extractor import OnTheFlyMelSpectrogram
from models.classical.xgboost_model import XGBoostModel
from models.deep.mobilenetv2 import MobileNetV2Audio
from models.deep.simple_cnn import SimpleCNN
from utils.config import load_config, resolve_model_config
from utils.logger import setup_logger
from utils.metrics import compute_classification_metrics
from utils.seed import get_device, set_seed


HYBRID_MODELS = ["hybrid_mfcc_cnn_xgboost", "hybrid_mobilenetv2_xgboost"]


def balanced_indices_from_samples(samples: list, max_samples: int, seed: int) -> list | None:
    if max_samples <= 0 or len(samples) <= max_samples:
        return None

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

    return sorted(selected[:max_samples])


def subset_dataset(dataset, indices: list | None):
    return Subset(dataset, indices) if indices is not None else dataset


def maybe_subset_array(array: np.ndarray, indices: list | None) -> np.ndarray:
    return array[indices] if indices is not None else array


def load_model_state_if_available(model, checkpoint_path: Path, logger) -> bool:
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint not found, using initialized extractor: {checkpoint_path}")
        return False

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=False)
    logger.info(f"Loaded feature extractor checkpoint: {checkpoint_path}")
    return True


def resolve_backbone_checkpoint(config: dict, project_root: Path, artifact_tag: str | None, checkpoint_model: str) -> Path:
    checkpoint_root = Path(config.get("paths", {}).get("checkpoints", "./checkpoints"))
    primary = checkpoint_root / checkpoint_model / "best.pt"
    if primary.exists():
        return primary
    if artifact_tag:
        fallback = project_root / "checkpoints" / checkpoint_model / "best.pt"
        if fallback.exists():
            return fallback
    return primary


def build_extractor(model_name: str, config: dict, project_root: Path, artifact_tag: str | None, logger):
    feature_cfg = config.get("model", {}).get("feature_extractor", {})
    backbone = feature_cfg.get("backbone")

    if backbone == "simple_cnn":
        cnn_config = load_config(config.get("_base_config_path", "configs/base_config.yaml"), resolve_model_config("simple_cnn"))
        model = SimpleCNN(cnn_config)
        checkpoint_model = feature_cfg.get("checkpoint_model", "simple_cnn")
        checkpoint_path = resolve_backbone_checkpoint(config, project_root, artifact_tag, checkpoint_model)
        load_model_state_if_available(model, checkpoint_path, logger)
    elif backbone == "mobilenetv2":
        cnn_config = load_config(config.get("_base_config_path", "configs/base_config.yaml"), resolve_model_config("mobilenetv2"))
        model = MobileNetV2Audio(cnn_config)
    else:
        raise ValueError(f"Unsupported hybrid backbone: {backbone}")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


@torch.no_grad()
def extract_embeddings(
    model,
    dataset,
    config: dict,
    device: torch.device,
    cache_path: Path,
    logger,
    batch_size: int,
) -> tuple:
    if cache_path.exists():
        data = np.load(cache_path)
        logger.info(f"Using cached embeddings: {cache_path}")
        return data["embeddings"], data["labels"]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.to(device)

    mel_cfg = config.get("mel_spectrogram", {})
    audio_cfg = config.get("audio", {})
    mel_extractor = OnTheFlyMelSpectrogram(
        n_mels=mel_cfg.get("n_mels", 80),
        n_fft=mel_cfg.get("n_fft", 512),
        win_length=mel_cfg.get("win_length", 400),
        hop_length=mel_cfg.get("hop_length", 160),
        f_min=mel_cfg.get("f_min", 20.0),
        f_max=mel_cfg.get("f_max", 8000.0),
        sample_rate=audio_cfg.get("sample_rate", 16000),
    ).to(device)

    loader = get_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embeddings = []
    labels_all = []

    for waveforms, labels, _ in tqdm(loader, desc=f"Embeddings {cache_path.stem}", leave=False):
        waveforms = waveforms.to(device)
        mel_specs = mel_extractor(waveforms)
        batch_embeddings = model.get_embedding(mel_specs)
        embeddings.append(batch_embeddings.detach().cpu().numpy())
        labels_all.extend(labels.numpy())

    X = np.concatenate(embeddings, axis=0)
    y = np.array(labels_all)
    np.savez_compressed(cache_path, embeddings=X, labels=y)
    logger.info(f"Saved embeddings: {cache_path} ({X.shape})")
    return X, y


def save_scores(result_dir: Path, filename: str, y_true, y_scores, y_pred) -> None:
    np.savez(result_dir / filename, y_true=y_true, y_scores=y_scores, y_pred=y_pred)


def save_confusion_matrix(result_dir: Path, model_name: str, y_true, y_pred, logger) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Bonafide", "Spoof"])
        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{model_name} - Confusion Matrix")
        fig.savefig(result_dir / "confusion_matrix.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logger.warning(f"Could not save confusion matrix: {exc}")


def save_hybrid_curve(result_dir: Path, model_name: str, validation_metrics: dict, evaluation_metrics: dict, logger) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = ["Dev", "Eval"] if evaluation_metrics else ["Dev"]
        eer = [validation_metrics.get("eer", 0.0) * 100]
        acc = [validation_metrics.get("accuracy", 0.0) * 100]
        if evaluation_metrics:
            eer.append(evaluation_metrics.get("eer", 0.0) * 100)
            acc.append(evaluation_metrics.get("accuracy", 0.0) * 100)

        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, eer, marker="o", label="EER (%)")
        ax.plot(x, acc, marker="s", label="Accuracy (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"{model_name} - Hybrid Validation Summary")
        ax.set_ylabel("Percent")
        ax.legend()
        fig.tight_layout()
        fig.savefig(result_dir / "training_curves.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logger.warning(f"Could not save hybrid curve: {exc}")


def model_size_mb(model) -> float:
    import tempfile
    fd, tmp_path = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)
    try:
        joblib.dump(model, tmp_path)
        return os.path.getsize(tmp_path) / (1024 * 1024)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(description="Train hybrid embedding + XGBoost model")
    parser.add_argument("--model", required=True, choices=HYBRID_MODELS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "auto", "cpu"])
    parser.add_argument("--use_gpu", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--artifact_tag", type=str, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    model_config_path = resolve_model_config(args.model)
    config = load_config(args.config, model_config_path)
    config["_base_config_path"] = args.config

    if args.artifact_tag:
        paths = config.setdefault("paths", {})
        for key in ("checkpoints", "results", "logs", "paper_figures"):
            paths[key] = os.path.join(paths.get(key, f"./{key}"), args.artifact_tag)
        paths["embedding_cache"] = os.path.join(paths.get("embedding_cache", "./cache/embeddings"), args.artifact_tag)

    if args.debug:
        args.max_train_samples = args.max_train_samples or 128
        args.max_val_samples = args.max_val_samples or 64
        args.max_eval_samples = args.max_eval_samples or 64

    set_seed(args.seed)
    device = get_device(args.device)
    logger = setup_logger(f"train_{args.model}", config.get("paths", {}).get("logs", "./logs"))
    logger.info(f"Training {args.model} on {device} with seed {args.seed}")

    audio_cfg = config.get("audio", {})
    data_root = config.get("paths", {}).get("asvspoof2019_root", "./data/ASVspoof2019_LA")

    datasets = {
        "train": ASVspoof2019LA(data_root, split="train", sample_rate=audio_cfg.get("sample_rate", 16000), max_samples=audio_cfg.get("max_samples", 64000)),
        "dev": ASVspoof2019LA(data_root, split="dev", sample_rate=audio_cfg.get("sample_rate", 16000), max_samples=audio_cfg.get("max_samples", 64000)),
        "eval": ASVspoof2019LA(data_root, split="eval", sample_rate=audio_cfg.get("sample_rate", 16000), max_samples=audio_cfg.get("max_samples", 64000)),
    }
    subset_indices = {
        "train": balanced_indices_from_samples(datasets["train"].samples, args.max_train_samples, args.seed),
        "dev": balanced_indices_from_samples(datasets["dev"].samples, args.max_val_samples, args.seed + 1),
        "eval": balanced_indices_from_samples(datasets["eval"].samples, args.max_eval_samples, args.seed + 2),
    }

    extractor = build_extractor(args.model, config, project_root, args.artifact_tag, logger)
    embedding_root = Path(config.get("paths", {}).get("embedding_cache", "./cache/embeddings"))
    extractor_name = "simplecnn" if args.model == "hybrid_mfcc_cnn_xgboost" else "mobilenetv2"
    batch_size = 16 if args.debug else min(config.get("training", {}).get("batch_size", 96), 64)

    embeddings = {}
    labels = {}
    for split in ("train", "dev", "eval"):
        cache_path = embedding_root / extractor_name / f"asvspoof2019_{split}.npz"
        if args.debug or any(subset_indices.values()):
            cache_path = embedding_root / extractor_name / "debug" / f"asvspoof2019_{split}_{len(subset_dataset(datasets[split], subset_indices[split]))}_seed{args.seed}.npz"
        X_embed, y = extract_embeddings(
            extractor,
            subset_dataset(datasets[split], subset_indices[split]),
            config,
            device,
            cache_path,
            logger,
            batch_size,
        )
        embeddings[split] = X_embed
        labels[split] = y

    if config.get("model", {}).get("feature_extractor", {}).get("include_mfcc", False):
        cache_dir = config.get("paths", {}).get("feature_cache", "./cache/features")
        for split, filename in [("train", "asvspoof2019_train.npz"), ("dev", "asvspoof2019_dev.npz"), ("eval", "asvspoof2019_eval.npz")]:
            mfcc, mfcc_labels = load_features(os.path.join(cache_dir, "mfcc", filename))
            mfcc = maybe_subset_array(mfcc, subset_indices[split])
            mfcc_labels = maybe_subset_array(mfcc_labels, subset_indices[split])
            if not np.array_equal(mfcc_labels, labels[split]):
                raise ValueError(f"MFCC labels do not align with embeddings for {split}")
            embeddings[split] = np.concatenate([mfcc, embeddings[split]], axis=1)

    X_train, X_val, scaler = normalize_features(embeddings["train"], embeddings["dev"])
    X_eval = scaler.transform(embeddings["eval"])
    y_train, y_val, y_eval = labels["train"], labels["dev"], labels["eval"]

    if args.use_gpu and torch.cuda.is_available():
        config["model"]["params"]["device"] = "cuda"
    else:
        config["model"]["params"]["device"] = "cpu"

    model = XGBoostModel(config)
    start = time.time()
    model.train(X_train, y_train, X_val, y_val)
    training_time = time.time() - start

    checkpoint_dir = Path(config.get("paths", {}).get("checkpoints", "./checkpoints")) / args.model
    result_dir = Path(config.get("paths", {}).get("results", "./results")) / args.model
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    model_path = checkpoint_dir / f"{args.model}_model.pkl"
    scaler_path = checkpoint_dir / "scaler.pkl"
    model.save(str(model_path))
    save_scaler(scaler, str(scaler_path))

    val_pred = model.predict(X_val)
    val_scores = model.get_scores(X_val)
    eval_pred = model.predict(X_eval)
    eval_scores = model.get_scores(X_eval)
    validation_metrics = compute_classification_metrics(y_val, val_pred, val_scores)
    evaluation_metrics = compute_classification_metrics(y_eval, eval_pred, eval_scores)

    params = model.get_params_count()
    size_mb = model.get_model_size_mb()
    training_info = {
        "model": args.model,
        "seed": args.seed,
        "training_time_seconds": training_time,
        "model_size_mb": size_mb,
        "parameters": params,
        "embedding_dim": int(embeddings["train"].shape[1]),
        "validation_metrics": validation_metrics,
        "evaluation_metrics": evaluation_metrics,
        "reported_split": "eval",
    }
    with open(checkpoint_dir / "training_info.json", "w", encoding="utf-8") as f:
        json.dump(training_info, f, indent=2)

    metrics = {
        **training_info,
        **evaluation_metrics,
        **{f"validation_{key}": value for key, value in validation_metrics.items()},
    }
    with open(result_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_scores(result_dir, "val_scores.npz", y_val, val_scores, val_pred)
    save_scores(result_dir, "eval_scores.npz", y_eval, eval_scores, eval_pred)
    save_confusion_matrix(result_dir, args.model, y_eval, eval_pred, logger)
    save_hybrid_curve(result_dir, args.model, validation_metrics, evaluation_metrics, logger)

    logger.info(f"Hybrid model saved: {model_path}")
    logger.info(f"Metrics saved: {result_dir / 'metrics.json'}")
    logger.info(f"Training completed in {training_time:.1f}s")


if __name__ == "__main__":
    main()
