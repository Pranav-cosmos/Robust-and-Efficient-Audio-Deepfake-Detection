"""Base Trainer class for PyTorch deep learning models.

Provides common training loop with:
    - Mixed precision (AMP)
    - Gradient clipping
    - Learning rate scheduling
    - Early stopping (patience=3)
    - Fast 25% validation during training, 100% full evaluation post-training
    - Checkpointing (best + last)
    - Per-model output artifacts (metrics.json, val_scores.npz, curves, confusion matrix)
"""
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from utils.checkpointing import save_checkpoint, load_checkpoint
from utils.early_stopping import EarlyStopping
from utils.logger import setup_logger, get_tensorboard_writer
from utils.metrics import compute_classification_metrics
from utils.timer import Timer


class BaseTrainer:
    """Base trainer for PyTorch deep learning models."""

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        model_name: str,
        seed: int = 42,
        device: Optional[torch.device] = None,
    ):
        """Initialize trainer.

        Args:
            model: PyTorch model.
            config: Training configuration.
            model_name: Name for logging and checkpointing.
            seed: Random seed (default: 42).
            device: Computation device.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.config = config
        self.model_name = model_name
        self.seed = seed

        # Training config
        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 15)
        self.lr = train_cfg.get("learning_rate", 1e-3)
        self.weight_decay = train_cfg.get("weight_decay", 1e-4)
        self.gradient_clip = train_cfg.get("gradient_clip_norm", 1.0)
        self.use_amp = train_cfg.get("amp", True) and torch.cuda.is_available()
        # Fast validation: validate on ~25% subset per epoch (0 = full set)
        self.val_subset_size = train_cfg.get("val_subset_size", 6200)

        # Paths
        paths = config.get("paths", {})
        self.checkpoint_dir = os.path.join(
            paths.get("checkpoints", "./checkpoints"), model_name
        )
        self.result_dir = os.path.join(
            paths.get("results", "./results"), model_name
        )
        self.log_dir = paths.get("logs", "./logs")

        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.result_dir).mkdir(parents=True, exist_ok=True)

        # Logger
        self.logger = setup_logger(f"{model_name}", self.log_dir)

        # TensorBoard (optional)
        self.tb_writer = get_tensorboard_writer(
            self.log_dir, "training", model_name, seed
        )

        # Optimizer
        optimizer_name = train_cfg.get("optimizer", "adam").lower()
        if optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        # Scheduler
        scheduler_name = train_cfg.get("scheduler", "cosine")
        if scheduler_name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs
            )
        elif scheduler_name == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=5, gamma=0.1
            )
        else:
            self.scheduler = None

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # AMP
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Early stopping
        patience = train_cfg.get("patience", 3)
        self.early_stopping = EarlyStopping(patience=patience, mode="min", verbose=True)

        # Tracking
        self.best_eer = float("inf")
        self.training_time = 0.0
        self.start_epoch = 0
        self.history: Dict[str, List] = {"train_loss": [], "val_eer": [], "val_acc": [], "val_f1": []}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        resume_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run training loop."""
        if resume_path and os.path.exists(resume_path):
            checkpoint = load_checkpoint(
                resume_path, self.model, self.optimizer,
                self.scheduler, self.scaler, self.device
            )
            self.start_epoch = checkpoint.get("epoch", 0) + 1
            self.logger.info(f"Resumed from epoch {self.start_epoch}")

        self.logger.info(f"Starting training: {self.model_name} (seed={self.seed})")
        self.logger.info(
            f"Epochs: {self.epochs}, LR: {self.lr}, "
            f"Batch: {self.config.get('training',{}).get('batch_size',64)}, "
            f"Val subset per epoch: {self.val_subset_size or 'full'}, Device: {self.device}"
        )

        total_timer = Timer()

        with total_timer:
            last_epoch = self.start_epoch
            for epoch in range(self.start_epoch, self.epochs):
                last_epoch = epoch
                # Train one epoch
                train_loss = self._train_epoch(train_loader, epoch)

                # Fast evaluation on validation subset (~25%)
                val_metrics = self._validate(val_loader, epoch, use_subset=True)

                # Record history
                self.history["train_loss"].append(train_loss)
                self.history["val_eer"].append(val_metrics["eer"])
                self.history["val_acc"].append(val_metrics["accuracy"])
                self.history["val_f1"].append(val_metrics.get("f1", 0.0))

                # Log progress
                self.logger.info(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val EER (subset): {val_metrics['eer']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f}"
                )

                if self.tb_writer:
                    self.tb_writer.add_scalar("train/loss", train_loss, epoch)
                    self.tb_writer.add_scalar("val/eer", val_metrics["eer"], epoch)
                    self.tb_writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
                    if self.scheduler:
                        self.tb_writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], epoch)

                # Save best model
                if val_metrics["eer"] < self.best_eer:
                    self.best_eer = val_metrics["eer"]
                    save_checkpoint(
                        self.model, self.optimizer, epoch, val_metrics,
                        os.path.join(self.checkpoint_dir, "best.pt"),
                        self.scheduler, self.scaler,
                    )

                # Save last checkpoint
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_metrics,
                    os.path.join(self.checkpoint_dir, "last.pt"),
                    self.scheduler, self.scaler,
                )

                # Step scheduler
                if self.scheduler:
                    self.scheduler.step()

                # Early stopping check
                if self.early_stopping(val_metrics["eer"], epoch):
                    self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

        self.training_time = total_timer.elapsed
        if self.tb_writer:
            self.tb_writer.close()

        self.logger.info(f"Training complete in {self.training_time:.1f}s. Performing 100% full evaluation...")

        validation_metrics = self._validate(val_loader, last_epoch, use_subset=False)
        self.logger.info(
            f"Full Validation EER: {validation_metrics['eer']:.4f} | "
            f"Accuracy: {validation_metrics['accuracy']:.4f}"
        )

        if eval_loader is not None:
            final_metrics = self._validate(eval_loader, last_epoch, use_subset=False)
            reported_split = "eval"
            self.logger.info(
                f"Full Evaluation EER: {final_metrics['eer']:.4f} | "
                f"Accuracy: {final_metrics['accuracy']:.4f}"
            )
        else:
            final_metrics = validation_metrics
            reported_split = "dev"

        # Save all per-model artifacts
        self._save_model_outputs(final_metrics, last_epoch, reported_split, validation_metrics)

        return {
            "model": self.model_name,
            "best_eer": self.best_eer,
            "final_eer": final_metrics["eer"],
            "accuracy": final_metrics["accuracy"],
            "f1": final_metrics.get("f1", 0.0),
            "training_time_seconds": self.training_time,
            "total_epochs": last_epoch + 1,
            "reported_split": reported_split,
            "best_checkpoint": os.path.join(self.checkpoint_dir, "best.pt"),
            "result_dir": self.result_dir,
        }

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for batch in pbar:
            inputs, labels = self.prepare_batch(batch)
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader, epoch: int, use_subset: bool = True) -> Dict[str, float]:
        """Validate the model on a subset (during training) or full set (post-training)."""
        self.model.eval()
        dataset = val_loader.dataset

        if use_subset and self.val_subset_size and self.val_subset_size < len(dataset):
            # Deterministic subset per epoch based on seed + epoch
            g = torch.Generator().manual_seed(self.seed + epoch)
            indices = torch.randperm(len(dataset), generator=g)[:self.val_subset_size].tolist()
            subset = Subset(dataset, indices)
            loader = DataLoader(
                subset,
                batch_size=val_loader.batch_size,
                shuffle=False,
                num_workers=0,
            )
            desc = f"Epoch {epoch+1} [Val-25%]"
        else:
            loader = val_loader
            desc = f"Full Val [100%]"

        all_labels = []
        all_scores = []
        all_preds = []

        for batch in tqdm(loader, desc=desc, leave=False):
            inputs, labels = self.prepare_batch(batch)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(inputs)
            else:
                outputs = self.model(inputs)

            probs = torch.softmax(outputs, dim=1)
            scores = probs[:, 1]  # Spoof probability
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        y_true = np.array(all_labels)
        y_scores = np.array(all_scores)
        y_pred = np.array(all_preds)

        metrics = compute_classification_metrics(y_true, y_pred, y_scores)
        metrics["_y_true"] = y_true
        metrics["_y_scores"] = y_scores
        metrics["_y_pred"] = y_pred
        return metrics

    def _save_model_outputs(
        self,
        final_metrics: Dict,
        last_epoch: int,
        reported_split: str = "dev",
        validation_metrics: Optional[Dict] = None,
    ) -> None:
        """Save per-model result artifacts to results/{model_name}/."""
        clean_metrics = {
            k: float(v) if hasattr(v, "item") else v
            for k, v in final_metrics.items()
            if not k.startswith("_")
        }
        clean_metrics["model"] = self.model_name
        clean_metrics["seed"] = self.seed
        clean_metrics["best_eer"] = self.best_eer
        clean_metrics["training_time_seconds"] = self.training_time
        clean_metrics["total_epochs"] = last_epoch + 1
        clean_metrics["parameters"] = sum(p.numel() for p in self.model.parameters())
        clean_metrics["reported_split"] = reported_split

        if validation_metrics is not None and reported_split != "dev":
            for key, value in validation_metrics.items():
                if not key.startswith("_"):
                    clean_metrics[f"validation_{key}"] = float(value) if hasattr(value, "item") else value

        metrics_path = os.path.join(self.result_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(clean_metrics, f, indent=2)
        self.logger.info(f"Metrics saved: {metrics_path}")

        # Scores for ROC curve generation. Keep val_scores for compatibility.
        score_name = "eval_scores.npz" if reported_split == "eval" else "val_scores.npz"
        scores_path = os.path.join(self.result_dir, score_name)
        np.savez(
            scores_path,
            y_true=final_metrics["_y_true"],
            y_scores=final_metrics["_y_scores"],
            y_pred=final_metrics["_y_pred"],
        )
        self.logger.info(f"Val scores saved: {scores_path}")

        if validation_metrics is not None and reported_split != "dev":
            val_scores_path = os.path.join(self.result_dir, "val_scores.npz")
            np.savez(
                val_scores_path,
                y_true=validation_metrics["_y_true"],
                y_scores=validation_metrics["_y_scores"],
                y_pred=validation_metrics["_y_pred"],
            )
            self.logger.info(f"Validation scores saved: {val_scores_path}")

        # training_curves.png
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            epochs_x = list(range(1, len(self.history["train_loss"]) + 1))
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            axes[0].plot(epochs_x, self.history["train_loss"], "b-o", markersize=4, label="Train Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_title(f"{self.model_name} — Training Loss")
            axes[0].legend()

            axes[1].plot(epochs_x, [v * 100 for v in self.history["val_eer"]], "r-o", markersize=4, label="Val EER %")
            axes[1].plot(epochs_x, [v * 100 for v in self.history["val_acc"]], "g-s", markersize=4, label="Val Acc %")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("%")
            axes[1].set_title(f"{self.model_name} — Validation Metrics")
            axes[1].legend()

            fig.tight_layout()
            curve_path = os.path.join(self.result_dir, "training_curves.png")
            fig.savefig(curve_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            self.logger.info(f"Training curves saved: {curve_path}")
        except Exception as e:
            self.logger.warning(f"Could not save training curves: {e}")

        # confusion_matrix.png
        try:
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            cm = confusion_matrix(final_metrics["_y_true"], final_metrics["_y_pred"])
            disp = ConfusionMatrixDisplay(cm, display_labels=["Bonafide", "Spoof"])
            fig, ax = plt.subplots(figsize=(5, 4))
            disp.plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(f"{self.model_name} — Confusion Matrix")
            cm_path = os.path.join(self.result_dir, "confusion_matrix.png")
            fig.savefig(cm_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            self.logger.info(f"Confusion matrix saved: {cm_path}")
        except Exception as e:
            self.logger.warning(f"Could not save confusion matrix: {e}")

    def prepare_batch(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare batch for model input."""
        waveforms, labels, _ = batch
        return waveforms.to(self.device), labels.to(self.device)
