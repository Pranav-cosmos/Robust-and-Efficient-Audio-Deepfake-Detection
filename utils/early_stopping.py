"""Early stopping to prevent overfitting during training."""
import numpy as np
from typing import Optional


class EarlyStopping:
    """Early stopping monitor that tracks a metric and stops when it stops improving.

    Supports both 'min' mode (e.g., loss, EER) and 'max' mode (e.g., accuracy).
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = True,
    ):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait after last improvement.
            min_delta: Minimum change to qualify as an improvement.
            mode: 'min' for metrics to minimize (loss, EER),
                  'max' for metrics to maximize (accuracy).
            verbose: If True, prints messages on improvement or stopping.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False
        self.best_epoch: int = 0

        if mode == "min":
            self.best_score = np.inf
            self._is_better = lambda current, best: current < best - min_delta
        elif mode == "max":
            self.best_score = -np.inf
            self._is_better = lambda current, best: current > best + min_delta
        else:
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

    def __call__(self, score: float, epoch: int) -> bool:
        """Check if training should stop.

        Args:
            score: Current metric value.
            epoch: Current epoch number.

        Returns:
            True if training should stop, False otherwise.
        """
        if self._is_better(score, self.best_score):
            if self.verbose:
                improvement = abs(score - self.best_score)
                print(
                    f"  Early stopping: {self.mode} score improved "
                    f"({self.best_score:.6f} -> {score:.6f}, delta={improvement:.6f})"
                )
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"  Early stopping: no improvement for {self.counter}/{self.patience} epochs"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(
                        f"  Early stopping triggered. "
                        f"Best {self.mode} score: {self.best_score:.6f} at epoch {self.best_epoch}"
                    )
                return True
            return False

    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.early_stop = False
        if self.mode == "min":
            self.best_score = np.inf
        else:
            self.best_score = -np.inf
