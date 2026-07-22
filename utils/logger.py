"""Logging utilities with file, console, and TensorBoard support."""
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# SummaryWriter is imported lazily inside get_tensorboard_writer() to avoid
# crashing when tensorboard is not installed.


def setup_logger(
    name: str,
    log_dir: str = "./logs",
    level: int = logging.INFO,
    console: bool = True,
    file_log: bool = True,
) -> logging.Logger:
    """Create and configure a logger with file and console handlers.

    Args:
        name: Logger name (typically model or experiment name).
        log_dir: Directory for log files.
        level: Logging level.
        console: Whether to add console handler.
        file_log: Whether to add file handler.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if file_log:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            log_path / f"{name}_{timestamp}.log", encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_tensorboard_writer(
    log_dir: str,
    experiment_name: str,
    model_name: str,
    seed: int,
):
    """Create a TensorBoard SummaryWriter for experiment tracking.

    Args:
        log_dir: Base log directory.
        experiment_name: Name of the experiment.
        model_name: Name of the model.
        seed: Random seed used.

    Returns:
        TensorBoard SummaryWriter, or None if tensorboard is not installed.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        import logging
        logging.getLogger(__name__).warning(
            "tensorboard not installed - TensorBoard logging disabled. "
            "Run: pip install tensorboard"
        )
        return None
    tb_dir = os.path.join(log_dir, "tensorboard", experiment_name, model_name, f"seed_{seed}")
    os.makedirs(tb_dir, exist_ok=True)
    return SummaryWriter(log_dir=tb_dir)
