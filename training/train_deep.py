"""Training script for deep learning models (SimpleCNN, MobileNetV2, ResNet18).

Usage:
    python training/train_deep.py --model simple_cnn --seed 42
    python training/train_deep.py --model mobilenetv2 --seed 42
    python training/train_deep.py --model resnet18 --seed 42
"""
import argparse
import copy
import os
import sys

# Must be set before any torch/numpy import to fix OpenMP duplicate runtime on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config, resolve_model_config
from utils.seed import set_seed, get_device
from utils.logger import setup_logger
from datasets.asvspoof2019 import ASVspoof2019LA
from datasets.data_utils import get_dataloader, make_balanced_subset
from features.mel_spectrogram_extractor import OnTheFlyMelSpectrogram
from models.deep.simple_cnn import SimpleCNN
from models.deep.mobilenetv2 import MobileNetV2Audio
from models.deep.resnet18 import ResNet18Audio
from training.trainer import BaseTrainer


MODEL_CLASSES = {
    "simple_cnn": SimpleCNN,
    "mobilenetv2": MobileNetV2Audio,
    "resnet18": ResNet18Audio,
}


class SpectrogramTrainer(BaseTrainer):
    """Trainer for spectrogram-based CNN models."""

    def __init__(self, model, config, model_name, seed, device):
        super().__init__(model, config, model_name, seed, device)
        mel_cfg = config.get("mel_spectrogram", {})
        self.mel_extractor = OnTheFlyMelSpectrogram(
            n_mels=mel_cfg.get("n_mels", 80),
            n_fft=mel_cfg.get("n_fft", 512),
            win_length=mel_cfg.get("win_length", 400),
            hop_length=mel_cfg.get("hop_length", 160),
            f_min=mel_cfg.get("f_min", 20.0),
            f_max=mel_cfg.get("f_max", 8000.0),
            sample_rate=config.get("audio", {}).get("sample_rate", 16000),
        ).to(device)

    def prepare_batch(self, batch):
        waveforms, labels, _ = batch
        waveforms = waveforms.to(self.device)
        labels = labels.to(self.device)
        mel_specs = self.mel_extractor(waveforms)
        return mel_specs, labels


def main():
    parser = argparse.ArgumentParser(description="Train deep learning model")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_CLASSES.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "auto", "cpu"])
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: 2 epochs, small batch")
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
        for key in ("checkpoints", "results", "logs", "paper_figures"):
            paths[key] = os.path.join(paths.get(key, f"./{key}"), args.artifact_tag)

    if args.debug:
        config["training"]["epochs"] = 2
        config["training"]["batch_size"] = 8
        config["training"]["val_subset_size"] = 64
        args.max_train_samples = args.max_train_samples or 128
        args.max_val_samples = args.max_val_samples or 64
        args.max_eval_samples = args.max_eval_samples or 64

    set_seed(args.seed)
    device = get_device(args.device)

    logger = setup_logger(f"train_{args.model}", config.get("paths", {}).get("logs", "./logs"))
    logger.info(f"Training {args.model} on {device} with seed {args.seed}")

    data_root = config.get("paths", {}).get("asvspoof2019_root", "./data/ASVspoof2019_LA")
    audio_cfg = config.get("audio", {})

    train_dataset = ASVspoof2019LA(
        root_dir=data_root, split="train",
        sample_rate=audio_cfg.get("sample_rate", 16000),
        max_samples=audio_cfg.get("max_samples", 64000),
    )
    val_dataset = ASVspoof2019LA(
        root_dir=data_root, split="dev",
        sample_rate=audio_cfg.get("sample_rate", 16000),
        max_samples=audio_cfg.get("max_samples", 64000),
    )
    eval_dataset = ASVspoof2019LA(
        root_dir=data_root, split="eval",
        sample_rate=audio_cfg.get("sample_rate", 16000),
        max_samples=audio_cfg.get("max_samples", 64000),
    )

    train_dataset = make_balanced_subset(train_dataset, args.max_train_samples, args.seed)
    val_dataset = make_balanced_subset(val_dataset, args.max_val_samples, args.seed + 1)
    eval_dataset = make_balanced_subset(eval_dataset, args.max_eval_samples, args.seed + 2)

    train_cfg = config.get("training", {})
    logger.info(
        f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples, "
        f"Eval: {len(eval_dataset)} samples"
    )

    configured_batch = train_cfg.get("batch_size", 64)
    fallback_batches = [configured_batch, 64, 32, 16, 8]
    fallback_batches = list(dict.fromkeys(b for b in fallback_batches if b <= configured_batch or b == configured_batch))

    last_oom = None
    for batch_size in fallback_batches:
        run_config = copy.deepcopy(config)
        run_config.setdefault("training", {})["batch_size"] = batch_size

        train_loader = get_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=train_cfg.get("num_workers", 0),
        )
        val_loader = get_dataloader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=train_cfg.get("num_workers", 0),
        )
        eval_loader = get_dataloader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=train_cfg.get("num_workers", 0),
        )

        model = MODEL_CLASSES[args.model](run_config)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Model parameters: total={total_params:,}, trainable={trainable_params:,}, "
            f"batch_size={batch_size}"
        )

        trainer = SpectrogramTrainer(model, run_config, args.model, args.seed, device)
        try:
            results = trainer.train(train_loader, val_loader, eval_loader, resume_path=args.resume)
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or batch_size == fallback_batches[-1]:
                raise
            last_oom = exc
            logger.warning(f"CUDA OOM at batch_size={batch_size}; retrying with a smaller batch")
            del trainer, model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        raise RuntimeError(f"Training failed after batch-size fallback: {last_oom}")

    logger.info(f"Training complete: {results}")


if __name__ == "__main__":
    main()
