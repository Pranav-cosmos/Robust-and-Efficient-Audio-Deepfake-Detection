"""Batch feature extraction and caching.

Extracts MFCC features for ASVspoof 2019 dataset splits and saves to disk cache.
Uses vectorized GPU/CPU batch processing for high throughput.

Usage:
    python scripts/extract_features.py --config configs/base_config.yaml
"""
import argparse
import os
import sys
from pathlib import Path

# Fix OpenMP duplicate-runtime crash on Windows before any torch import
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.seed import set_seed, get_device
from utils.logger import setup_logger
from datasets.asvspoof2019 import ASVspoof2019LA
from datasets.data_utils import get_dataloader
from features.mfcc_extractor import MFCCExtractor
from features.feature_utils import save_features, get_feature_cache_path


def extract_mfcc_features(
    config: dict,
    dataset_name: str,
    split: str,
    dataset,
    cache_dir: str,
    logger,
    device: torch.device,
    debug: bool = False,
) -> None:
    """Extract and cache MFCC utterance-level features (240-dim)."""
    cache_path = get_feature_cache_path(cache_dir, "mfcc", dataset_name, split)

    if os.path.exists(cache_path) and not debug:
        logger.info(f"MFCC features already cached: {cache_path}")
        return

    # In debug mode, sample only 500 audio files
    if debug and len(dataset) > 500:
        dataset = torch.utils.data.Subset(dataset, range(500))

    mfcc_cfg = config.get("mfcc", {})
    extractor = MFCCExtractor(
        n_mfcc=mfcc_cfg.get("n_mfcc", 40),
        n_fft=mfcc_cfg.get("n_fft", 512),
        win_length=mfcc_cfg.get("win_length", 400),
        hop_length=mfcc_cfg.get("hop_length", 160),
        sample_rate=config.get("audio", {}).get("sample_rate", 16000),
        device=device,
    )

    batch_size = 128
    dataloader = get_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_features = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f"MFCC {dataset_name}/{split}")
    for waveforms, labels, _ in pbar:
        batch_feats = extractor.extract_batch_utterance_level(waveforms)
        all_features.append(batch_feats)
        all_labels.extend(labels.numpy())

    X = np.concatenate(all_features, axis=0)
    y = np.array(all_labels)

    if not debug:
        save_features(X, y, cache_path)
        logger.info(f"Saved MFCC features: {cache_path} ({X.shape})")
    else:
        logger.info(f"Debug feature extraction done: ({X.shape})")


def main():
    parser = argparse.ArgumentParser(description="Extract features")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "auto", "cpu"])
    parser.add_argument("--debug", action="store_true", help="Quick debug feature extraction")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("seed", 42)
    set_seed(seed)
    device = get_device(args.device)

    logger = setup_logger("extract_features", config.get("paths", {}).get("logs", "./logs"))
    cache_dir = config.get("paths", {}).get("feature_cache", "./cache/features")

    paths = config.get("paths", {})
    audio_cfg = config.get("audio", {})

    splits = ["train", "dev", "eval"]
    for split in splits:
        try:
            dataset = ASVspoof2019LA(
                root_dir=paths.get("asvspoof2019_root", "./data/ASVspoof2019_LA"),
                split=split,
                sample_rate=audio_cfg.get("sample_rate", 16000),
                max_samples=audio_cfg.get("max_samples", 64000),
            )
            logger.info(f"ASVspoof2019 {split}: {len(dataset)} samples")
            extract_mfcc_features(config, "asvspoof2019", split, dataset, cache_dir, logger, device, debug=args.debug)

        except FileNotFoundError as e:
            logger.warning(f"Skipping ASVspoof2019 {split}: {e}")

    logger.info("Feature extraction complete!")


if __name__ == "__main__":
    main()
