"""Run perturbation robustness evaluation for benchmark models.

Usage:
    python experiments/run_perturbation.py --seed 42
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.seed import set_seed
from utils.logger import setup_logger

RETAINED_MODELS = ["random_forest", "xgboost", "simple_cnn", "mobilenetv2", "resnet18"]


def main():
    parser = argparse.ArgumentParser(description="Run perturbation evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument("--test_set", type=str, default="asvspoof2019_eval")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)

    logger = setup_logger("perturbation_eval", config.get("paths", {}).get("logs", "./logs"))
    models = args.models or RETAINED_MODELS

    logger.info("=" * 60)
    logger.info("PERTURBATION ROBUSTNESS EVALUATION")
    logger.info(f"Models: {models}")
    logger.info(f"Test set: {args.test_set}")
    logger.info(f"Seed: {args.seed}")
    logger.info("=" * 60)

    results_dir = config.get("paths", {}).get("results", "./results")
    Path(results_dir).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
