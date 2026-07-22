"""Run evaluation for all retained benchmark models.

Usage:
    python experiments/run_evaluation.py --seed 42
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
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--models", type=str, nargs="+", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)

    logger = setup_logger("run_evaluation", config.get("paths", {}).get("logs", "./logs"))
    all_models = args.models or RETAINED_MODELS

    logger.info(f"Evaluating benchmark models: {all_models}")
    logger.info(f"Seed: {args.seed}")

    # Forwards to aggregate_results
    import subprocess
    cmd = [sys.executable, "evaluation/aggregate_results.py", "--config", args.config]
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
