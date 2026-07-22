"""Run training for a specific benchmark model.

Usage:
    python experiments/run_training.py --model simple_cnn --seed 42
    python experiments/run_training.py --model xgboost --seed 42
"""
import argparse
import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PYTHON = sys.executable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASSICAL = ["random_forest", "xgboost"]
DEEP = ["simple_cnn", "mobilenetv2", "resnet18"]


def main():
    parser = argparse.ArgumentParser(description="Train a specific benchmark model")
    parser.add_argument("--model", type=str, required=True, choices=CLASSICAL + DEEP)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    extra_args = []
    if args.debug:
        extra_args.append("--debug")
    if args.resume:
        extra_args.extend(["--resume", args.resume])

    if args.model in CLASSICAL:
        cmd = [PYTHON, os.path.join(PROJECT_ROOT, "training", "train_classical.py"),
               "--model", args.model, "--seed", str(args.seed),
               "--config", args.config]
        if args.model == "xgboost":
            cmd.append("--use_gpu")
        cmd.extend(extra_args)

    elif args.model in DEEP:
        cmd = [PYTHON, os.path.join(PROJECT_ROOT, "training", "train_deep.py"),
               "--model", args.model, "--seed", str(args.seed),
               "--config", args.config, "--device", args.device] + extra_args

    else:
        print(f"Unknown model: {args.model}")
        sys.exit(1)

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
