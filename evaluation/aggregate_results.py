"""Aggregate all model training results into a single JSON for paper generation.

Scans ./results/ and ./checkpoints/ for metrics.json / training_info.json files
and merges them into ./results/all_results.json.

Usage:
    python evaluation/aggregate_results.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config

RETAINED_MODELS = [
    "random_forest",
    "xgboost",
    "simple_cnn",
    "mobilenetv2",
    "resnet18",
    "hybrid_mfcc_cnn_xgboost",
    "hybrid_mobilenetv2_xgboost",
]


def scan_results(results_dirs: list[str], checkpoints_dir: str, allowed_models: list | None = None) -> dict:
    """Scan metrics.json and training_info.json files."""
    results = {}
    allowed = set(allowed_models) if allowed_models else set(RETAINED_MODELS)

    # Scan result directories in order. Later directories intentionally replace
    # earlier ones for the same model so additive runs can refresh one model.
    for results_dir in results_dirs:
        r_root = Path(results_dir)
        if r_root.exists():
            for metrics_file in sorted(r_root.rglob("metrics.json")):
                try:
                    with open(metrics_file) as f:
                        data = json.load(f)
                    model_name = data.get("model", metrics_file.parent.name)
                    if model_name in allowed:
                        results[model_name] = data
                        print(f"  [OK] Found results for {model_name} from {metrics_file}")
                except Exception as e:
                    print(f"  [WARN] Could not parse {metrics_file}: {e}")

    # Fallback to checkpoints_dir if missing
    c_root = Path(checkpoints_dir)
    if c_root.exists():
        for info_file in sorted(c_root.rglob("training_info.json")):
            try:
                with open(info_file) as f:
                    data = json.load(f)
                model_name = data.get("model", info_file.parent.name)
                if model_name not in results and model_name in allowed:
                    entry = {**data, **data.get("validation_metrics", {})}
                    if "best_eer" in data:
                        entry["eer"] = data["best_eer"]
                    results[model_name] = entry
                    print(f"  [OK] Found checkpoint info for {model_name} from {info_file}")
            except Exception as e:
                print(f"  [WARN] Could not parse {info_file}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Aggregate training results for paper")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--results_dir", type=str, action="append", default=None,
                        help="Result directory to scan. Can be passed multiple times; later directories win.")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--models", nargs="+", choices=RETAINED_MODELS, default=None,
                        help="Only aggregate this completed model subset")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config.get("paths", {})

    results_dirs = args.results_dir or [paths.get("results", "./results")]
    checkpoints_dir = paths.get("checkpoints", "./checkpoints")
    output_path = args.output or os.path.join(results_dirs[-1], "all_results.json")

    print("=" * 60)
    print("AGGREGATING TRAINING RESULTS")
    print("=" * 60)
    print(f"\nScanning results: {results_dirs}")
    print(f"Scanning checkpoints fallback: {checkpoints_dir}")

    models_data = scan_results(results_dirs, checkpoints_dir, args.models)

    if not models_data:
        print("\n[WARNING] No training results found. Run training first.")
        return

    print(f"\nFound results for {len(models_data)} model(s): {list(models_data.keys())}")

    # Convert single-run metrics to mean keys for seamless table generator compatibility
    aggregated = {}
    for model_name, data in models_data.items():
        agg_entry = {**data}
        metric_keys = ["eer", "accuracy", "precision", "recall", "f1", "roc_auc", "mcc", "training_time_seconds"]
        for key in metric_keys:
            if key in data and data[key] is not None:
                agg_entry[f"{key}_mean"] = data[key]
                agg_entry[f"{key}_std"] = 0.0
        agg_entry["n_seeds"] = 1
        aggregated[model_name] = agg_entry

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "per_model": models_data,
        "aggregated": aggregated,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[SAVED] {output_path}")
    print("\nSummary:")
    print(f"  {'Model':<25} {'EER (%)':>10} {'Accuracy (%)':>15} {'F1-Score':>12} {'Time (s)':>12}")
    print("  " + "-" * 78)
    for model_name, agg in aggregated.items():
        eer = agg.get("eer", float("nan")) * 100
        acc = agg.get("accuracy", float("nan")) * 100
        f1 = agg.get("f1", float("nan"))
        t_sec = agg.get("training_time_seconds", 0.0)
        print(f"  {model_name:<25} {eer:>10.2f}% {acc:>14.2f}% {f1:>12.4f} {t_sec:>11.1f}s")


if __name__ == "__main__":
    main()
