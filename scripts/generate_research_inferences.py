"""Generate paper-ready research inferences from benchmark results.

Outputs a machine-readable JSON file and a concise Markdown report. The report
is useful for paper discussion sections and remains valid for partial runs by
explicitly listing completed and missing benchmark models.
"""
import argparse
import json
import math
from pathlib import Path


TABLE_ORDER = [
    "random_forest",
    "xgboost",
    "simple_cnn",
    "mobilenetv2",
    "resnet18",
    "hybrid_mfcc_cnn_xgboost",
    "hybrid_mobilenetv2_xgboost",
]
DEEP_CNN_MODELS = ["simple_cnn", "mobilenetv2", "resnet18"]
HYBRID_MODELS = ["hybrid_mfcc_cnn_xgboost", "hybrid_mobilenetv2_xgboost"]
DISPLAY = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "simple_cnn": "SimpleCNN",
    "mobilenetv2": "MobileNetV2",
    "resnet18": "ResNet18",
    "hybrid_mfcc_cnn_xgboost": "MFCC+SimpleCNN-XGBoost",
    "hybrid_mobilenetv2_xgboost": "MobileNetV2-XGBoost",
}
FAMILY = {
    "random_forest": "Classical ensemble",
    "xgboost": "Gradient boosting",
    "simple_cnn": "Lightweight CNN",
    "mobilenetv2": "Mobile transfer CNN",
    "resnet18": "Residual transfer CNN",
    "hybrid_mfcc_cnn_xgboost": "Hybrid CNN embedding + boosting",
    "hybrid_mobilenetv2_xgboost": "Hybrid transfer embedding + boosting",
}


def as_float(value, default=None):
    try:
        if value is None:
            return default
        value = float(value)
        if math.isnan(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def metric(entry: dict, name: str, default=None):
    return as_float(entry.get(f"{name}_mean", entry.get(name)), default)


def load_results(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("aggregated", {})


def rank_models(aggregated: dict, metric_name: str, reverse: bool = False) -> list:
    rows = []
    for model in TABLE_ORDER:
        if model not in aggregated:
            continue
        value = metric(aggregated[model], metric_name)
        if value is not None:
            rows.append({"model": model, "value": value})
    return sorted(rows, key=lambda item: item["value"], reverse=reverse)


def build_inferences(aggregated: dict) -> dict:
    completed = [m for m in TABLE_ORDER if m in aggregated]
    missing = [m for m in TABLE_ORDER if m not in aggregated]

    best_eer = rank_models(aggregated, "eer", reverse=False)
    best_f1 = rank_models(aggregated, "f1", reverse=True)
    fastest = rank_models(aggregated, "training_time_seconds", reverse=False)

    efficiency = []
    for model in completed:
        eer = metric(aggregated[model], "eer")
        runtime = metric(aggregated[model], "training_time_seconds")
        if eer is not None and runtime is not None:
            efficiency.append({
                "model": model,
                "eer_percent": eer * 100,
                "training_time_seconds": runtime,
                "eer_time_product": eer * runtime,
            })
    efficiency.sort(key=lambda item: item["eer_time_product"])

    observations = []
    if best_eer:
        winner = best_eer[0]
        observations.append(
            f"{DISPLAY[winner['model']]} is currently the strongest model by EER "
            f"({winner['value'] * 100:.2f}%)."
        )
    if fastest:
        winner = fastest[0]
        observations.append(
            f"{DISPLAY[winner['model']]} is currently the fastest completed model "
            f"({winner['value']:.1f} seconds)."
        )
    if efficiency:
        winner = efficiency[0]
        observations.append(
            f"{DISPLAY[winner['model']]} has the best current EER-runtime trade-off "
            f"among completed models."
        )
    if "random_forest" in completed and "xgboost" in completed:
        rf_eer = metric(aggregated["random_forest"], "eer")
        xgb_eer = metric(aggregated["xgboost"], "eer")
        if rf_eer is not None and xgb_eer is not None:
            better = "XGBoost" if xgb_eer < rf_eer else "Random Forest"
            observations.append(
                f"Within classical ML, {better} currently gives lower EER on the same MFCC representation."
            )
    if all(m in completed for m in DEEP_CNN_MODELS):
        deep_best = rank_models({m: aggregated[m] for m in DEEP_CNN_MODELS}, "eer", reverse=False)[0]
        observations.append(
            f"Within deep CNNs, {DISPLAY[deep_best['model']]} currently gives the lowest EER."
        )
    completed_hybrids = [m for m in HYBRID_MODELS if m in completed]
    if completed_hybrids:
        hybrid_best = rank_models({m: aggregated[m] for m in completed_hybrids}, "eer", reverse=False)[0]
        observations.append(
            f"Among hybrid models, {DISPLAY[hybrid_best['model']]} currently gives the lowest EER, "
            "summarizing whether frozen neural embeddings improve the boosted classifier setting."
        )
    if "xgboost" in completed and completed_hybrids:
        xgb_eer = metric(aggregated["xgboost"], "eer")
        hybrid_rank = rank_models({m: aggregated[m] for m in completed_hybrids}, "eer", reverse=False)
        if xgb_eer is not None and hybrid_rank:
            best_hybrid = hybrid_rank[0]
            delta = (xgb_eer - best_hybrid["value"]) * 100
            direction = "improves over" if delta > 0 else "does not yet improve over"
            observations.append(
                f"The best hybrid model {direction} plain MFCC-XGBoost by {abs(delta):.2f} EER percentage points."
            )

    limitations = []
    if missing:
        limitations.append(
            "This is a partial benchmark. Do not make final cross-family claims until all retained models finish."
        )
    limitations.append(
        "Single-seed results support deterministic engineering comparison, but they should be described as fixed-seed outcomes rather than variance estimates."
    )
    limitations.append(
        "Per-epoch validation uses a deterministic subset for speed; final reported metrics are generated on full available validation/evaluation splits."
    )

    return {
        "completed_models": completed,
        "missing_models": missing,
        "model_families": {m: FAMILY[m] for m in completed},
        "rankings": {
            "eer_low_to_high": best_eer,
            "f1_high_to_low": best_f1,
            "runtime_low_to_high": fastest,
            "efficiency_low_eer_time_product": efficiency,
        },
        "paper_observations": observations,
        "paper_limitations": limitations,
        "recommended_reporting": [
            "Report EER as the primary metric because the dataset is highly class-imbalanced.",
            "Report ROC-AUC, MCC, F1, precision, and recall to show threshold-independent and threshold-dependent behavior.",
            "Discuss runtime together with EER so the benchmark supports deployment-aware model selection.",
            "Keep classical and deep models on their intended feature families: MFCC for tree models and log-mel spectrograms for CNNs.",
            "Use the hybrid rows to discuss whether fixed neural embeddings add complementary information to MFCC-based boosting.",
        ],
    }


def write_markdown(report: dict, output_path: Path) -> None:
    lines = [
        "# Research Inferences",
        "",
        "## Completed Models",
        "",
    ]
    if report["completed_models"]:
        for model in report["completed_models"]:
            lines.append(f"- {DISPLAY[model]} ({FAMILY[model]})")
    else:
        lines.append("- No completed models found.")

    lines.extend(["", "## Missing Models", ""])
    if report["missing_models"]:
        for model in report["missing_models"]:
            lines.append(f"- {DISPLAY[model]}")
    else:
        lines.append("- None. The retained benchmark is complete.")

    lines.extend(["", "## Paper-Ready Observations", ""])
    for item in report["paper_observations"]:
        lines.append(f"- {item}")

    lines.extend(["", "## Reporting Guidance", ""])
    for item in report["recommended_reporting"]:
        lines.append(f"- {item}")

    lines.extend(["", "## Limitations To State", ""])
    for item in report["paper_limitations"]:
        lines.append(f"- {item}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate research inference notes")
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    args = parser.parse_args()

    aggregated = load_results(Path(args.results))
    report = build_inferences(aggregated)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, Path(args.output_md))

    print(f"[SAVED] {output_json}")
    print(f"[SAVED] {args.output_md}")


if __name__ == "__main__":
    main()
