"""Generate publication-ready LaTeX and CSV tables from aggregated benchmark results.

Produces:
    Table 1: Model performance comparison (EER %, Accuracy %, F1, AUC-ROC, MCC)
    Table 2: Computational efficiency (Parameters, Model Size MB, Training Time)

Output formats: LaTeX (.tex) and CSV (.csv) for each table.

Usage:
    python scripts/generate_paper_tables.py
    python scripts/generate_paper_tables.py --results results/all_results.json
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config

# Human-readable display names for the 5 retained benchmark models
MODEL_DISPLAY_NAMES = {
    "random_forest": "Random Forest",
    "xgboost":       "XGBoost",
    "simple_cnn":    "Simple CNN",
    "mobilenetv2":   "MobileNetV2",
    "resnet18":      "ResNet-18",
}

MODEL_CATEGORIES = {
    "random_forest": "Classical Ensemble",
    "xgboost":       "Gradient Boosting",
    "simple_cnn":    "Lightweight CNN",
    "mobilenetv2":   "Efficient Mobile CNN",
    "resnet18":      "Residual CNN",
}

TABLE_ORDER = ["random_forest", "xgboost", "simple_cnn", "mobilenetv2", "resnet18"]


def write_latex_table(rows: list, headers: list, caption: str, label: str, output_path: str) -> None:
    """Write a LaTeX booktabs table."""
    n_cols = len(headers)
    col_spec = "l" + "r" * (n_cols - 1)

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \begin{tabular}{" + col_spec + "}",
        r"    \toprule",
        "    " + " & ".join(f"\\textbf{{{h}}}" for h in headers) + r" \\",
        r"    \midrule",
    ]

    last_category = None
    for row in rows:
        if len(row) > n_cols:
            category = row[0]
            row = row[1:]
            if category != last_category:
                if last_category is not None:
                    lines.append(r"    \midrule")
                lines.append(f"    \\multicolumn{{{n_cols}}}{{l}}{{\\textit{{{category}}}}} \\\\")
                last_category = category

        lines.append("    " + " & ".join(str(c) for c in row) + r" \\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [LaTeX] {output_path}")


def write_csv_table(rows: list, headers: list, output_path: str) -> None:
    """Write a CSV table."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    n_headers = len(headers)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            if len(row) == n_headers + 1:
                row = row[1:]
            writer.writerow(row)
    print(f"  [CSV]   {output_path}")


def generate_table1_performance(aggregated: dict, out_dir: str) -> None:
    """Table 1: Model performance comparison."""
    print("\n[TABLE 1] Performance comparison")

    headers = ["Model", "EER (%)", "Accuracy (%)", "F1 Score", "AUC-ROC", "MCC"]
    rows = []

    for model_key in TABLE_ORDER:
        if model_key not in aggregated:
            continue
        agg = aggregated[model_key]
        name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        category = MODEL_CATEGORIES.get(model_key, "Benchmark")

        eer_val = agg.get("eer_mean", agg.get("eer"))
        acc_val = agg.get("accuracy_mean", agg.get("accuracy"))
        f1_val = agg.get("f1_mean", agg.get("f1"))
        auc_val = agg.get("roc_auc_mean", agg.get("roc_auc"))
        mcc_val = agg.get("mcc_mean", agg.get("mcc"))

        def fmv(val, pct=False, fmt=".4f"):
            if val is None:
                return "---"
            if pct:
                return f"{val * 100:.2f}"
            return f"{val:{fmt}}"

        rows.append([
            category,
            name,
            fmv(eer_val, pct=True),
            fmv(acc_val, pct=True),
            fmv(f1_val),
            fmv(auc_val),
            fmv(mcc_val),
        ])

    caption = (
        "Comparative evaluation of deepfake detection models on ASVspoof 2019 LA evaluation set. "
        "EER (\\%) lower is better; all other metrics higher is better."
    )
    write_latex_table(rows, headers, caption, "tab:performance", os.path.join(out_dir, "table1_performance.tex"))
    write_csv_table(rows, headers, os.path.join(out_dir, "table1_performance.csv"))


def generate_table2_efficiency(aggregated: dict, out_dir: str) -> None:
    """Table 2: Training efficiency."""
    print("\n[TABLE 2] Training efficiency")

    headers = ["Model", "Parameters", "Size (MB)", "Train Time (s)", "GPU Acceleration"]
    rows = []

    for model_key in TABLE_ORDER:
        if model_key not in aggregated:
            continue
        agg = aggregated[model_key]
        name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        category = MODEL_CATEGORIES.get(model_key, "Benchmark")

        params = agg.get("parameters")
        size = agg.get("model_size_mb")
        time_sec = agg.get("training_time_seconds_mean", agg.get("training_time_seconds"))

        if isinstance(params, dict):
            params = params.get("total_params")
        params_str = f"{params:,}" if params else "---"
        size_str = f"{size:.2f}" if size else "---"
        time_str = f"{time_sec:.1f}" if time_sec else "---"
        is_gpu = "CPU Multi-thread" if model_key == "random_forest" else "CUDA Enabled"

        rows.append([category, name, params_str, size_str, time_str, is_gpu])

    caption = (
        "Computational efficiency breakdown including parameter counts, disk storage size, "
        "and training runtime on an NVIDIA GPU environment."
    )
    write_latex_table(rows, headers, caption, "tab:efficiency", os.path.join(out_dir, "table2_efficiency.tex"))
    write_csv_table(rows, headers, os.path.join(out_dir, "table2_efficiency.csv"))


def main():
    parser = argparse.ArgumentParser(description="Generate paper tables")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--results", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config.get("paths", {})

    results_path = args.results or os.path.join(paths.get("results", "./results"), "all_results.json")
    out_dir = args.output_dir or os.path.join(paths.get("results", "./results"), "paper_tables")

    print("=" * 60)
    print("GENERATING PAPER TABLES")
    print("=" * 60)
    print(f"\nLoading results: {results_path}")

    if not os.path.exists(results_path):
        print(f"[ERROR] Results file not found: {results_path}")
        sys.exit(1)

    with open(results_path) as f:
        data = json.load(f)

    aggregated = data.get("aggregated", {})
    print(f"  Models found: {list(aggregated.keys())}")

    generate_table1_performance(aggregated, out_dir)
    generate_table2_efficiency(aggregated, out_dir)

    print(f"\n[DONE] Tables saved to: {out_dir}")


if __name__ == "__main__":
    main()
