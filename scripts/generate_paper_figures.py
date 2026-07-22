"""Generate publication-ready figures for the research paper.

Produces:
    Figure 1: EER bar chart across all 5 benchmark models
    Figure 2: Accuracy bar chart
    Figure 3: ROC curves (TPR vs FPR for trained models)
    Figure 4: Multi-metric radar / spider chart
    Figure 5: MFCC feature distribution (Bonafide vs Spoof mean +/- std)
    Figure 6: Efficiency frontier scatter plot (Training Time vs EER %)

Usage:
    python scripts/generate_paper_figures.py
    python scripts/generate_paper_figures.py --results results/all_results.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[ERROR] matplotlib not installed.")
    sys.exit(1)

try:
    from sklearn.metrics import roc_curve, auc
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from utils.config import load_config

# Styling
STYLE = {
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 2.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
}
plt.rcParams.update(STYLE)

PALETTE = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]

MODEL_DISPLAY = {
    "random_forest": "Random Forest",
    "xgboost":       "XGBoost",
    "simple_cnn":    "Simple CNN",
    "mobilenetv2":   "MobileNetV2",
    "resnet18":      "ResNet-18",
}

TABLE_ORDER = ["random_forest", "xgboost", "simple_cnn", "mobilenetv2", "resnet18"]


def save_fig(fig, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    pdf_path = path.replace(".png", ".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")
    print(f"  [SAVED] {pdf_path}")


def fig_eer_bar(aggregated: dict, out_dir: str) -> None:
    """Figure 1: EER bar chart."""
    print("\n[FIG 1] EER bar chart")
    models = [m for m in TABLE_ORDER if m in aggregated]
    if not models:
        return

    names = [MODEL_DISPLAY.get(m, m) for m in models]
    means = [aggregated[m].get("eer_mean", aggregated[m].get("eer", 0)) * 100 for m in models]
    colors = PALETTE[:len(models)]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(models))
    bars = ax.bar(x, means, capsize=5, color=colors, alpha=0.85, edgecolor="white", linewidth=0.8)

    for bar, mean in zip(bars, means):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.2, f"{mean:.2f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Equal Error Rate (%)")
    ax.set_title("Model Comparison: Equal Error Rate (ASVspoof 2019 LA)")
    ax.set_ylim(0, max(means + [5.0]) * 1.25)

    save_fig(fig, os.path.join(out_dir, "fig1_eer_bar.png"))


def fig_accuracy_bar(aggregated: dict, out_dir: str) -> None:
    """Figure 2: Accuracy comparison (horizontal bar)."""
    print("\n[FIG 2] Accuracy comparison")
    models = [m for m in reversed(TABLE_ORDER) if m in aggregated]
    if not models:
        return

    names = [MODEL_DISPLAY.get(m, m) for m in models]
    means = [aggregated[m].get("accuracy_mean", aggregated[m].get("accuracy", 0)) * 100 for m in models]
    colors = [PALETTE[TABLE_ORDER.index(m) % len(PALETTE)] for m in models]

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(models) * 0.6)))
    y = np.arange(len(models))
    bars = ax.barh(y, means, color=colors, alpha=0.85, edgecolor="white", height=0.6)

    for bar, mean in zip(bars, means):
        ax.text(mean + 0.5, bar.get_y() + bar.get_height() / 2, f"{mean:.2f}%",
                va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Model Comparison: Accuracy (ASVspoof 2019 LA)")
    ax.set_xlim(0, 110)

    save_fig(fig, os.path.join(out_dir, "fig2_accuracy_bar.png"))


def fig_roc_curves(results_dir: str, out_dir: str, allowed_models: set | None = None) -> None:
    """Figure 3: ROC curves from saved val_scores.npz files."""
    print("\n[FIG 3] ROC curves")
    if not HAS_SKLEARN:
        return

    scores_files = list(Path(results_dir).rglob("eval_scores.npz"))
    if not scores_files:
        scores_files = list(Path(results_dir).rglob("val_scores.npz"))
    if not scores_files:
        print("  [INFO] No val_scores.npz found. Skipping ROC curves.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")

    for i, scores_file in enumerate(scores_files):
        try:
            data = np.load(scores_file)
            y_true = data["y_true"]
            y_scores = data["y_scores"]
            model_name = scores_file.parent.name
            if allowed_models is not None and model_name not in allowed_models:
                continue

            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            name = MODEL_DISPLAY.get(model_name, model_name)
            ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)],
                    label=f"{name} (AUC={roc_auc:.3f})", linewidth=2)
        except Exception as e:
            print(f"  [WARN] Could not load {scores_file}: {e}")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — ASVspoof 2019 LA Benchmark")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_aspect("equal")

    save_fig(fig, os.path.join(out_dir, "fig3_roc_curves.png"))


def fig_metrics_radar(aggregated: dict, out_dir: str) -> None:
    """Figure 4: Multi-metric radar chart."""
    print("\n[FIG 4] Metrics radar chart")
    metrics = ["accuracy", "f1", "roc_auc", "mcc"]
    metric_labels = ["Accuracy", "F1 Score", "AUC-ROC", "MCC"]
    N = len(metrics)

    models = [m for m in TABLE_ORDER if m in aggregated]
    if not models:
        return

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, size=11)
    ax.set_ylim(0, 1.0)

    for i, model in enumerate(models):
        agg = aggregated[model]
        values = [agg.get(f"{k}_mean", agg.get(k, 0.0)) for k in metrics]
        values += values[:1]
        color = PALETTE[i % len(PALETTE)]
        name = MODEL_DISPLAY.get(model, model)
        ax.plot(angles, values, color=color, linewidth=2, label=name)
        ax.fill(angles, values, color=color, alpha=0.1)

    ax.set_title("Multi-Metric Comparison (Radar Chart)", pad=20, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), framealpha=0.9)

    save_fig(fig, os.path.join(out_dir, "fig4_radar_chart.png"))


def fig_mfcc_visualization(config: dict, out_dir: str) -> None:
    """Figure 5: MFCC feature distribution (Bonafide vs Spoof)."""
    print("\n[FIG 5] MFCC feature visualization")
    cache_dir = config.get("paths", {}).get("feature_cache", "./cache/features")
    train_path = os.path.join(cache_dir, "mfcc", "asvspoof2019_train.npz")

    if not os.path.exists(train_path):
        print(f"  [INFO] Feature cache not found: {train_path}. Skipping MFCC plot.")
        return

    data = np.load(train_path)
    X_key = "X" if "X" in data.files else "features"
    y_key = "y" if "y" in data.files else "labels"
    X = data[X_key]
    y = data[y_key]

    bonafide = X[y == 0]
    spoof = X[y == 1]

    b_mean, b_std = bonafide.mean(axis=0), bonafide.std(axis=0)
    s_mean, s_std = spoof.mean(axis=0), spoof.std(axis=0)

    x_axis = np.arange(X.shape[1])

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(x_axis, b_mean, color=PALETTE[0], label="Bonafide", linewidth=1.5)
    axes[0].fill_between(x_axis, b_mean - b_std, b_mean + b_std, color=PALETTE[0], alpha=0.15)
    axes[0].plot(x_axis, s_mean, color=PALETTE[1], label="Spoof", linewidth=1.5)
    axes[0].fill_between(x_axis, s_mean - s_std, s_mean + s_std, color=PALETTE[1], alpha=0.15)
    axes[0].set_ylabel("Feature Value")
    axes[0].set_title("MFCC Feature Distribution (240-dim Mean + Std)")
    axes[0].legend()

    diff = b_mean - s_mean
    colors_diff = [PALETTE[0] if v >= 0 else PALETTE[1] for v in diff]
    axes[1].bar(x_axis, diff, color=colors_diff, alpha=0.7)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Feature Dimension Index")
    axes[1].set_ylabel("Bonafide − Spoof")
    axes[1].set_title("Mean Feature Difference (Bonafide − Spoof)")

    patch_b = mpatches.Patch(color=PALETTE[0], label="Bonafide > Spoof")
    patch_s = mpatches.Patch(color=PALETTE[1], label="Spoof > Bonafide")
    axes[1].legend(handles=[patch_b, patch_s])

    save_fig(fig, os.path.join(out_dir, "fig5_mfcc_visualization.png"))


def fig_efficiency_scatter(aggregated: dict, out_dir: str) -> None:
    """Figure 6: Efficiency frontier scatter plot (Time vs EER %)."""
    print("\n[FIG 6] Efficiency scatter (Time vs EER %)")
    models = [m for m in TABLE_ORDER if m in aggregated]
    if not models:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, model in enumerate(models):
        agg = aggregated[model]
        t_sec = agg.get("training_time_seconds_mean", agg.get("training_time_seconds", 0))
        eer = agg.get("eer_mean", agg.get("eer", 0)) * 100
        name = MODEL_DISPLAY.get(model, model)
        color = PALETTE[i % len(PALETTE)]

        ax.scatter(t_sec, eer, color=color, s=120, label=name, zorder=5)
        ax.annotate(name, (t_sec, eer), textcoords="offset points", xytext=(6, 4), fontsize=9)

    ax.set_xlabel("Training Time (seconds)")
    ax.set_ylabel("Equal Error Rate (%)")
    ax.set_title("Efficiency Frontier: Training Runtime vs. EER")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    save_fig(fig, os.path.join(out_dir, "fig6_efficiency_scatter.png"))


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--results", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--models", nargs="+", choices=TABLE_ORDER, default=None,
                        help="Only include this completed model subset in score-based figures")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config.get("paths", {})

    results_path = args.results or os.path.join(paths.get("results", "./results"), "all_results.json")
    results_dir = str(Path(results_path).parent) if args.results else paths.get("results", "./results")
    out_dir = args.output_dir or paths.get("paper_figures", "./paper_figures")

    print("=" * 60)
    print("GENERATING PAPER FIGURES")
    print("=" * 60)

    aggregated = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            data = json.load(f)
        aggregated = data.get("aggregated", {})

    fig_eer_bar(aggregated, out_dir)
    fig_accuracy_bar(aggregated, out_dir)
    fig_roc_curves(results_dir, out_dir, set(args.models) if args.models else None)
    fig_metrics_radar(aggregated, out_dir)
    fig_mfcc_visualization(config, out_dir)
    fig_efficiency_scatter(aggregated, out_dir)

    print(f"\n[DONE] Figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
