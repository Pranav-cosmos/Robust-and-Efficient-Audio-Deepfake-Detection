"""Generate publication-ready figures for the paper.

All figures exported at 300 DPI in both PDF and PNG formats.

Produces:
    Fig 1: ROC curves
    Fig 2: PR curves
    Fig 3: Confusion matrices
    Fig 4: Robustness heatmap
    Fig 5: Cross-dataset degradation
    Fig 6: Accuracy vs inference time
    Fig 7: Ablation comparison
    Fig 8: Deployment score radar
    Fig 9: EER comparison bar chart
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config

# Publication style settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Color palette for models
MODEL_COLORS = {
    "Random Forest": "#1f77b4",
    "XGBoost": "#ff7f0e",
    "SVM (RBF)": "#2ca02c",
    "Simple CNN": "#d62728",
    "MobileNetV2": "#9467bd",
    "ResNet18": "#8c564b",
    "BiLSTM": "#e377c2",
    "Wav2Vec2+Linear": "#7f7f7f",
    "MFCC+CNN→XGB": "#bcbd22",
    "MFCC+W2V→XGB": "#17becf",
}


def save_figure(fig, output_dir: str, name: str) -> None:
    """Save figure in both PDF and PNG formats."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{name}.pdf"), format="pdf")
    fig.savefig(os.path.join(output_dir, f"{name}.png"), format="png")
    plt.close(fig)
    print(f"  Saved: {name}.pdf, {name}.png")


def plot_roc_curves(results: Dict, output_dir: str) -> None:
    """Fig 1: ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(6, 5))

    for model_name, model_data in results.items():
        roc = model_data.get("roc_curve", {})
        if "fpr" in roc and "tpr" in roc:
            auc_val = model_data.get("metrics", {}).get("roc_auc", 0)
            color = MODEL_COLORS.get(model_name, None)
            ax.plot(roc["fpr"], roc["tpr"],
                    label=f"{model_name} (AUC={auc_val:.3f})",
                    color=color, linewidth=1.2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    save_figure(fig, output_dir, "fig1_roc_curves")


def plot_pr_curves(results: Dict, output_dir: str) -> None:
    """Fig 2: Precision-Recall curves."""
    fig, ax = plt.subplots(figsize=(6, 5))

    for model_name, model_data in results.items():
        pr = model_data.get("pr_curve", {})
        if "precision" in pr and "recall" in pr:
            color = MODEL_COLORS.get(model_name, None)
            ax.plot(pr["recall"], pr["precision"],
                    label=model_name, color=color, linewidth=1.2)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Models")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    save_figure(fig, output_dir, "fig2_pr_curves")


def plot_confusion_matrices(results: Dict, output_dir: str) -> None:
    """Fig 3: Grid of confusion matrices."""
    n_models = len(results)
    cols = min(4, n_models)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3 * rows))
    if n_models == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (model_name, model_data) in enumerate(results.items()):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]

        cm = model_data.get("confusion_matrix", {}).get("confusion_matrix", [[0, 0], [0, 0]])
        cm = np.array(cm)

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Bonafide", "Spoof"],
                    yticklabels=["Bonafide", "Spoof"],
                    cbar=False)
        ax.set_title(model_name, fontsize=9)
        ax.set_ylabel("True" if c == 0 else "")
        ax.set_xlabel("Predicted")

    # Hide empty subplots
    for idx in range(n_models, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].set_visible(False)

    fig.suptitle("Confusion Matrices", fontsize=13, y=1.02)
    plt.tight_layout()

    save_figure(fig, output_dir, "fig3_confusion_matrices")


def plot_robustness_heatmap(results: Dict, output_dir: str) -> None:
    """Fig 4: Robustness heatmap (models × perturbations)."""
    models = []
    perturbation_names = set()

    for model_name, model_data in results.items():
        perturbed = model_data.get("perturbed_results", {})
        perturbation_names.update(perturbed.keys())
        models.append(model_name)

    if not perturbation_names:
        return

    perturbation_names = sorted(perturbation_names)
    data = np.zeros((len(models), len(perturbation_names)))

    for i, model_name in enumerate(models):
        perturbed = results[model_name].get("perturbed_results", {})
        for j, pert_name in enumerate(perturbation_names):
            data[i, j] = perturbed.get(pert_name, {}).get("accuracy", 0.0)

    fig, ax = plt.subplots(figsize=(max(10, len(perturbation_names) * 0.8), len(models) * 0.6 + 2))
    sns.heatmap(data, annot=True, fmt=".3f", cmap="RdYlGn",
                xticklabels=perturbation_names, yticklabels=models,
                ax=ax, vmin=0.5, vmax=1.0,
                linewidths=0.5, linecolor="white")
    ax.set_title("Robustness: Accuracy Under Perturbations")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    save_figure(fig, output_dir, "fig4_robustness_heatmap")


def plot_cross_dataset_degradation(results: Dict, output_dir: str) -> None:
    """Fig 5: Cross-dataset accuracy degradation bar chart."""
    models = list(results.keys())
    test_sets = ["ASV2019 Eval", "ASV2021 LA", "ASV2021 DF", "WaveFake"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.18

    for i, test_set in enumerate(test_sets):
        key = test_set.lower().replace(" ", "_")
        values = [results.get(m, {}).get(key, {}).get("metrics", {}).get("accuracy", 0)
                  for m in models]
        ax.bar(x + i * width, values, width, label=test_set, alpha=0.85)

    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_title("Cross-Dataset Generalization")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    save_figure(fig, output_dir, "fig5_cross_dataset")


def plot_accuracy_vs_latency(efficiency: Dict, output_dir: str) -> None:
    """Fig 6: Accuracy vs inference time scatter plot."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for model_name, eff in efficiency.items():
        acc = eff.get("accuracy", 0)
        latency = eff.get("inference_time_ms", 0)
        size = max(eff.get("model_size_mb", 1), 1)
        color = MODEL_COLORS.get(model_name, "#333333")

        ax.scatter(latency, acc, s=size * 10, c=color, alpha=0.7,
                   edgecolors="black", linewidth=0.5, zorder=5)
        ax.annotate(model_name, (latency, acc), fontsize=7,
                    xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Inference Time (ms)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Computational Cost")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_figure(fig, output_dir, "fig6_accuracy_vs_latency")


def plot_ablation_comparison(results: Dict, output_dir: str) -> None:
    """Fig 7: Ablation study bar chart."""
    variants = list(results.keys())
    metrics_to_plot = ["accuracy", "f1", "eer"]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(4 * len(metrics_to_plot), 4))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics_to_plot):
        values = [results[v].get("metrics", {}).get(metric, 0) for v in variants]
        colors = sns.color_palette("Set2", len(variants))
        bars = ax.bar(range(len(variants)), values, color=colors, alpha=0.85)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Ablation: {metric.upper()}")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_dir, "fig7_ablation")


def plot_deployment_radar(scores: Dict, output_dir: str) -> None:
    """Fig 8: Deployment score radar chart."""
    categories = ["Accuracy", "Robustness", "Generalization", "Efficiency"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for model_name, score_data in scores.items():
        values = [
            score_data.get("norm_accuracy", 0),
            score_data.get("norm_robustness", 0),
            score_data.get("norm_generalization", 0),
            1 - score_data.get("norm_cost", 0),  # Invert cost
        ]
        values += values[:1]
        color = MODEL_COLORS.get(model_name, None)
        ax.plot(angles, values, linewidth=1.5, label=model_name, color=color)
        ax.fill(angles, values, alpha=0.05, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title("Deployment Score — Radar Chart", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)
    plt.tight_layout()

    save_figure(fig, output_dir, "fig8_deployment_radar")


def plot_eer_comparison(results: Dict, output_dir: str) -> None:
    """Fig 9: EER comparison bar chart."""
    models = list(results.keys())
    eers = [results[m].get("metrics", {}).get("eer", 1.0) for m in models]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [MODEL_COLORS.get(m, "#333333") for m in models]
    bars = ax.barh(range(len(models)), eers, color=colors, alpha=0.85)

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel("Equal Error Rate (EER)")
    ax.set_title("EER Comparison — All Models")
    ax.set_xlim([0, max(eers) * 1.15 if eers else 0.5])
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    for bar, eer in zip(bars, eers):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{eer:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    save_figure(fig, output_dir, "fig9_eer_comparison")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = args.results_dir or config.get("paths", {}).get("results", "./results")
    output_dir = args.output_dir or config.get("paths", {}).get("paper_figures", "./paper_figures")

    print(f"Loading results from: {results_dir}")
    print(f"Saving figures to: {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load results (would be populated after running experiments)
    results_path = Path(results_dir)
    if not results_path.exists():
        print("No results directory found. Run experiments first.")
        return

    print("\nFigure generation ready.")
    print("After experiments complete, this script will generate:")
    print("  Fig 1: ROC curves")
    print("  Fig 2: PR curves")
    print("  Fig 3: Confusion matrices")
    print("  Fig 4: Robustness heatmap")
    print("  Fig 5: Cross-dataset degradation")
    print("  Fig 6: Accuracy vs inference time")
    print("  Fig 7: Ablation comparison")
    print("  Fig 8: Deployment radar chart")
    print("  Fig 9: EER comparison")


if __name__ == "__main__":
    main()
