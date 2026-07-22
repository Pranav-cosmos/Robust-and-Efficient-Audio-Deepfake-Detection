"""Generate LaTeX and CSV result tables for the paper.

Produces:
    - Table 1: Clean evaluation (all models × all metrics)
    - Table 2: Cross-dataset results
    - Table 3: Robustness under perturbations
    - Table 4: Ablation study
    - Table 5: Computational efficiency
    - Table 6: Deployment scores
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config


MODEL_DISPLAY_NAMES = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "svm": "SVM (RBF)",
    "simple_cnn": "Simple CNN",
    "mobilenetv2": "MobileNetV2",
    "resnet18": "ResNet18",
    "bilstm": "BiLSTM",
    "wav2vec2_linear": "Wav2Vec2+Linear",
    "hybrid_mfcc_cnn": "MFCC+CNN→XGB",
    "hybrid_mfcc_wav2vec2": "MFCC+W2V→XGB",
    "mfcc_only": "MFCC→XGB",
    "cnn_embed_only": "CNN Emb→XGB",
    "wav2vec2_embed_only": "W2V Emb→XGB",
    "mfcc_cnn": "MFCC+CNN→XGB",
    "mfcc_wav2vec2": "MFCC+W2V→XGB",
}


def load_results(results_dir: str, pattern: str = "*.json") -> Dict:
    """Load all result JSON files from a directory."""
    results = {}
    results_path = Path(results_dir)
    if results_path.exists():
        for f in results_path.rglob(pattern):
            with open(f, "r") as fp:
                key = f.stem
                results[key] = json.load(fp)
    return results


def generate_clean_eval_table(results: Dict, output_dir: str) -> None:
    """Generate Table 1: Clean evaluation results."""
    rows = []
    metrics_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "eer", "mcc"]

    for model_name, model_results in results.items():
        metrics = model_results.get("metrics", {})
        row = {"Model": MODEL_DISPLAY_NAMES.get(model_name, model_name)}
        for m in metrics_cols:
            val = metrics.get(m, 0.0)
            row[m.upper()] = f"{val:.4f}" if isinstance(val, float) else str(val)
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, "table1_clean_eval.csv"), index=False)
        # Generate LaTeX
        latex = df.to_latex(index=False, caption="Clean Audio Evaluation Results",
                           label="tab:clean_eval", escape=False)
        with open(os.path.join(output_dir, "table1_clean_eval.tex"), "w") as f:
            f.write(latex)


def generate_cross_dataset_table(results: Dict, output_dir: str) -> None:
    """Generate Table 2: Cross-dataset evaluation."""
    rows = []
    test_sets = ["ASV2019 Eval", "ASV2021 LA", "ASV2021 DF", "WaveFake"]

    for model_name, model_results in results.items():
        row = {"Model": MODEL_DISPLAY_NAMES.get(model_name, model_name)}
        for test_set in test_sets:
            key = test_set.lower().replace(" ", "_")
            val = model_results.get(key, {}).get("metrics", {}).get("accuracy", "-")
            row[test_set] = f"{val:.4f}" if isinstance(val, float) else str(val)
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, "table2_cross_dataset.csv"), index=False)
        latex = df.to_latex(index=False, caption="Cross-Dataset Evaluation",
                           label="tab:cross_dataset", escape=False)
        with open(os.path.join(output_dir, "table2_cross_dataset.tex"), "w") as f:
            f.write(latex)


def generate_robustness_table(results: Dict, output_dir: str) -> None:
    """Generate Table 3: Robustness under perturbations."""
    rows = []

    for model_name, model_results in results.items():
        row = {"Model": MODEL_DISPLAY_NAMES.get(model_name, model_name)}
        perturbed = model_results.get("perturbed_results", {})

        for pert_name, pert_metrics in perturbed.items():
            acc = pert_metrics.get("accuracy", 0.0)
            row[pert_name] = f"{acc:.4f}"

        rob_score = model_results.get("robustness", {}).get("robustness_score", 0.0)
        row["Robustness"] = f"{rob_score:.4f}"
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, "table3_robustness.csv"), index=False)
        latex = df.to_latex(index=False, caption="Robustness Under Perturbations",
                           label="tab:robustness", escape=False)
        with open(os.path.join(output_dir, "table3_robustness.tex"), "w") as f:
            f.write(latex)


def generate_ablation_table(results: Dict, output_dir: str) -> None:
    """Generate Table 4: Ablation study results."""
    rows = []
    metrics_cols = ["accuracy", "f1", "eer", "roc_auc"]

    for variant_name, variant_results in results.items():
        metrics = variant_results.get("metrics", variant_results.get("validation_metrics", {}))
        row = {"Configuration": MODEL_DISPLAY_NAMES.get(variant_name, variant_name)}
        for m in metrics_cols:
            val = metrics.get(m, 0.0)
            row[m.upper()] = f"{val:.4f}" if isinstance(val, float) else str(val)
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, "table4_ablation.csv"), index=False)
        latex = df.to_latex(index=False, caption="Ablation Study Results",
                           label="tab:ablation", escape=False)
        with open(os.path.join(output_dir, "table4_ablation.tex"), "w") as f:
            f.write(latex)


def generate_efficiency_table(results: Dict, output_dir: str) -> None:
    """Generate Table 5: Computational efficiency comparison."""
    rows = []

    for model_name, eff in results.items():
        row = {
            "Model": MODEL_DISPLAY_NAMES.get(model_name, model_name),
            "Params": f"{eff.get('total_params', 0):,}",
            "Size (MB)": f"{eff.get('model_size_mb', 0):.1f}",
            "Inf. Time (ms)": f"{eff.get('inference_time_ms', 0):.2f}",
            "Train Time (s)": f"{eff.get('training_time_seconds', 0):.0f}",
            "GPU Mem (MB)": f"{eff.get('gpu_memory_mb', 0):.0f}",
        }
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, "table5_efficiency.csv"), index=False)
        latex = df.to_latex(index=False, caption="Computational Efficiency",
                           label="tab:efficiency", escape=False)
        with open(os.path.join(output_dir, "table5_efficiency.tex"), "w") as f:
            f.write(latex)


def generate_deployment_table(scores: Dict, output_dir: str) -> None:
    """Generate Table 6: Deployment scores."""
    rows = []

    for model_name, score_data in sorted(
        scores.items(), key=lambda x: x[1].get("rank", 99)
    ):
        row = {
            "Rank": score_data.get("rank", "-"),
            "Model": MODEL_DISPLAY_NAMES.get(model_name, model_name),
            "Score": f"{score_data.get('deployment_score', 0):.4f}",
            "Accuracy": f"{score_data.get('raw_accuracy', 0):.4f}",
            "Robustness": f"{score_data.get('raw_robustness', 0):.4f}",
            "Generalization": f"{score_data.get('raw_generalization', 0):.4f}",
            "Cost (ms)": f"{score_data.get('raw_cost_ms', 0):.2f}",
        }
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, "table6_deployment.csv"), index=False)
        latex = df.to_latex(index=False, caption="Deployment Score Rankings",
                           label="tab:deployment", escape=False)
        with open(os.path.join(output_dir, "table6_deployment.tex"), "w") as f:
            f.write(latex)


def main():
    parser = argparse.ArgumentParser(description="Generate result tables")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = args.results_dir or config.get("paths", {}).get("results", "./results")
    output_dir = args.output_dir or os.path.join(results_dir, "tables")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_dir}")
    print(f"Saving tables to: {output_dir}")

    # Load and generate tables
    results = load_results(results_dir)
    if results:
        generate_clean_eval_table(results, output_dir)
        generate_cross_dataset_table(results, output_dir)
        generate_robustness_table(results, output_dir)
        generate_ablation_table(results, output_dir)
        generate_efficiency_table(results, output_dir)
        generate_deployment_table(results, output_dir)
        print(f"Tables generated in {output_dir}")
    else:
        print("No results found. Run experiments first.")


if __name__ == "__main__":
    main()
