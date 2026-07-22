"""Master experiment runner for the complete benchmark pipeline.

The runner is intentionally crash-resilient: after every successfully trained
model it refreshes aggregate metrics, paper tables, paper figures, and research
inference notes. If a later model fails or the user runs only a subset, all
completed-model artifacts remain usable for paper drafting.

Usage:
    python scripts/run_experiments.py
    python scripts/run_experiments.py --models random_forest xgboost
    python scripts/run_experiments.py --smoke --device auto
    python scripts/run_experiments.py --skip_features
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

CLASSICAL_MODELS = ["random_forest", "xgboost"]
DEEP_MODELS = ["simple_cnn", "mobilenetv2", "resnet18"]
ALL_MODELS = CLASSICAL_MODELS + DEEP_MODELS


def run(cmd: list, label: str, log_dir: str) -> tuple:
    """Run a subprocess command and stream output to console and log file."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, f"{label.replace(' ', '_')}.txt")

    print(f"\n{'=' * 65}")
    print(f"  RUNNING STAGE: {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'=' * 65}", flush=True)

    start = time.time()
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE", "PYTHONIOENCODING": "utf-8"}

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        bufsize=1,
    )

    with open(log_file, "w", encoding="utf-8") as lf:
        for line in proc.stdout:
            stdout_encoding = sys.stdout.encoding or "utf-8"
            safe_line = line.encode(stdout_encoding, errors="replace").decode(stdout_encoding)
            sys.stdout.write(safe_line)
            sys.stdout.flush()
            lf.write(line)

    proc.wait()
    elapsed = time.time() - start
    success = proc.returncode == 0

    if success:
        print(f"\n  [OK] Stage '{label}' completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)", flush=True)
    else:
        print(f"\n  [FAILED] Stage '{label}' exit code {proc.returncode} - see log: {log_file}", flush=True)

    return success, elapsed


def result_paths(project_root: Path, artifact_tag: str | None) -> dict:
    result_root = project_root / "results" / artifact_tag if artifact_tag else project_root / "results"
    figure_root = project_root / "paper_figures" / artifact_tag if artifact_tag else project_root / "paper_figures"
    return {
        "results_dir": result_root,
        "all_results": result_root / "all_results.json",
        "tables_dir": result_root / "paper_tables",
        "figures_dir": figure_root,
        "inference_json": result_root / "research_inferences.json",
        "inference_md": result_root / "research_inferences.md",
        "summary": result_root / "runner_summary.json",
    }


def write_summary(project_root: Path, summary: dict, start_time: float, artifact_tag: str | None) -> None:
    summary["total_time_seconds"] = time.time() - start_time
    paths = result_paths(project_root, artifact_tag)
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    with open(paths["summary"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved: {paths['summary']}")


def build_publish_commands(
    args,
    project_root: Path,
    artifact_tag: str | None,
    completed_models: list,
) -> list:
    paths = result_paths(project_root, artifact_tag)
    commands = [
        (
            "aggregate_results",
            [
                args.python,
                str(project_root / "evaluation" / "aggregate_results.py"),
                "--config",
                args.config,
                "--results_dir",
                str(paths["results_dir"]),
                "--output",
                str(paths["all_results"]),
                "--models",
                *completed_models,
            ],
        ),
        (
            "generate_paper_tables",
            [
                args.python,
                str(project_root / "scripts" / "generate_paper_tables.py"),
                "--config",
                args.config,
                "--results",
                str(paths["all_results"]),
                "--output_dir",
                str(paths["tables_dir"]),
            ],
        ),
        (
            "generate_paper_figures",
            [
                args.python,
                str(project_root / "scripts" / "generate_paper_figures.py"),
                "--config",
                args.config,
                "--results",
                str(paths["all_results"]),
                "--output_dir",
                str(paths["figures_dir"]),
                "--models",
                *completed_models,
            ],
        ),
        (
            "generate_research_inferences",
            [
                args.python,
                str(project_root / "scripts" / "generate_research_inferences.py"),
                "--results",
                str(paths["all_results"]),
                "--output_json",
                str(paths["inference_json"]),
                "--output_md",
                str(paths["inference_md"]),
            ],
        ),
    ]
    return commands


def publish_partial_artifacts(
    args,
    project_root: Path,
    artifact_tag: str | None,
    logs_dir: str,
    reason: str,
    completed_models: list,
) -> tuple:
    """Refresh global paper artifacts using whatever model outputs exist now."""
    if not completed_models:
        print("\n[ARTIFACT REFRESH] No completed models yet; skipping paper asset refresh.")
        return True, 0.0

    print(f"\n[ARTIFACT REFRESH] Updating paper assets after {reason}.")
    elapsed_total = 0.0
    all_ok = True

    for name, cmd in build_publish_commands(args, project_root, artifact_tag, completed_models):
        success, elapsed = run(cmd, f"artifact_{reason}_{name}", logs_dir)
        elapsed_total += elapsed
        all_ok = all_ok and success
        if not success:
            break

    return all_ok, elapsed_total


def main():
    parser = argparse.ArgumentParser(description="Run complete deepfake detection benchmark pipeline")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "auto", "cpu"])
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS,
                        help="Subset of retained models to train in benchmark order")
    parser.add_argument("--skip_features", action="store_true", help="Skip MFCC feature extraction")
    parser.add_argument("--debug", action="store_true", help="Debug mode: 2 epochs, fast subset")
    parser.add_argument("--smoke", action="store_true",
                        help="Run all stages on tiny deterministic subsets for a sub-15-minute sanity check")
    parser.add_argument("--continue_on_error", action="store_true",
                        help="Continue later stages after a failure instead of failing fast")
    parser.add_argument("--no_incremental_artifacts", action="store_true",
                        help="Only generate aggregate paper artifacts at the end")
    parser.add_argument("--python", type=str, default=sys.executable)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    artifact_tag = "smoke" if args.smoke else None
    logs_dir = str(project_root / "logs" / (artifact_tag or "") / "experiment_runner")
    selected_models = [m for m in ALL_MODELS if m in set(args.models)]

    summary = {
        "config": vars(args),
        "selected_models": selected_models,
        "results": {},
        "completed_models": [],
        "failed_models": [],
    }
    total_start = time.time()

    def record_stage(key: str, success: bool, elapsed: float) -> None:
        summary["results"][key] = {"success": success, "time_s": elapsed}
        write_summary(project_root, summary, total_start, artifact_tag)
        if not success and not args.continue_on_error:
            print(f"[ERROR] Stage '{key}' failed. Stopping pipeline.")
            sys.exit(1)

    if not args.skip_features:
        feat_cmd = [
            args.python,
            str(project_root / "scripts" / "extract_features.py"),
            "--config",
            args.config,
            "--device",
            args.device,
        ]
        if args.debug or args.smoke:
            feat_cmd.append("--debug")
        success, elapsed = run(feat_cmd, "01_feature_extraction", logs_dir)
        record_stage("feature_extraction", success, elapsed)
    else:
        print("[SKIP] Feature extraction")

    for model in selected_models:
        prefix = "02" if model in CLASSICAL_MODELS else "03"
        label = f"{prefix}_train_{model}"

        if model in CLASSICAL_MODELS:
            cmd = [
                args.python,
                str(project_root / "training" / "train_classical.py"),
                "--model",
                model,
                "--seed",
                str(args.seed),
                "--config",
                args.config,
            ]
            if model == "xgboost":
                cmd.append("--use_gpu")
            if args.debug or args.smoke:
                cmd.append("--debug")
            if artifact_tag:
                cmd.extend(["--artifact_tag", artifact_tag])
            if args.smoke:
                cmd.extend([
                    "--max_train_samples", "1000",
                    "--max_val_samples", "500",
                    "--max_eval_samples", "500",
                ])
        else:
            cmd = [
                args.python,
                str(project_root / "training" / "train_deep.py"),
                "--model",
                model,
                "--seed",
                str(args.seed),
                "--config",
                args.config,
                "--device",
                args.device,
            ]
            if args.debug or args.smoke:
                cmd.append("--debug")
            if artifact_tag:
                cmd.extend(["--artifact_tag", artifact_tag])
            if args.smoke:
                cmd.extend([
                    "--max_train_samples", "128",
                    "--max_val_samples", "64",
                    "--max_eval_samples", "64",
                ])

        success, elapsed = run(cmd, label, logs_dir)
        record_stage(label, success, elapsed)

        if success:
            summary["completed_models"].append(model)
            write_summary(project_root, summary, total_start, artifact_tag)
            if not args.no_incremental_artifacts:
                artifact_success, artifact_elapsed = publish_partial_artifacts(
                    args, project_root, artifact_tag, logs_dir, model, summary["completed_models"]
                )
                record_stage(f"artifacts_after_{model}", artifact_success, artifact_elapsed)
        else:
            summary["failed_models"].append(model)
            write_summary(project_root, summary, total_start, artifact_tag)
            if not args.continue_on_error:
                sys.exit(1)

    final_success, final_elapsed = publish_partial_artifacts(
        args, project_root, artifact_tag, logs_dir, "final", summary["completed_models"]
    )
    record_stage("final_artifacts", final_success, final_elapsed)

    total_elapsed = time.time() - total_start
    passed = sum(1 for v in summary["results"].values() if v.get("success"))
    failed = len(summary["results"]) - passed

    print(f"\n{'=' * 65}")
    print("PIPELINE EXPERIMENT RUNNER COMPLETE")
    print(f"  Total Runtime: {total_elapsed / 3600:.2f} hours ({total_elapsed:.0f} seconds)")
    print(f"  Completed Models: {summary['completed_models']}")
    print(f"  Stages Passed: {passed} / {len(summary['results'])}")
    print(f"{'=' * 65}")

    write_summary(project_root, summary, total_start, artifact_tag)

    if failed > 0:
        print(f"\n[WARNING] {failed} stage(s) failed during execution.")
        sys.exit(1)


if __name__ == "__main__":
    main()
