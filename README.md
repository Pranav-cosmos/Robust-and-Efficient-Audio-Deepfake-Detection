# Audio Deepfake Detection: Efficient Comparative Benchmark

This repository implements a publication-oriented audio deepfake detection benchmark on the ASVspoof 2019 Logical Access (LA) dataset. The pipeline keeps a focused five-model comparison, uses fixed deterministic execution, and generates checkpoints, metrics, CSV/LaTeX tables, and paper figures from a single command.

The benchmark is optimized for a single modern GPU with an expected full training and reporting runtime under 4 hours, depending on storage speed and GPU class. A separate smoke mode verifies the full execution path in under 15 minutes without overwriting full-run artifacts.

## Retained Models

| Family | Model | Input | Key settings |
| --- | --- | --- | --- |
| Classical ML | Random Forest | 240-dim MFCC mean/std vector | `n_estimators=150`, `max_depth=20`, balanced class weighting, `n_jobs=-1` |
| Classical ML | XGBoost | 240-dim MFCC mean/std vector | `n_estimators=150`, `max_depth=6`, `learning_rate=0.1`, early stopping, CUDA when available |
| Deep Learning | SimpleCNN | 80-band log-mel spectrogram | Channels `16,32,64,128`, FC `128`, dropout `0.3`, BatchNorm |
| Deep Learning | MobileNetV2 | 80-band log-mel spectrogram | ImageNet weights, 1-channel stem, early blocks frozen, width multiplier `1.0` |
| Deep Learning | ResNet18 | 80-band log-mel spectrogram | ImageNet weights, 1-channel stem, initial conv/BN and `layer1` frozen |

Removed model families and dependencies: Wav2Vec2, BiLSTM, SVM, MFCC-Wav2Vec hybrids, and transformer-based feature extraction.

## Features

Only two feature representations are used:

| Feature | Used by | Details |
| --- | --- | --- |
| MFCC | Random Forest, XGBoost | 40 MFCCs plus delta and delta-delta, mean/std pooled to 240 dimensions |
| Log-mel spectrogram | SimpleCNN, MobileNetV2, ResNet18 | 80 mel bands, extracted on the fly from 16 kHz, 4-second audio |

## Training Configuration

| Setting | Value |
| --- | --- |
| Seed | `42` |
| Deep epochs | `12` |
| Early stopping patience | `2` |
| Batch size | `96`, with OOM fallback to `64`, `32`, `16`, then `8` |
| Validation during training | deterministic subset of `4096` dev samples by default |
| Final reporting | full dev validation and official eval split where available |
| Mixed precision | enabled on CUDA |

The seed is applied to Python `random`, NumPy, PyTorch CPU/CUDA, CuDNN deterministic mode, and `PYTHONHASHSEED`.

## Single-Command Execution

Run the complete benchmark:

```bash
python scripts/run_experiments.py
```

Run only selected models while preserving publication artifacts for the completed subset:

```bash
python scripts/run_experiments.py --models random_forest xgboost simple_cnn
```

Run a fast end-to-end smoke test:

```bash
python scripts/run_experiments.py --smoke --device auto
```

The runner refreshes `all_results.json`, paper tables, paper figures, and research inference notes after every successfully completed model. If a later model fails or a subset is requested, completed-model outputs remain available and paper assets reflect the partial benchmark with missing models explicitly noted.

Smoke mode runs feature extraction, all five model trainers, aggregation, table generation, figure generation, and inference-note generation on small deterministic subsets. It writes to tagged locations such as `results/smoke/`, `checkpoints/smoke/`, `logs/smoke/`, and `paper_figures/smoke/`.

## Individual Stages

```bash
python scripts/extract_features.py --config configs/base_config.yaml

python training/train_classical.py --model random_forest --seed 42
python training/train_classical.py --model xgboost --seed 42 --use_gpu

python training/train_deep.py --model simple_cnn --seed 42 --device cuda
python training/train_deep.py --model mobilenetv2 --seed 42 --device cuda
python training/train_deep.py --model resnet18 --seed 42 --device cuda

python evaluation/aggregate_results.py
python scripts/generate_paper_tables.py
python scripts/generate_paper_figures.py
python scripts/generate_research_inferences.py --results results/all_results.json --output_json results/research_inferences.json --output_md results/research_inferences.md
```

## Output Artifacts

| Path | Contents |
| --- | --- |
| `checkpoints/<model>/` | `best.pt`/`last.pt` for deep models, `.pkl` weights and scalers for classical models |
| `results/<model>/metrics.json` | Accuracy, precision, recall, F1, ROC-AUC, MCC, EER, runtime, and split metadata |
| `results/<model>/val_scores.npz` | Dev labels, scores, and predictions for ROC generation |
| `results/<model>/eval_scores.npz` | Eval labels, scores, and predictions when eval data is available |
| `results/<model>/confusion_matrix.png` | Per-model confusion matrix |
| `results/<model>/training_curves.png` | Deep model loss/EER/accuracy curves |
| `results/all_results.json` | Aggregated benchmark metrics |
| `results/paper_tables/` | CSV and LaTeX performance/efficiency tables |
| `paper_figures/` | PNG and PDF paper figures |
| `results/research_inferences.json` | Structured conclusions, rankings, caveats, and reporting guidance |
| `results/research_inferences.md` | Paper-ready inference notes for discussion and limitations |
| `logs/` | Structured stage and training logs |

## Computational Budget

| Model | Expected full runtime |
| --- | ---: |
| Random Forest | under 10 min |
| XGBoost | under 10 min |
| SimpleCNN | about 35-50 min |
| MobileNetV2 | about 45-65 min |
| ResNet18 | about 75-105 min |
| Full pipeline | under 4 hr on a modern CUDA GPU |

Actual runtime depends on GPU, storage throughput, and whether MFCC caches already exist.

## Verification

The repository includes a strict smoke execution path:

```bash
python scripts/run_experiments.py --smoke --device auto
```

The latest local smoke run completed all 9 stages successfully in 74 seconds and generated smoke metrics, tables, figures, logs, and checkpoints.
