# Research Inferences

## Completed Models

- Random Forest (Classical ensemble)
- XGBoost (Gradient boosting)
- SimpleCNN (Lightweight CNN)
- MobileNetV2 (Mobile transfer CNN)
- ResNet18 (Residual transfer CNN)
- MFCC+SimpleCNN-XGBoost (Hybrid CNN embedding + boosting)
- MobileNetV2-XGBoost (Hybrid transfer embedding + boosting)

## Missing Models

- None. The retained benchmark is complete.

## Paper-Ready Observations

- ResNet18 is currently the strongest model by EER (5.34%).
- MFCC+SimpleCNN-XGBoost is currently the fastest completed model (0.4 seconds).
- MobileNetV2-XGBoost has the best current EER-runtime trade-off among completed models.
- Within classical ML, XGBoost currently gives lower EER on the same MFCC representation.
- Within deep CNNs, ResNet18 currently gives the lowest EER.
- Among hybrid models, MobileNetV2-XGBoost currently gives the lowest EER, summarizing whether frozen neural embeddings improve the boosted classifier setting.
- The best hybrid model does not yet improve over plain MFCC-XGBoost by 5.01 EER percentage points.

## Reporting Guidance

- Report EER as the primary metric because the dataset is highly class-imbalanced.
- Report ROC-AUC, MCC, F1, precision, and recall to show threshold-independent and threshold-dependent behavior.
- Discuss runtime together with EER so the benchmark supports deployment-aware model selection.
- Keep classical and deep models on their intended feature families: MFCC for tree models and log-mel spectrograms for CNNs.
- Use the hybrid rows to discuss whether fixed neural embeddings add complementary information to MFCC-based boosting.

## Limitations To State

- Single-seed results support deterministic engineering comparison, but they should be described as fixed-seed outcomes rather than variance estimates.
- Per-epoch validation uses a deterministic subset for speed; final reported metrics are generated on full available validation/evaluation splits.
