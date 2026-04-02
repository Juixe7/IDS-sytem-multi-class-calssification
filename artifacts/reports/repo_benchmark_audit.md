# Repo Benchmark Audit

## What Was Reproduced

- Source file: `C:\Users\ramil\OneDrive\Desktop\New folder\.codex-tmp\IoT-Network-Intrusion-Detection-System-UNSW-NB15\datasets\multi_data.csv`
- Model: RandomForestClassifier(n_estimators=100, random_state=50, n_jobs=1)
- Split: train_test_split(test_size=0.30, random_state=100, stratify=label)

## Findings

- Accuracy with target-derived `attack_cat_*` features left in: **97.27%**
- Macro F1 with target-derived `attack_cat_*` features left in: **0.6865**
- Accuracy after removing those leaked target-derived features: **89.21%**
- Macro F1 after removing those leaked target-derived features: **0.4149**

## Interpretation

- The public 97% benchmark is reproducible on the repo-prepared dataset.
- A large part of that score comes from `attack_cat_*` columns that are deterministic functions of the label being predicted.
- Those columns are valid only after the label is already known, so they are not legitimate raw-input features for a SaaS inference pipeline.
- The production pipeline should keep using non-leaky raw network features only.