# Phase 2 Performance Upgrade Report

## Summary

This report logs the controlled Phase 2 upgrade applied after the leakage and preprocessing fixes. The goal was to improve multiclass robustness, especially Macro F1, without changing pipeline structure, public APIs, file organization, or evaluation flow.

The upgrade implemented four targeted changes:

- increased `feature_cap` from `25` to `50`
- kept model-level class weighting for `lr`, `rf`, and `lgbm`
- added explicit XGBoost sample weights during fit
- added SMOTE on training data only
- calibrated only the selected production `final_model`

Execution command:

```powershell
.\.venv-py311\Scripts\python.exe main.py --mode train_all --retrain-final-model
```

## Code Changes Logged

### Configuration and preprocessing

- [config.py](C:\Users\ramil\OneDrive\Desktop\New folder\src\config.py)
  - `feature_cap` changed from `25` to `50`

- [preprocess.py](C:\Users\ramil\OneDrive\Desktop\New folder\src\preprocess.py)
  - retained safe feature-selection guard:
    - `k = min(config.feature_cap, X_train_processed.shape[1])`
  - added explicit logs for:
    - features before selection
    - feature cap value
    - features after selection
    - `id` presence checks
    - duplicate counts

### Training-time balancing and weighting

- [train.py](C:\Users\ramil\OneDrive\Desktop\New folder\src\train.py)
  - added `_class_distribution(...)`
  - added `_apply_smote(...)`
  - added `_compute_xgb_sample_weight(...)`
  - added `_fit_estimator(...)`
  - added `train_calibrated_final_model(...)`
  - applied SMOTE only to training subsets:
    - CV train folds for `lr` and `rf`
    - inner-train only for `xgb` and `lgbm`
    - final full-train or fit-train only
  - kept `class_weight="balanced"` for:
    - Logistic Regression
    - Random Forest
    - LightGBM
  - applied explicit `sample_weight` to XGBoost fits

### Final-model calibration

- [main.py](C:\Users\ramil\OneDrive\Desktop\New folder\main.py)
  - preserved offline multi-model comparison
  - calibrated only the selected winner after model comparison
  - saved the calibrated production artifact to:
    - [final_model.pkl](C:\Users\ramil\OneDrive\Desktop\New folder\models\final_model.pkl)
  - saved calibrated final-model metrics into:
    - [final_summary.json](C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\final_summary.json)

- [evaluate.py](C:\Users\ramil\OneDrive\Desktop\New folder\src\evaluate.py)
  - added a lightweight final-model-only evaluation helper
  - left multi-model comparison logic unchanged

### Calibration-compatible explainability and inference

- [inference.py](C:\Users\ramil\OneDrive\Desktop\New folder\src\inference.py)
  - added safe unwrapping for calibrated estimators during explanation
  - left prediction and probability behavior unchanged

- [explain.py](C:\Users\ramil\OneDrive\Desktop\New folder\src\explain.py)
  - added safe unwrapping for calibrated estimators in SHAP generation

### Dependency record

- [requirements.txt](C:\Users\ramil\OneDrive\Desktop\New folder\requirements.txt)
  - added `imbalanced-learn==0.12.3`

## Validation Logs

### Feature checks

- features before selection: `184`
- feature cap set to: `50`
- features after selection: `50`
- `id` in raw feature columns: `False`
- `id` in transformed feature names: `False`
- `id` in selected feature names: `False`

### Duplicate checks

- train duplicates removed after `id` drop: `26,387`
- test duplicates removed after `id` drop: `67,601`

### SMOTE checks

Example final production-model training path:

- before SMOTE: `X=(55945, 50)`, `y=(55945,)`
- after SMOTE: `X=(342060, 50)`, `y=(342060,)`

Class distribution before SMOTE:

```text
{0: 446, 1: 346, 2: 1718, 3: 7609, 4: 4838, 5: 3657, 6: 34206, 7: 2703, 8: 378, 9: 44}
```

Class distribution after SMOTE:

```text
{0: 34206, 1: 34206, 2: 34206, 3: 34206, 4: 34206, 5: 34206, 6: 34206, 7: 34206, 8: 34206, 9: 34206}
```

### Model and calibration checks

- `lr` class weighting: applied
- `rf` class weighting: applied
- `lgbm` class weighting: applied
- `xgb` sample weighting: applied
- saved production model type: `CalibratedClassifierCV`
- SHAP still executed successfully for the calibrated production model
- inference still returned predictions and probabilities successfully

## Metrics After Upgrade

Offline comparison metrics from [model_metrics.json](C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\model_metrics.json):

| Model | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| `LR` | 68.53% | 0.4320 | 0.7122 |
| `RF` | 74.11% | 0.4999 | 0.7115 |
| `XGB` | 72.85% | 0.4964 | 0.7037 |
| `LGBM` | 73.74% | 0.5248 | 0.7010 |

Winning offline comparison model:

- `LGBM`

Calibrated production `final_model` metrics from [final_summary.json](C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\final_summary.json):

- accuracy: `73.49%`
- macro F1: `0.4992`
- weighted F1: `0.6890`

## Per-Class F1 for Calibrated Production Model

| Class | F1-score |
| --- | ---: |
| `analysis` | 0.1979 |
| `backdoor` | 0.1899 |
| `dos` | 0.2838 |
| `exploits` | 0.8061 |
| `fuzzers` | 0.1390 |
| `generic` | 0.8398 |
| `normal` | 0.8568 |
| `reconnaissance` | 0.7538 |
| `shellcode` | 0.5695 |
| `worms` | 0.3556 |

## Before vs After Comparison

Previous reliable post-fix baseline:

- selected model: `LGBM`
- accuracy: `71.71%`
- macro F1: `0.4341`
- weighted F1: `0.6982`

Offline comparison winner after Phase 2:

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Accuracy | 71.71% | 73.74% | `+2.03` pts |
| Macro F1 | 0.4341 | 0.5248 | `+0.0907` |
| Weighted F1 | 0.6982 | 0.7010 | `+0.0028` |

Deployed calibrated production model after Phase 2:

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Accuracy | 71.71% | 73.49% | `+1.78` pts |
| Macro F1 | 0.4341 | 0.4992 | `+0.0651` |
| Weighted F1 | 0.6982 | 0.6890 | `-0.0092` |

## Runtime and Stability Notes

- preprocessing remained fast:
  - about `3.9s`
- training phase became much heavier because of SMOTE:
  - about `1866.68s`
- final calibration phase added significant cost:
  - about `851.19s`
- SHAP remained bounded and completed successfully:
  - about `21.89s`

Warnings observed but non-fatal:

- Logistic Regression convergence warnings
- XGBoost early-stopping deprecation warnings
- seaborn palette future warnings
- joblib core-count warning on Windows

## Final Conclusion

The Phase 2 upgrade was implemented successfully and kept the pipeline stable.

Key outcomes:

- pipeline structure and public interfaces were preserved
- multi-model offline evaluation still works
- production inference now uses a calibrated final model
- Macro F1 improved materially
- the selected best offline model remains `LGBM`
- the deployed production artifact is a calibrated `LGBM`

Current reliable Phase 2 production baseline:

- final model: `LGBM` calibrated with isotonic calibration
- accuracy: `73.49%`
- macro F1: `0.4992`
- weighted F1: `0.6890`
