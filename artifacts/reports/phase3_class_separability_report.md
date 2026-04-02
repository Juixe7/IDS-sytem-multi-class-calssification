# Phase 3 Class-Separability Report

## Summary

This report captures the Phase 3 upgrade focused on class separability, especially for minority classes such as `fuzzers`. The upgrade kept the existing architecture intact while adding interaction features, expanding the feature cap, reducing SMOTE aggressiveness, and tuning minority-class thresholds for the calibrated production model.

Execution command:

```powershell
.\.venv-py311\Scripts\python.exe main.py --mode train_all --retrain-final-model
```

## Code Changes

### Preprocessing

- Increased `feature_cap` from `50` to `75`
- Added interaction features before correlation pruning and encoding:
  - `sload_dload_ratio`
  - `src_dst_ltm_ratio`
  - `ttl_diff`
- Kept the existing selection safeguard:
  - `k = min(config.feature_cap, X_train_processed.shape[1])`

### Training

- Replaced fully balancing SMOTE with controlled multiclass upsampling equivalent to:
  - `sampling_strategy = 0.3`
- Applied SMOTE only to training subsets, never validation or test data
- Kept class weights for `lr`, `rf`, and `lgbm`
- Kept explicit XGBoost sample weighting

### Production final model

- Kept calibration only for the selected `final_model`
- Held out a validation split before SMOTE
- Tuned thresholds for:
  - `fuzzers`
  - `backdoor`
  - `analysis`
  - `worms`
- Stored tuned thresholds in:
  - `models/feature_info.pkl`
  - `artifacts/reports/final_summary.json`
- Updated inference to apply threshold overrides before argmax fallback

## Validation Logs

### Feature-space checks

- Interaction features added:
  - `sload_dload_ratio`
  - `src_dst_ltm_ratio`
  - `ttl_diff`
- Features before selection: `186`
- Features after selection: `75`
- `id` in raw features: `False`
- `id` in transformed features: `False`
- `id` in selected features: `False`

### Correlation pruning

- Dropped correlated features:
  - `sbytes`
  - `dbytes`
  - `sloss`
  - `dloss`
  - `ct_ftp_cmd`
  - `is_sm_ips_ports`
  - `sload_dload_ratio`

### Duplicate checks

- Train duplicates removed after `id` drop: `26,387`
- Test duplicates removed after `id` drop: `67,601`

### Controlled SMOTE checks

Example final production-model training path:

- Before SMOTE: `X=(50350, 75)`, `y=(50350,)`
- After SMOTE: `X=(113909, 75)`, `y=(113909,)`

Class distribution before SMOTE:

```text
{0: 401, 1: 312, 2: 1546, 3: 6848, 4: 4354, 5: 3291, 6: 30785, 7: 2433, 8: 340, 9: 40}
```

Class distribution after SMOTE:

```text
{0: 9236, 1: 9236, 2: 9236, 3: 9236, 4: 9236, 5: 9236, 6: 30785, 7: 9236, 8: 9236, 9: 9236}
```

This confirms minority classes were expanded to 30% of the majority class instead of being fully balanced.

### Threshold tuning checks

Validation macro F1 used for tuning:

- `0.5845`

Tuned thresholds:

```text
{'fuzzers': 0.3, 'backdoor': 0.15, 'analysis': 0.2, 'worms': 0.15}
```

Saved production model type:

- `CalibratedClassifierCV`

## Metrics After Phase 3

### Offline comparison metrics

| Model | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| `LR` | 68.60% | 0.4307 | 0.7087 |
| `RF` | 74.48% | 0.5032 | 0.7171 |
| `XGB` | 75.81% | 0.5307 | 0.7434 |
| `LGBM` | 75.34% | 0.5346 | 0.7244 |

Selected offline winner:

- `LGBM`

### Calibrated + thresholded production metrics

From the deployed production path:

- Accuracy: `74.73%`
- Macro F1: `0.5269`
- Weighted F1: `0.7125`

### Fuzzers production performance

- Precision: `0.7076`
- Recall: `0.1557`
- F1: `0.2552`

## Comparison vs Phase 2

### Offline winning model comparison

| Metric | Phase 2 | Phase 3 | Change |
| --- | ---: | ---: | ---: |
| Accuracy | 73.74% | 75.34% | `+1.60` pts |
| Macro F1 | 0.5248 | 0.5346 | `+0.0098` |
| Weighted F1 | 0.7010 | 0.7244 | `+0.0234` |

### Deployed production model comparison

| Metric | Phase 2 | Phase 3 | Change |
| --- | ---: | ---: | ---: |
| Accuracy | 73.49% | 74.73% | `+1.24` pts |
| Macro F1 | 0.4992 | 0.5269 | `+0.0277` |
| Weighted F1 | 0.6890 | 0.7125 | `+0.0235` |

### Fuzzers production comparison

| Metric | Phase 2 | Phase 3 | Change |
| --- | ---: | ---: | ---: |
| Recall | 0.0777 | 0.1557 | `+0.0780` |
| F1 | 0.1390 | 0.2552 | `+0.1162` |

## Final Conclusion

Phase 3 improved the production pipeline, especially for `fuzzers`, but it did **not** reach the target of `0.60+` Macro F1.

What improved:

- minority-class recovery improved, especially `fuzzers`
- production Macro F1 improved from `0.4992` to `0.5269`
- production accuracy improved from `73.49%` to `74.73%`
- controlled SMOTE reduced oversampling pressure while keeping gains
- threshold tuning helped the deployable model more than the base argmax path

Current reliable Phase 3 production baseline:

- final model: calibrated and thresholded `LGBM`
- accuracy: `74.73%`
- macro F1: `0.5269`
- weighted F1: `0.7125`

Remaining limitation:

- `fuzzers` is better than before but still materially under-recalled, which is the main reason Macro F1 remains below the requested `0.60+` target.

## Final Logged Note

Phase 3 improved the IDS pipeline in a measurable and stable way, but it did not achieve the requested `0.60+` Macro F1 target.

Latest logged production baseline:

- final model: thresholded and calibrated `LGBM`
- accuracy: `74.73%`
- macro F1: `0.5269`
- weighted F1: `0.7125`
- fuzzers recall: `0.1557`
- fuzzers F1: `0.2552`

Bottom line:

- pipeline integrity preserved
- minority-class handling improved
- fuzzers improved materially versus Phase 2
- remaining gap to `0.60+` Macro F1 is still significant
