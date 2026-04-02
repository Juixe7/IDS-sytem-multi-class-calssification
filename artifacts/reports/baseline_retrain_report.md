# Fresh Baseline Retrain Report

## Summary

This report captures the clean retraining and evaluation run performed after fixing three verified preprocessing defects:

- `id` column leakage removed
- feature selection restored to `config.feature_cap`
- duplicate removal corrected to run after `id` drop

The pipeline was retrained from a clean state using:

```powershell
.\.venv-py311\Scripts\python.exe main.py --mode train_all --retrain-final-model
```

## Clean State Reset

Before retraining, previous generated artifacts were removed so the pipeline rebuilt everything from scratch.

Deleted or regenerated from scratch:

- `models/*.pkl`
- `outputs/*`
- `artifacts/plots/*.png`
- `artifacts/reports/*.json`
- `artifacts/reports/*.md`
- `artifacts/logs/*.md`
- `unsw/train.parquet`
- `unsw/test.parquet`

Outcome:

- No previous model was reused
- No previous scaler or selector was reused
- Preprocessing executed fresh
- Evaluation artifacts were regenerated fresh

## Training Summary

Training completed successfully for all configured models:

| Model | Training Status | Notes |
| --- | --- | --- |
| `LR` | Success | Convergence warning observed, but training completed |
| `RF` | Success | No training failure |
| `XGB` | Success | Early-stopping deprecation warning observed, but training completed |
| `LGBM` | Success | No training failure |

Additional pipeline stages:

- Evaluation completed successfully
- SHAP explainability completed successfully
- Production final model saved successfully
- Inference smoke test completed successfully

Selected production final model after fresh evaluation:

- `LGBM`

## Overall Metrics

Fresh metrics from regenerated `model_metrics.json`:

| Model | Accuracy | Macro F1 | Weighted F1 | Avg CV Accuracy | Avg CV Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `LR` | 63.29% | 0.3699 | 0.6761 | 57.01% | 0.3365 |
| `RF` | 71.13% | 0.4147 | 0.6690 | 83.94% | 0.4426 |
| `XGB` | 70.41% | 0.3641 | 0.6424 | 84.94% | 0.4371 |
| `LGBM` | 71.71% | 0.4341 | 0.6982 | 82.69% | 0.4703 |

New reliable baseline under the pipeline ranking rule:

- Final model: `LGBM`
- Accuracy: `71.71%`
- Macro F1: `0.4341`
- Weighted F1: `0.6982`

## Per-Class Metrics for Final Baseline Model (`LGBM`)

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| `analysis` | 0.1591 | 0.3174 | 0.2120 | 1594 |
| `backdoor` | 0.1701 | 0.1134 | 0.1360 | 1535 |
| `dos` | 0.2278 | 0.3484 | 0.2755 | 3806 |
| `exploits` | 0.8100 | 0.7023 | 0.7523 | 19844 |
| `fuzzers` | 0.6712 | 0.2172 | 0.3281 | 16150 |
| `generic` | 0.8154 | 0.8144 | 0.8149 | 4181 |
| `normal` | 0.8055 | 0.9583 | 0.8753 | 51890 |
| `reconnaissance` | 0.5613 | 0.5686 | 0.5649 | 7522 |
| `shellcode` | 0.2477 | 0.3529 | 0.2911 | 1091 |
| `worms` | 0.0739 | 0.1181 | 0.0909 | 127 |

## Validation Logs

### Feature and Leakage Checks

- Features before selection: `184`
- Feature cap: `25`
- Features after selection: `25`
- Selected feature count: `25`
- `id` in raw feature columns: `False`
- `id` in transformed feature names: `False`
- `id` in selected feature names: `False`
- `id` in `preprocessor.feature_names_in_`: `False`

### Duplicate Removal

- Train duplicates removed after `id` drop: `26,387`
- Test duplicates removed after `id` drop: `67,601`

### Fresh Inference Compatibility

- Raw row containing `id` still predicts successfully: `True`
- Saved preprocessor input schema contains `id`: `False`

## Cross-Validation vs Test Consistency

| Model | Avg CV Accuracy | Test Accuracy | Gap | Avg CV Macro F1 | Test Macro F1 | Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `LR` | 57.01% | 63.29% | `+6.28` pts | 0.3365 | 0.3699 | `+0.0334` |
| `RF` | 83.94% | 71.13% | `-12.81` pts | 0.4426 | 0.4147 | `-0.0278` |
| `XGB` | 84.94% | 70.41% | `-14.53` pts | 0.4371 | 0.3641 | `-0.0730` |
| `LGBM` | 82.69% | 71.71% | `-10.98` pts | 0.4703 | 0.4341 | `-0.0362` |

Interpretation:

- `RF`, `XGB`, and `LGBM` still show train-to-test drop, so some overfitting remains.
- `LGBM` has the strongest final test macro F1 and the smallest tree-model CV-to-test accuracy gap among the best-performing models.
- `LR` performs better on test than CV, which suggests it is underfitting rather than overfitting.

## Before vs After Fixes

Comparison source for the "before" baseline:

- last verified pre-fix evaluation run from the earlier pipeline state before the clean reset
- previous selected best model: `RF`
- previous headline metrics: Accuracy `66.08%`, Macro F1 `0.3588`

Pipeline-selected best model before vs after:

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Accuracy | 66.08% | 71.71% | `+5.63` pts |
| Macro F1 | 0.3588 | 0.4341 | `+0.0753` |

Same-model `RF` comparison:

| Metric | RF Before | RF After | Change |
| --- | ---: | ---: | ---: |
| Accuracy | 66.08% | 71.13% | `+5.05` pts |
| Macro F1 | 0.3588 | 0.4147 | `+0.0559` |

## Duplicate Removal Impact on Class Distribution

Duplicate removal materially changed the effective class distribution because duplicates only became visible after `id` was removed.

### Train Split Examples

- `generic`: 22.92% -> 6.54% (`-16.38` percentage points)
- `normal`: 44.94% -> 61.14% (`+16.20` percentage points)
- `fuzzers`: 7.36% -> 8.65% (`+1.28` percentage points)

### Test Split Examples

- `generic`: 22.81% -> 3.88% (`-18.93` percentage points)
- `normal`: 31.94% -> 48.16% (`+16.22` percentage points)
- `fuzzers`: 10.37% -> 14.99% (`+4.62` percentage points)

## Root Cause Confirmation

### 1. Did removing `id` reduce CV accuracy but improve test accuracy?

Yes for the cleanest like-for-like comparison on `RF`.

- RF average CV accuracy before fixes: about `87.94%`
- RF average CV accuracy after fixes: `83.94%`
- RF test accuracy before fixes: `66.08%`
- RF test accuracy after fixes: `71.13%`

This supports the conclusion that the earlier pipeline was benefiting from leakage-driven overfitting.

### 2. Did feature selection improve generalization?

The corrected pipeline now enforces:

- transformed features before selection: `184`
- selected features after selection: `25`

Generalization improved overall, but the effect is combined with leakage removal and duplicate correction, so it cannot be isolated perfectly from this single retrain.

Evidence:

- RF CV-test accuracy gap before fixes: about `21.86` pts
- RF CV-test accuracy gap after fixes: `12.81` pts
- Best-model macro F1 improved from `0.3588` to `0.4341`

### 3. Did duplicate removal change class distribution?

Yes. It changed the class distribution substantially in both train and test because many duplicates were previously masked by unique `id` values.

Evidence:

- Train duplicates removed: `26,387`
- Test duplicates removed: `67,601`
- Major class balance shifts were observed, especially in `generic` and `normal`

## Final Conclusion

The corrected pipeline is now valid as a fresh, leakage-free, reproducible baseline.

New reliable baseline:

- Final model: `LGBM`
- Accuracy: `71.71%`
- Macro F1: `0.4341`
- Weighted F1: `0.6982`

Key conclusions:

- `id` leakage is removed end-to-end
- feature selection is restored and applied correctly
- duplicate removal is now meaningful and materially changes the dataset composition
- all artifacts were rebuilt from scratch
- training, evaluation, SHAP, and inference all still work after the fixes

Remaining note:

- `LR` still emits convergence warnings, but this does not invalidate the baseline because the selected final model is `LGBM`.
