# Final Delivery Report

## Pipeline Status

- Status: `success`
- Rebuild command: `.\.venv-py311\Scripts\python.exe main.py --mode train_all --retrain-final-model`
- Rebuild completed from a cleaned generated-artifact state
- Final deployable model: calibrated and threshold-tuned `LGBM`
- Target check: the production pipeline improved materially, but it did **not** reach the `0.60+` macro F1 target

## GUI Status

- Status: `working`
- GUI entrypoint: `dashboard.py`
- API entrypoint: `app.py`
- Browser automation validation: `passed`
- Verified flows:
  - dashboard startup without errors
  - overview page loads current metrics
  - manual single-record prediction renders class and probabilities
  - CSV upload containing `id` predicts successfully without schema errors
  - explain page loads saved SHAP visuals

## Metrics Summary

### Offline comparison winner

- Model: `LGBM`
- Accuracy: `75.34%`
- Macro F1: `0.5346`
- Weighted F1: `0.7244`

### Deployed production model

- Model: calibrated + thresholded `LGBM`
- Accuracy: `74.73%`
- Macro F1: `0.5269`
- Weighted F1: `0.7125`

### Minority-class note

- `fuzzers` recall: `0.1557`
- `fuzzers` F1: `0.2552`

## Validation Highlights

- `id` present in production input row: prediction still succeeds
- `id` in saved preprocessor schema: `False`
- Selected feature count: `75`
- Interaction features present:
  - `sload_dload_ratio`
  - `src_dst_ltm_ratio`
  - `ttl_diff`
- Controlled SMOTE confirmed on train-only paths
- Thresholds persisted:
  - `fuzzers = 0.30`
  - `backdoor = 0.15`
  - `analysis = 0.20`
  - `worms = 0.15`

## Generated Files

### Models

- `models/final_model.pkl`
- `models/model_lr.pkl`
- `models/model_rf.pkl`
- `models/model_xgb.pkl`
- `models/model_lgbm.pkl`
- `models/preprocessor.pkl`
- `models/selector.pkl`
- `models/scaler.pkl`
- `models/feature_info.pkl`
- `models/label_encoder.pkl`

### Reports

- `artifacts/reports/model_metrics.json`
- `artifacts/reports/final_summary.json`
- `artifacts/reports/feature_report.json`
- `artifacts/reports/classification_reports.md`
- `artifacts/reports/gui_validation.json`
- `artifacts/reports/final_delivery_report.md`

### Plots

- `artifacts/plots/label_distribution.png`
- `artifacts/plots/confusion_matrix_lgbm.png`
- `artifacts/plots/model_metric_comparison.png`
- `artifacts/plots/per_class_f1_heatmap.png`
- `artifacts/plots/roc_curve_comparison.png`
- `artifacts/plots/training_time_comparison.png`
- `artifacts/plots/shap_bar.png`
- `artifacts/plots/performance_evolution.png`

### GUI Screenshots

- `artifacts/screenshots/dashboard_overview.png`
- `artifacts/screenshots/dashboard_predict_form.png`
- `artifacts/screenshots/dashboard_prediction_result.png`

## PPT File

- `artifacts/reports/ids_final_delivery_presentation.pptx`
- Slide count: `14`
- Embedded content includes:
  - regenerated metrics
  - rebuilt plots
  - GUI screenshots
  - system evolution story across Phases 1, 2, and 3
