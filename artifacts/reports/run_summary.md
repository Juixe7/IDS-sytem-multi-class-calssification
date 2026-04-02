# Run Summary

Initialized: 2026-04-01 23:16:44

## Configuration

| Key | Value |
| --- | --- |
| Project root | `C:\Users\ramil\OneDrive\Desktop\New folder` |
| Train CSV | `C:\Users\ramil\OneDrive\Desktop\New folder\unsw\UNSW_NB15_training-set.csv` |
| Test CSV | `C:\Users\ramil\OneDrive\Desktop\New folder\unsw\UNSW_NB15_testing-set.csv` |
| Target column | `attack_cat` |
| Random state | `42` |
| Mode | `train_all` |
| Default production model | `lgbm` |
| Feature cap | `25` |
| CV folds | `3` |
| SHAP sample size | `300` |

## Phase 1

Local workspace folders were created, environment metadata was captured, and the run configuration was initialized.

## Phase 2

Loaded 82332 training rows and 175341 testing rows with a schema match of True.

## Phase 3 and 5

Removed 6 correlated numeric features and retained 25 transformed features for modeling.

## Phase 4

EDA visuals were generated for class balance, numeric spread, outliers, correlation, and categorical frequencies.

## Phase 6 and 7

train_all mode trained the full comparison set and preserved the offline evaluation path.

## Phase 8

train_all mode completed offline evaluation and selected LGBM as the production final_model.

## Phase 8B

The external repo benchmark was reproduced and audited; its 97% result depends heavily on target-derived attack_cat dummy columns that are not valid production inputs.

## Phase 9

SHAP explainability was generated for the production final_model LGBM using a bounded stratified sample.

## Phase 10

Final delivery packaging completed successfully. The project was rebuilt from a cleaned generated-artifact state, the dashboard GUI passed automated validation, screenshots were captured reproducibly, and the final presentation was generated at `artifacts/reports/ids_final_delivery_presentation.pptx`.

## Phase 10

The Phase 2 performance upgrade was logged in a dedicated report. It raised feature capacity to 50, added training-only SMOTE and XGBoost sample weighting, calibrated only the deployed final_model, and established a new calibrated LGBM production baseline with 73.49% accuracy and 0.4992 macro F1.

## Phase 11

The Phase 3 class-separability upgrade was logged in a dedicated report. It raised feature capacity to 75, added three interaction features, reduced SMOTE aggressiveness to a controlled 30%-of-majority target, tuned minority-class thresholds for the production final_model, and established a new thresholded calibrated LGBM production baseline with 74.73% accuracy and 0.5269 macro F1.

## Phase 1

Local workspace folders were created, environment metadata was captured, and the run configuration was initialized.

## Phase 2

Loaded 82332 training rows and 175341 testing rows with a schema match of True.

## Phase 3 and 5

Removed 6 correlated numeric features and retained 50 transformed features for modeling.

## Phase 4

EDA visuals were generated for class balance, numeric spread, outliers, correlation, and categorical frequencies.

## Phase 6 and 7

train_all mode trained the full comparison set and preserved the offline evaluation path.

## Phase 8

train_all mode completed offline evaluation, selected LGBM as the production final_model, and saved a calibrated production artifact.

## Phase 8B

The external repo benchmark was reproduced and audited; its 97% result depends heavily on target-derived attack_cat dummy columns that are not valid production inputs.

## Phase 9

SHAP explainability was generated for the production final_model LGBM using a bounded stratified sample.

## Phase 1

Local workspace folders were created, environment metadata was captured, and the run configuration was initialized.

## Phase 2

Loaded 82332 training rows and 175341 testing rows with a schema match of True.

## Phase 3 and 5

Removed 7 correlated numeric features and retained 75 transformed features for modeling.

## Phase 4

EDA visuals were generated for class balance, numeric spread, outliers, correlation, and categorical frequencies.

## Phase 6 and 7

train_all mode trained the full comparison set and preserved the offline evaluation path.

## Phase 8

train_all mode completed offline evaluation, selected LGBM as the production final_model, and saved a calibrated production artifact.

## Phase 8B

The external repo benchmark was reproduced and audited; its 97% result depends heavily on target-derived attack_cat dummy columns that are not valid production inputs.

## Phase 9

SHAP explainability was generated for the production final_model LGBM using a bounded stratified sample.

## Phase 1

Local workspace folders were created, environment metadata was captured, and the run configuration was initialized.

## Phase 1

Local workspace folders were created, environment metadata was captured, and the run configuration was initialized.

## Phase 2

Loaded 82332 training rows and 175341 testing rows with a schema match of True.

## Phase 3 and 5

Removed 7 correlated numeric features and retained 75 transformed features for modeling.

## Phase 4

EDA visuals were generated for class balance, numeric spread, outliers, correlation, and categorical frequencies.

## Phase 6 and 7

train_all mode trained the full comparison set and preserved the offline evaluation path.

## Phase 8

train_all mode completed offline evaluation, selected LGBM as the production final_model, and saved a calibrated production artifact.

## Phase 8B

The external repo benchmark was reproduced and audited; its 97% result depends heavily on target-derived attack_cat dummy columns that are not valid production inputs.

## Phase 9

SHAP explainability was generated for the production final_model LGBM using a bounded stratified sample.
