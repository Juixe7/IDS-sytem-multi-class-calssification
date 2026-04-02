# Project Journal

Initialized: 2026-04-01 23:16:44

This journal is updated after each pipeline phase.

## Phase 1 - Setup and Run Governance (2026-04-01 23:16:44)

**Objective**: Initialize folders, configuration, and runtime metadata for the local pipeline.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder

**Methods applied**
- Created artifacts/models/outputs folders
- Initialized journal and run summary
- Captured environment metadata

**Metrics/plots generated**
- Environment summary table
- Package version report

**Decisions and justification**
- Using local Python files as the source of truth keeps the workflow reproducible without depending on notebook tooling.
- A fixed configuration object prevents drift in model and reporting settings.

**Challenges and resolution**
- None

**Observations and insights**
- Workspace scaffolding is ready for dataset-driven phases.

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\logs\project_journal.md
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\run_summary.md
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\environment_versions.json

**Next step**: Load and validate the predefined train/test datasets.
## Phase 2 - Data Ingestion and Profiling (2026-04-01 23:16:49)

**Objective**: Load the predefined UNSW-NB15 train/test split, validate schema integrity, and profile the target distribution.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\unsw\UNSW_NB15_training-set.csv
- C:\Users\ramil\OneDrive\Desktop\New folder\unsw\UNSW_NB15_testing-set.csv

**Methods applied**
- Loaded CSV files with pandas
- Validated schema and target presence
- Attempted parquet caching for repeated local reads

**Metrics/plots generated**
- Train/test label distribution plot
- Missing values summary plot
- Schema profile JSON

**Decisions and justification**
- The predefined train/test split is preserved exactly to avoid leakage and keep the project bounded.
- Parquet caching is enabled when available because it reduces repeated local read time without changing semantics.

**Challenges and resolution**
- None

**Observations and insights**
- Schema match: True
- Target present in both splits: True
- Binary label retained for reference but excluded from multiclass target modeling: True

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\data_profile.json
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\label_distribution.png
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\missing_values_train.png

**Next step**: Clean the data and prepare train-only feature engineering.
## Phase 3 and 5 - Cleaning, Preprocessing, and Feature Reduction (2026-04-01 23:16:54)

**Objective**: Clean the train/test splits, encode the multiclass target, reduce redundant features, and save preprocessing artifacts.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\unsw\UNSW_NB15_training-set.csv
- C:\Users\ramil\OneDrive\Desktop\New folder\unsw\UNSW_NB15_testing-set.csv

**Methods applied**
- Replaced infinite values with NaN
- Dropped exact duplicate rows
- Applied median and most-frequent imputations through a train-fitted preprocessor
- Removed highly correlated numeric features above the fixed threshold
- Encoded the target with LabelEncoder and selected the top 25 transformed features
- Scaled the selected feature matrix for Logistic Regression

**Metrics/plots generated**
- Missingness plots before and after preprocessing
- Feature report with selected features, dropped correlations, and class mapping

**Decisions and justification**
- Target modeling uses attack_cat because it aligns with the multiclass IDS objective; the binary label column is excluded from features to prevent leakage.
- Feature selection is capped at 25 to reduce compute while preserving a manageable representation for all four models.
- All fitting decisions were learned from training data only, then applied to test data unchanged.

**Challenges and resolution**
- None

**Observations and insights**
- Dropped correlated features: 6
- Selected transformed features: 25
- Classes modeled: analysis, backdoor, dos, exploits, fuzzers, generic, normal, reconnaissance, shellcode, worms

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\models\label_encoder.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\preprocessor.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\selector.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\scaler.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\feature_info.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\feature_report.json

**Next step**: Generate compact EDA visuals on the cleaned and bounded feature space.
## Phase 4 - Decision-Focused EDA (2026-04-01 23:17:06)

**Objective**: Produce only the exploratory visuals needed to justify preprocessing and model choices.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\unsw\UNSW_NB15_training-set.csv

**Methods applied**
- Generated numeric distribution, boxplot, correlation, and categorical frequency visuals

**Metrics/plots generated**
- Numeric distribution panel
- Top variance boxplots
- Correlation heatmap
- Categorical frequency plot
- EDA summary Markdown

**Decisions and justification**
- EDA was intentionally constrained to decision-relevant analysis so the project stays finishable.
- Correlation and feature spread findings feed directly into the feature reduction strategy.

**Challenges and resolution**
- None

**Observations and insights**
- The training split contains both numeric and categorical predictors, so a mixed preprocessing pipeline is required.
- Outlier presence supports using tree models alongside a scaled linear baseline.

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\eda_summary.md
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\eda_numeric_distributions.png
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\eda_boxplots.png
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\eda_correlation_heatmap.png

**Next step**: Train the four bounded-complexity models and compare their cross-validation behavior.
## Phase 6 and 7 - Evaluation Setup and Model Training (2026-04-01 23:20:31)

**Objective**: Train all configured models for offline comparison and persist them for evaluation.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\models\scaler.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\feature_info.pkl

**Methods applied**
- Used 3-fold StratifiedKFold on training data only
- Trained Logistic Regression on scaled selected features
- Trained Random Forest, XGBoost, and LightGBM on the selected feature matrix
- Applied internal validation splits for XGBoost and LightGBM final fitting

**Metrics/plots generated**
- Cross-validation score summary
- Training time comparison data

**Decisions and justification**
- train_all mode preserves the experimental comparison pipeline for academic reporting.
- The offline comparison path remains separate from the production inference artifact.

**Challenges and resolution**
- None

**Observations and insights**
- All requested comparison models were trained and serialized.
- Training times and CV behavior are available for side-by-side comparison.

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_lr.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_rf.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_xgb.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_lgbm.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\training_cv_summary.json

**Next step**: Evaluate every trained model on the untouched predefined test set.
## Phase 8 - Final Evaluation and Comparison (2026-04-01 23:21:57)

**Objective**: Compare all trained models on the untouched test split and choose the production final_model.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_lr.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_rf.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_xgb.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_lgbm.pkl

**Methods applied**
- Generated predictions and probabilities on the untouched test set
- Computed accuracy, macro F1, weighted F1, per-class metrics, confusion matrices, ROC-AUC, and false alarm rates
- Ranked models by the existing macro-F1-first selection rule
- Saved the winning estimator again as models/final_model.pkl for production inference

**Metrics/plots generated**
- Per-model confusion matrices
- ROC comparison plot
- Metric comparison chart
- Training time comparison chart
- Per-class F1 heatmap

**Decisions and justification**
- The production final_model is selected by measured evaluation metrics; current winner: LGBM.
- Macro F1 remains the leading ranking signal because multiclass robustness matters more than raw accuracy alone.

**Challenges and resolution**
- None

**Observations and insights**
- Production final_model selected: LGBM
- Offline comparison artifacts remain intact for academic reporting.

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\model_metrics.json
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\classification_reports.md
- C:\Users\ramil\OneDrive\Desktop\New folder\models\final_model.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\model_metric_comparison.png
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\per_class_f1_heatmap.png

**Next step**: Run bounded SHAP explainability on the selected final_model.
## Phase 8B - External Repo Benchmark Audit (2026-04-01 23:22:12)

**Objective**: Reproduce the public repo benchmark and measure how much of its score depends on target-derived leakage features.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\.codex-tmp\IoT-Network-Intrusion-Detection-System-UNSW-NB15\datasets\multi_data.csv

**Methods applied**
- Loaded the repo-prepared multi_data.csv file
- Reproduced the published RandomForest benchmark with the repo split
- Reran the same benchmark after removing attack_cat_* leakage columns from the feature matrix

**Metrics/plots generated**
- Repo benchmark leakage audit Markdown report
- Accuracy and macro-F1 comparison plot

**Decisions and justification**
- The external repo benchmark is documented for comparison, but it is not used as the production model selection source.
- Production inference remains limited to raw non-leaky features that are actually available at prediction time.

**Challenges and resolution**
- None

**Observations and insights**
- Repo-style benchmark accuracy with leakage: 97.27%
- Repo-style benchmark accuracy without attack_cat leakage columns: 89.21%

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\repo_benchmark_audit.json
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\repo_benchmark_audit.md
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\repo_benchmark_accuracy_comparison.png

**Next step**: Generate SHAP for the final production model and keep the SaaS inference path honest.
## Phase 9 - SHAP Explainability (2026-04-01 23:22:25)

**Objective**: Generate bounded-cost explainability artifacts for the production final_model.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\models\final_model.pkl

**Methods applied**
- Sampled up to 300 training rows in a class-aware way
- Computed SHAP values for the production final_model
- Generated summary, bar, and waterfall explainability plots

**Metrics/plots generated**
- SHAP summary plot
- SHAP feature-importance bar plot
- SHAP waterfall example

**Decisions and justification**
- SHAP is generated only for the production final_model so inference explainability stays single-model and predictable.
- The bounded sample keeps explainability practical for SaaS-oriented production runs.

**Challenges and resolution**
- None

**Observations and insights**
- The explainability package complements the evaluation metrics with feature-level reasoning.
- Interpretation should still be treated as sample-based evidence rather than a full-population guarantee.

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\outputs\shap_values.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\shap_summary.md
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\shap_summary.png
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\shap_bar.png
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\shap_waterfall_example.png

**Next step**: Review saved artifacts and confirm the local-only pipeline is reproducible.
## Phase 10 - Final Delivery Packaging (2026-04-02 16:30:00)

**Objective**: Rebuild the pipeline cleanly, validate the GUI end to end, refresh delivery artifacts, and package the final presentation.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\main.py
- C:\Users\ramil\OneDrive\Desktop\New folder\dashboard.py
- C:\Users\ramil\OneDrive\Desktop\New folder\app.py
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\model_metrics.json
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\final_summary.json

**Methods applied**
- Removed generated models, plots, selected reports, outputs, and parquet caches before retraining
- Re-ran `main.py --mode train_all --retrain-final-model`
- Validated saved artifact state for model type, thresholds, selected features, and `id` removal
- Launched the API and Streamlit dashboard for browser-automated GUI testing
- Captured dashboard screenshots and generated the final PowerPoint presentation

**Metrics/plots generated**
- Refreshed full evaluation metric set
- Refreshed confusion matrices, ROC curve, heatmap, and SHAP visuals
- Added `performance_evolution.png`
- Added GUI screenshots for overview, prediction form, and prediction result

**Decisions and justification**
- Kept the deployable model as the calibrated + thresholded `LGBM` because it remains the best production artifact under the current pipeline rules.
- Used browser automation instead of manual screenshots so GUI evidence is reproducible.

**Challenges and resolution**
- Playwright needed to run outside the sandbox on Windows because its subprocess creation was permission-blocked.
- Standalone Matplotlib plot generation needed `MPLCONFIGDIR` pinned to the local workspace cache.

**Observations and insights**
- Offline best-model metrics: accuracy `75.34%`, macro F1 `0.5346`, weighted F1 `0.7244`
- Production metrics: accuracy `74.73%`, macro F1 `0.5269`, weighted F1 `0.7125`
- GUI validation passed, including CSV upload containing `id`
- The production pipeline improved materially, but the `0.60+` macro F1 target was still not reached

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\final_delivery_report.md
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\gui_validation.json
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\ids_final_delivery_presentation.pptx
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\performance_evolution.png
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\screenshots\dashboard_overview.png
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\screenshots\dashboard_predict_form.png
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\screenshots\dashboard_prediction_result.png

**Next step**: Deliver the refreshed artifacts and presentation for evaluation.
## Phase 10 - Phase 2 Performance Upgrade (2026-04-02 14:06:00)

**Objective**: Apply targeted model-performance upgrades without changing pipeline structure, then record the resulting baseline and validation evidence.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\src\config.py
- C:\Users\ramil\OneDrive\Desktop\New folder\src\preprocess.py
- C:\Users\ramil\OneDrive\Desktop\New folder\src\train.py
- C:\Users\ramil\OneDrive\Desktop\New folder\src\evaluate.py
- C:\Users\ramil\OneDrive\Desktop\New folder\src\inference.py
- C:\Users\ramil\OneDrive\Desktop\New folder\src\explain.py
- C:\Users\ramil\OneDrive\Desktop\New folder\main.py

**Methods applied**
- Increased feature selection cap from 25 to 50
- Added SMOTE to training subsets only
- Kept class weighting for Logistic Regression, Random Forest, and LightGBM
- Added explicit XGBoost sample weights during fit
- Calibrated only the selected production final_model with isotonic calibration
- Re-ran the full pipeline in train_all mode and regenerated metrics, reports, and production artifacts

**Metrics/plots generated**
- Updated model_metrics.json
- Updated final_summary.json with calibrated final-model metrics
- Updated feature_report.json with 50-feature selection evidence
- Updated SHAP artifacts for the calibrated production model
- Consolidated Phase 2 upgrade report

**Decisions and justification**
- Feature capacity was increased to preserve more minority-class signal while staying within a bounded classical-ML setup.
- SMOTE was restricted to training subsets only to avoid validation and test contamination.
- Calibration was applied only to the deployed final_model so offline comparison reporting stayed structurally unchanged.
- Threshold tuning was intentionally skipped to keep the upgrade controlled and backward compatible.

**Challenges and resolution**
- Training time increased substantially because SMOTE expanded every training subset.
- The calibrated production artifact required safe SHAP unwrapping, which was handled inside the existing explainability and inference modules without changing external APIs.
- Logistic Regression still emitted convergence warnings, but the run completed and did not block the final model path.

**Observations and insights**
- Features before selection: 184
- Features after selection: 50
- Train duplicates removed after id drop: 26387
- Test duplicates removed after id drop: 67601
- Offline best model after upgrade: LGBM
- Offline LGBM macro F1: 0.5248
- Calibrated production macro F1: 0.4992
- Macro F1 improved materially versus the prior reliable baseline of 0.4341

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\phase2_performance_upgrade_report.md
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\model_metrics.json
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\final_summary.json
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\feature_report.json
- C:\Users\ramil\OneDrive\Desktop\New folder\models\final_model.pkl

**Next step**: Use the logged Phase 2 baseline for dashboard presentation, SaaS demo validation, and any future controlled optimization work.
## Phase 11 - Phase 3 Class-Separability Upgrade (2026-04-02 15:10:00)

**Objective**: Improve class separability and Macro F1 with targeted interaction features, controlled SMOTE, and threshold-tuned production inference while preserving the existing pipeline structure.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\src\config.py
- C:\Users\ramil\OneDrive\Desktop\New folder\src\preprocess.py
- C:\Users\ramil\OneDrive\Desktop\New folder\src\train.py
- C:\Users\ramil\OneDrive\Desktop\New folder\src\evaluate.py
- C:\Users\ramil\OneDrive\Desktop\New folder\src\inference.py
- C:\Users\ramil\OneDrive\Desktop\New folder\main.py

**Methods applied**
- Increased feature cap from 50 to 75
- Added three interaction features before correlation pruning and encoding
- Replaced fully balancing SMOTE with controlled multiclass upsampling to 30% of the majority class
- Tuned thresholds for fuzzers, backdoor, analysis, and worms on a held-out validation split for the calibrated final_model
- Re-ran the full train_all pipeline and regenerated production artifacts

**Metrics/plots generated**
- Updated model_metrics.json
- Updated final_summary.json with thresholded production metrics and tuned thresholds
- Updated feature_report.json with interaction-feature and 75-feature selection evidence
- Updated SHAP artifacts for the calibrated production model
- Consolidated Phase 3 class-separability report

**Decisions and justification**
- Interaction features were added to improve separability between overlapping traffic patterns without redesigning the feature pipeline.
- Controlled SMOTE was used instead of full balancing to reduce overfitting pressure on extremely rare classes.
- Threshold tuning was limited to the deployed final_model so offline comparison artifacts remained structurally unchanged.
- Macro F1 remained the tuning objective, with fuzzers recovery treated as the main practical target inside that objective.

**Challenges and resolution**
- `sampling_strategy=0.3` is not directly valid for multiclass SMOTE, so an equivalent multiclass target dictionary was computed to achieve the requested behavior safely.
- The production inference path needed threshold-aware prediction without changing the external API, so thresholds were persisted inside the existing feature_info artifact.
- The target of 0.60+ Macro F1 was not reached even after the upgrade; this was logged explicitly rather than hidden.

**Observations and insights**
- Interaction features added: sload_dload_ratio, src_dst_ltm_ratio, ttl_diff
- Features before selection: 186
- Features after selection: 75
- Offline best model after Phase 3: LGBM
- Offline LGBM macro F1: 0.5346
- Calibrated thresholded production macro F1: 0.5269
- Fuzzers production recall improved from 0.0777 to 0.1557
- Fuzzers production F1 improved from 0.1390 to 0.2552

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\phase3_class_separability_report.md
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\model_metrics.json
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\final_summary.json
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\feature_report.json
- C:\Users\ramil\OneDrive\Desktop\New folder\models\final_model.pkl

**Next step**: Use the Phase 3 report to decide whether the next iteration should target fuzzers-specific feature design or a different deployment tradeoff, since the 0.60+ Macro F1 target remains unmet.
## Phase 11A - Final Phase 3 Outcome Note (2026-04-02 15:18:00)

**Objective**: Record the final Phase 3 outcome summary in plain text for quick reference.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\phase3_class_separability_report.md
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\final_summary.json

**Methods applied**
- Added a concise closing summary of the final Phase 3 production baseline
- Explicitly recorded that the 0.60+ macro-F1 target was not reached

**Metrics/plots generated**
- No new plots
- Final logged metrics note

**Decisions and justification**
- The final outcome note was added so the key result can be found quickly without scanning the full report tables.

**Challenges and resolution**
- None

**Observations and insights**
- Final production model: thresholded and calibrated LGBM
- Production accuracy: 74.73%
- Production macro F1: 0.5269
- Fuzzers recall: 0.1557
- Fuzzers F1: 0.2552
- The target of 0.60+ macro F1 remains unmet

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\phase3_class_separability_report.md
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\logs\project_journal.md

**Next step**: Use the logged baseline to guide the next controlled optimization cycle.
## Phase 1 - Setup and Run Governance (2026-04-02 11:24:33)

**Objective**: Initialize folders, configuration, and runtime metadata for the local pipeline.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder

**Methods applied**
- Created artifacts/models/outputs folders
- Initialized journal and run summary
- Captured environment metadata

**Metrics/plots generated**
- Environment summary table
- Package version report

**Decisions and justification**
- Using local Python files as the source of truth keeps the workflow reproducible without depending on notebook tooling.
- A fixed configuration object prevents drift in model and reporting settings.

**Challenges and resolution**
- None

**Observations and insights**
- Workspace scaffolding is ready for dataset-driven phases.

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\logs\project_journal.md
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\run_summary.md
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\environment_versions.json

**Next step**: Load and validate the predefined train/test datasets.
## Phase 2 - Data Ingestion and Profiling (2026-04-02 11:24:40)

**Objective**: Load the predefined UNSW-NB15 train/test split, validate schema integrity, and profile the target distribution.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\unsw\UNSW_NB15_training-set.csv
- C:\Users\ramil\OneDrive\Desktop\New folder\unsw\UNSW_NB15_testing-set.csv

**Methods applied**
- Loaded CSV files with pandas
- Validated schema and target presence
- Attempted parquet caching for repeated local reads

**Metrics/plots generated**
- Train/test label distribution plot
- Missing values summary plot
- Schema profile JSON

**Decisions and justification**
- The predefined train/test split is preserved exactly to avoid leakage and keep the project bounded.
- Parquet caching is enabled when available because it reduces repeated local read time without changing semantics.

**Challenges and resolution**
- None

**Observations and insights**
- Schema match: True
- Target present in both splits: True
- Binary label retained for reference but excluded from multiclass target modeling: True

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\data_profile.json
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\label_distribution.png
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\missing_values_train.png

**Next step**: Clean the data and prepare train-only feature engineering.
## Phase 3 and 5 - Cleaning, Preprocessing, and Feature Reduction (2026-04-02 11:24:46)

**Objective**: Clean the train/test splits, encode the multiclass target, reduce redundant features, and save preprocessing artifacts.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\unsw\UNSW_NB15_training-set.csv
- C:\Users\ramil\OneDrive\Desktop\New folder\unsw\UNSW_NB15_testing-set.csv

**Methods applied**
- Replaced infinite values with NaN
- Dropped exact duplicate rows
- Applied median and most-frequent imputations through a train-fitted preprocessor
- Removed highly correlated numeric features above the fixed threshold
- Encoded the target with LabelEncoder and selected the top 50 transformed features
- Scaled the selected feature matrix for Logistic Regression

**Metrics/plots generated**
- Missingness plots before and after preprocessing
- Feature report with selected features, dropped correlations, and class mapping

**Decisions and justification**
- Target modeling uses attack_cat because it aligns with the multiclass IDS objective; the binary label column is excluded from features to prevent leakage.
- Feature selection is capped at 50 to preserve more minority-class signal while keeping compute bounded.
- All fitting decisions were learned from training data only, then applied to test data unchanged.

**Challenges and resolution**
- None

**Observations and insights**
- Dropped correlated features: 6
- Selected transformed features: 50
- Classes modeled: analysis, backdoor, dos, exploits, fuzzers, generic, normal, reconnaissance, shellcode, worms

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\models\label_encoder.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\preprocessor.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\selector.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\scaler.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\feature_info.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\feature_report.json

**Next step**: Generate compact EDA visuals on the cleaned and bounded feature space.
## Phase 4 - Decision-Focused EDA (2026-04-02 11:25:01)

**Objective**: Produce only the exploratory visuals needed to justify preprocessing and model choices.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\unsw\UNSW_NB15_training-set.csv

**Methods applied**
- Generated numeric distribution, boxplot, correlation, and categorical frequency visuals

**Metrics/plots generated**
- Numeric distribution panel
- Top variance boxplots
- Correlation heatmap
- Categorical frequency plot
- EDA summary Markdown

**Decisions and justification**
- EDA was intentionally constrained to decision-relevant analysis so the project stays finishable.
- Correlation and feature spread findings feed directly into the feature reduction strategy.

**Challenges and resolution**
- None

**Observations and insights**
- The training split contains both numeric and categorical predictors, so a mixed preprocessing pipeline is required.
- Outlier presence supports using tree models alongside a scaled linear baseline.

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\eda_summary.md
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\eda_numeric_distributions.png
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\eda_boxplots.png
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\eda_correlation_heatmap.png

**Next step**: Train the four bounded-complexity models and compare their cross-validation behavior.
## Phase 6 and 7 - Evaluation Setup and Model Training (2026-04-02 11:56:07)

**Objective**: Train all configured models for offline comparison and persist them for evaluation.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\models\scaler.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\feature_info.pkl

**Methods applied**
- Used 3-fold StratifiedKFold on training data only
- Trained Logistic Regression on scaled selected features
- Trained Random Forest, XGBoost, and LightGBM on the selected feature matrix
- Applied internal validation splits for XGBoost and LightGBM final fitting

**Metrics/plots generated**
- Cross-validation score summary
- Training time comparison data

**Decisions and justification**
- train_all mode preserves the experimental comparison pipeline for academic reporting.
- The offline comparison path remains separate from the production inference artifact.

**Challenges and resolution**
- None

**Observations and insights**
- All requested comparison models were trained and serialized.
- Training times and CV behavior are available for side-by-side comparison.

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_lr.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_rf.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_xgb.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_lgbm.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\training_cv_summary.json

**Next step**: Evaluate every trained model on the untouched predefined test set.
## Phase 8 - Final Evaluation and Comparison (2026-04-02 12:16:19)

**Objective**: Compare all trained models on the untouched test split and choose the production final_model.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_lr.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_rf.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_xgb.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_lgbm.pkl

**Methods applied**
- Generated predictions and probabilities on the untouched test set
- Computed accuracy, macro F1, weighted F1, per-class metrics, confusion matrices, ROC-AUC, and false alarm rates
- Ranked models by the existing macro-F1-first selection rule
- Calibrated only the winning model for production inference and saved it as models/final_model.pkl

**Metrics/plots generated**
- Per-model confusion matrices
- ROC comparison plot
- Metric comparison chart
- Training time comparison chart
- Per-class F1 heatmap

**Decisions and justification**
- The production final_model is selected by measured evaluation metrics; current winner: LGBM.
- Macro F1 remains the leading ranking signal because multiclass robustness matters more than raw accuracy alone.

**Challenges and resolution**
- None

**Observations and insights**
- Production final_model selected: LGBM
- Calibrated final_model accuracy: 0.7349
- Calibrated final_model macro F1: 0.4992
- Offline comparison artifacts remain intact for academic reporting.

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\model_metrics.json
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\classification_reports.md
- C:\Users\ramil\OneDrive\Desktop\New folder\models\final_model.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\model_metric_comparison.png
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\per_class_f1_heatmap.png

**Next step**: Run bounded SHAP explainability on the selected final_model.
## Phase 8B - External Repo Benchmark Audit (2026-04-02 12:16:34)

**Objective**: Reproduce the public repo benchmark and measure how much of its score depends on target-derived leakage features.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\.codex-tmp\IoT-Network-Intrusion-Detection-System-UNSW-NB15\datasets\multi_data.csv

**Methods applied**
- Loaded the repo-prepared multi_data.csv file
- Reproduced the published RandomForest benchmark with the repo split
- Reran the same benchmark after removing attack_cat_* leakage columns from the feature matrix

**Metrics/plots generated**
- Repo benchmark leakage audit Markdown report
- Accuracy and macro-F1 comparison plot

**Decisions and justification**
- The external repo benchmark is documented for comparison, but it is not used as the production model selection source.
- Production inference remains limited to raw non-leaky features that are actually available at prediction time.

**Challenges and resolution**
- None

**Observations and insights**
- Repo-style benchmark accuracy with leakage: 97.27%
- Repo-style benchmark accuracy without attack_cat leakage columns: 89.21%

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\repo_benchmark_audit.json
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\repo_benchmark_audit.md
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\repo_benchmark_accuracy_comparison.png

**Next step**: Generate SHAP for the final production model and keep the SaaS inference path honest.
## Phase 9 - SHAP Explainability (2026-04-02 12:16:56)

**Objective**: Generate bounded-cost explainability artifacts for the production final_model.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\models\final_model.pkl

**Methods applied**
- Sampled up to 300 training rows in a class-aware way
- Computed SHAP values for the production final_model
- Generated summary, bar, and waterfall explainability plots

**Metrics/plots generated**
- SHAP summary plot
- SHAP feature-importance bar plot
- SHAP waterfall example

**Decisions and justification**
- SHAP is generated only for the production final_model so inference explainability stays single-model and predictable.
- The bounded sample keeps explainability practical for SaaS-oriented production runs.

**Challenges and resolution**
- None

**Observations and insights**
- The explainability package complements the evaluation metrics with feature-level reasoning.
- Interpretation should still be treated as sample-based evidence rather than a full-population guarantee.

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\outputs\shap_values.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\shap_summary.md
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\shap_summary.png
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\shap_bar.png
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\shap_waterfall_example.png

**Next step**: Review saved artifacts and confirm the local-only pipeline is reproducible.
## Phase 1 - Setup and Run Governance (2026-04-02 13:25:54)

**Objective**: Initialize folders, configuration, and runtime metadata for the local pipeline.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder

**Methods applied**
- Created artifacts/models/outputs folders
- Initialized journal and run summary
- Captured environment metadata

**Metrics/plots generated**
- Environment summary table
- Package version report

**Decisions and justification**
- Using local Python files as the source of truth keeps the workflow reproducible without depending on notebook tooling.
- A fixed configuration object prevents drift in model and reporting settings.

**Challenges and resolution**
- None

**Observations and insights**
- Workspace scaffolding is ready for dataset-driven phases.

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\logs\project_journal.md
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\run_summary.md
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\environment_versions.json

**Next step**: Load and validate the predefined train/test datasets.
## Phase 2 - Data Ingestion and Profiling (2026-04-02 13:26:00)

**Objective**: Load the predefined UNSW-NB15 train/test split, validate schema integrity, and profile the target distribution.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\unsw\UNSW_NB15_training-set.csv
- C:\Users\ramil\OneDrive\Desktop\New folder\unsw\UNSW_NB15_testing-set.csv

**Methods applied**
- Loaded CSV files with pandas
- Validated schema and target presence
- Attempted parquet caching for repeated local reads

**Metrics/plots generated**
- Train/test label distribution plot
- Missing values summary plot
- Schema profile JSON

**Decisions and justification**
- The predefined train/test split is preserved exactly to avoid leakage and keep the project bounded.
- Parquet caching is enabled when available because it reduces repeated local read time without changing semantics.

**Challenges and resolution**
- None

**Observations and insights**
- Schema match: True
- Target present in both splits: True
- Binary label retained for reference but excluded from multiclass target modeling: True

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\data_profile.json
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\label_distribution.png
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\missing_values_train.png

**Next step**: Clean the data and prepare train-only feature engineering.
## Phase 3 and 5 - Cleaning, Preprocessing, and Feature Reduction (2026-04-02 13:26:07)

**Objective**: Clean the train/test splits, encode the multiclass target, reduce redundant features, and save preprocessing artifacts.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\unsw\UNSW_NB15_training-set.csv
- C:\Users\ramil\OneDrive\Desktop\New folder\unsw\UNSW_NB15_testing-set.csv

**Methods applied**
- Replaced infinite values with NaN
- Dropped exact duplicate rows
- Applied median and most-frequent imputations through a train-fitted preprocessor
- Removed highly correlated numeric features above the fixed threshold
- Encoded the target with LabelEncoder and selected the top 75 transformed features
- Scaled the selected feature matrix for Logistic Regression

**Metrics/plots generated**
- Missingness plots before and after preprocessing
- Feature report with selected features, dropped correlations, and class mapping

**Decisions and justification**
- Target modeling uses attack_cat because it aligns with the multiclass IDS objective; the binary label column is excluded from features to prevent leakage.
- Feature selection is capped at 75 to preserve more minority-class signal while keeping compute bounded.
- All fitting decisions were learned from training data only, then applied to test data unchanged.

**Challenges and resolution**
- None

**Observations and insights**
- Dropped correlated features: 7
- Selected transformed features: 75
- Classes modeled: analysis, backdoor, dos, exploits, fuzzers, generic, normal, reconnaissance, shellcode, worms

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\models\label_encoder.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\preprocessor.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\selector.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\scaler.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\feature_info.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\feature_report.json

**Next step**: Generate compact EDA visuals on the cleaned and bounded feature space.
## Phase 4 - Decision-Focused EDA (2026-04-02 13:26:21)

**Objective**: Produce only the exploratory visuals needed to justify preprocessing and model choices.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\unsw\UNSW_NB15_training-set.csv

**Methods applied**
- Generated numeric distribution, boxplot, correlation, and categorical frequency visuals

**Metrics/plots generated**
- Numeric distribution panel
- Top variance boxplots
- Correlation heatmap
- Categorical frequency plot
- EDA summary Markdown

**Decisions and justification**
- EDA was intentionally constrained to decision-relevant analysis so the project stays finishable.
- Correlation and feature spread findings feed directly into the feature reduction strategy.

**Challenges and resolution**
- None

**Observations and insights**
- The training split contains both numeric and categorical predictors, so a mixed preprocessing pipeline is required.
- Outlier presence supports using tree models alongside a scaled linear baseline.

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\eda_summary.md
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\eda_numeric_distributions.png
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\eda_boxplots.png
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\eda_correlation_heatmap.png

**Next step**: Train the four bounded-complexity models and compare their cross-validation behavior.
## Phase 6 and 7 - Evaluation Setup and Model Training (2026-04-02 13:38:03)

**Objective**: Train all configured models for offline comparison and persist them for evaluation.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\models\scaler.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\feature_info.pkl

**Methods applied**
- Used 3-fold StratifiedKFold on training data only
- Trained Logistic Regression on scaled selected features
- Trained Random Forest, XGBoost, and LightGBM on the selected feature matrix
- Applied internal validation splits for XGBoost and LightGBM final fitting

**Metrics/plots generated**
- Cross-validation score summary
- Training time comparison data

**Decisions and justification**
- train_all mode preserves the experimental comparison pipeline for academic reporting.
- The offline comparison path remains separate from the production inference artifact.

**Challenges and resolution**
- None

**Observations and insights**
- All requested comparison models were trained and serialized.
- Training times and CV behavior are available for side-by-side comparison.

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_lr.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_rf.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_xgb.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_lgbm.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\training_cv_summary.json

**Next step**: Evaluate every trained model on the untouched predefined test set.
## Phase 8 - Final Evaluation and Comparison (2026-04-02 13:51:47)

**Objective**: Compare all trained models on the untouched test split and choose the production final_model.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_lr.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_rf.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_xgb.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_lgbm.pkl

**Methods applied**
- Generated predictions and probabilities on the untouched test set
- Computed accuracy, macro F1, weighted F1, per-class metrics, confusion matrices, ROC-AUC, and false alarm rates
- Ranked models by the existing macro-F1-first selection rule
- Calibrated only the winning model for production inference and tuned minority-class thresholds on a held-out validation split

**Metrics/plots generated**
- Per-model confusion matrices
- ROC comparison plot
- Metric comparison chart
- Training time comparison chart
- Per-class F1 heatmap

**Decisions and justification**
- The production final_model is selected by measured evaluation metrics; current winner: LGBM.
- Macro F1 remains the leading ranking signal because multiclass robustness matters more than raw accuracy alone.

**Challenges and resolution**
- None

**Observations and insights**
- Production final_model selected: LGBM
- Calibrated final_model accuracy: 0.7473
- Calibrated final_model macro F1: 0.5269
- Tuned thresholds: {'fuzzers': 0.3, 'backdoor': 0.15, 'analysis': 0.2, 'worms': 0.15}
- Offline comparison artifacts remain intact for academic reporting.

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\model_metrics.json
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\classification_reports.md
- C:\Users\ramil\OneDrive\Desktop\New folder\models\final_model.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\model_metric_comparison.png
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\per_class_f1_heatmap.png

**Next step**: Run bounded SHAP explainability on the selected final_model.
## Phase 8B - External Repo Benchmark Audit (2026-04-02 13:52:03)

**Objective**: Reproduce the public repo benchmark and measure how much of its score depends on target-derived leakage features.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\.codex-tmp\IoT-Network-Intrusion-Detection-System-UNSW-NB15\datasets\multi_data.csv

**Methods applied**
- Loaded the repo-prepared multi_data.csv file
- Reproduced the published RandomForest benchmark with the repo split
- Reran the same benchmark after removing attack_cat_* leakage columns from the feature matrix

**Metrics/plots generated**
- Repo benchmark leakage audit Markdown report
- Accuracy and macro-F1 comparison plot

**Decisions and justification**
- The external repo benchmark is documented for comparison, but it is not used as the production model selection source.
- Production inference remains limited to raw non-leaky features that are actually available at prediction time.

**Challenges and resolution**
- None

**Observations and insights**
- Repo-style benchmark accuracy with leakage: 97.27%
- Repo-style benchmark accuracy without attack_cat leakage columns: 89.21%

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\repo_benchmark_audit.json
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\repo_benchmark_audit.md
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\repo_benchmark_accuracy_comparison.png

**Next step**: Generate SHAP for the final production model and keep the SaaS inference path honest.
## Phase 9 - SHAP Explainability (2026-04-02 13:52:27)

**Objective**: Generate bounded-cost explainability artifacts for the production final_model.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\models\final_model.pkl

**Methods applied**
- Sampled up to 300 training rows in a class-aware way
- Computed SHAP values for the production final_model
- Generated summary, bar, and waterfall explainability plots

**Metrics/plots generated**
- SHAP summary plot
- SHAP feature-importance bar plot
- SHAP waterfall example

**Decisions and justification**
- SHAP is generated only for the production final_model so inference explainability stays single-model and predictable.
- The bounded sample keeps explainability practical for SaaS-oriented production runs.

**Challenges and resolution**
- None

**Observations and insights**
- The explainability package complements the evaluation metrics with feature-level reasoning.
- Interpretation should still be treated as sample-based evidence rather than a full-population guarantee.

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\outputs\shap_values.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\shap_summary.md
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\shap_summary.png
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\shap_bar.png
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\shap_waterfall_example.png

**Next step**: Review saved artifacts and confirm the local-only pipeline is reproducible.
## Phase 1 - Setup and Run Governance (2026-04-02 15:34:11)

**Objective**: Initialize folders, configuration, and runtime metadata for the local pipeline.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder

**Methods applied**
- Created artifacts/models/outputs folders
- Initialized journal and run summary
- Captured environment metadata

**Metrics/plots generated**
- Environment summary table
- Package version report

**Decisions and justification**
- Using local Python files as the source of truth keeps the workflow reproducible without depending on notebook tooling.
- A fixed configuration object prevents drift in model and reporting settings.

**Challenges and resolution**
- None

**Observations and insights**
- Workspace scaffolding is ready for dataset-driven phases.

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\logs\project_journal.md
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\run_summary.md
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\environment_versions.json

**Next step**: Load and validate the predefined train/test datasets.
## Phase 1 - Setup and Run Governance (2026-04-02 15:34:59)

**Objective**: Initialize folders, configuration, and runtime metadata for the local pipeline.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder

**Methods applied**
- Created artifacts/models/outputs folders
- Initialized journal and run summary
- Captured environment metadata

**Metrics/plots generated**
- Environment summary table
- Package version report

**Decisions and justification**
- Using local Python files as the source of truth keeps the workflow reproducible without depending on notebook tooling.
- A fixed configuration object prevents drift in model and reporting settings.

**Challenges and resolution**
- None

**Observations and insights**
- Workspace scaffolding is ready for dataset-driven phases.

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\logs\project_journal.md
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\run_summary.md
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\environment_versions.json

**Next step**: Load and validate the predefined train/test datasets.
## Phase 2 - Data Ingestion and Profiling (2026-04-02 15:35:06)

**Objective**: Load the predefined UNSW-NB15 train/test split, validate schema integrity, and profile the target distribution.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\unsw\UNSW_NB15_training-set.csv
- C:\Users\ramil\OneDrive\Desktop\New folder\unsw\UNSW_NB15_testing-set.csv

**Methods applied**
- Loaded CSV files with pandas
- Validated schema and target presence
- Attempted parquet caching for repeated local reads

**Metrics/plots generated**
- Train/test label distribution plot
- Missing values summary plot
- Schema profile JSON

**Decisions and justification**
- The predefined train/test split is preserved exactly to avoid leakage and keep the project bounded.
- Parquet caching is enabled when available because it reduces repeated local read time without changing semantics.

**Challenges and resolution**
- None

**Observations and insights**
- Schema match: True
- Target present in both splits: True
- Binary label retained for reference but excluded from multiclass target modeling: True

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\data_profile.json
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\label_distribution.png
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\missing_values_train.png

**Next step**: Clean the data and prepare train-only feature engineering.
## Phase 3 and 5 - Cleaning, Preprocessing, and Feature Reduction (2026-04-02 15:35:14)

**Objective**: Clean the train/test splits, encode the multiclass target, reduce redundant features, and save preprocessing artifacts.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\unsw\UNSW_NB15_training-set.csv
- C:\Users\ramil\OneDrive\Desktop\New folder\unsw\UNSW_NB15_testing-set.csv

**Methods applied**
- Replaced infinite values with NaN
- Dropped exact duplicate rows
- Applied median and most-frequent imputations through a train-fitted preprocessor
- Removed highly correlated numeric features above the fixed threshold
- Encoded the target with LabelEncoder and selected the top 75 transformed features
- Scaled the selected feature matrix for Logistic Regression

**Metrics/plots generated**
- Missingness plots before and after preprocessing
- Feature report with selected features, dropped correlations, and class mapping

**Decisions and justification**
- Target modeling uses attack_cat because it aligns with the multiclass IDS objective; the binary label column is excluded from features to prevent leakage.
- Feature selection is capped at 75 to preserve more minority-class signal while keeping compute bounded.
- All fitting decisions were learned from training data only, then applied to test data unchanged.

**Challenges and resolution**
- None

**Observations and insights**
- Dropped correlated features: 7
- Selected transformed features: 75
- Classes modeled: analysis, backdoor, dos, exploits, fuzzers, generic, normal, reconnaissance, shellcode, worms

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\models\label_encoder.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\preprocessor.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\selector.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\scaler.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\feature_info.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\feature_report.json

**Next step**: Generate compact EDA visuals on the cleaned and bounded feature space.
## Phase 4 - Decision-Focused EDA (2026-04-02 15:35:30)

**Objective**: Produce only the exploratory visuals needed to justify preprocessing and model choices.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\unsw\UNSW_NB15_training-set.csv

**Methods applied**
- Generated numeric distribution, boxplot, correlation, and categorical frequency visuals

**Metrics/plots generated**
- Numeric distribution panel
- Top variance boxplots
- Correlation heatmap
- Categorical frequency plot
- EDA summary Markdown

**Decisions and justification**
- EDA was intentionally constrained to decision-relevant analysis so the project stays finishable.
- Correlation and feature spread findings feed directly into the feature reduction strategy.

**Challenges and resolution**
- None

**Observations and insights**
- The training split contains both numeric and categorical predictors, so a mixed preprocessing pipeline is required.
- Outlier presence supports using tree models alongside a scaled linear baseline.

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\eda_summary.md
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\eda_numeric_distributions.png
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\eda_boxplots.png
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\eda_correlation_heatmap.png

**Next step**: Train the four bounded-complexity models and compare their cross-validation behavior.
## Phase 6 and 7 - Evaluation Setup and Model Training (2026-04-02 15:48:16)

**Objective**: Train all configured models for offline comparison and persist them for evaluation.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\models\scaler.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\feature_info.pkl

**Methods applied**
- Used 3-fold StratifiedKFold on training data only
- Trained Logistic Regression on scaled selected features
- Trained Random Forest, XGBoost, and LightGBM on the selected feature matrix
- Applied internal validation splits for XGBoost and LightGBM final fitting

**Metrics/plots generated**
- Cross-validation score summary
- Training time comparison data

**Decisions and justification**
- train_all mode preserves the experimental comparison pipeline for academic reporting.
- The offline comparison path remains separate from the production inference artifact.

**Challenges and resolution**
- None

**Observations and insights**
- All requested comparison models were trained and serialized.
- Training times and CV behavior are available for side-by-side comparison.

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_lr.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_rf.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_xgb.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_lgbm.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\training_cv_summary.json

**Next step**: Evaluate every trained model on the untouched predefined test set.
## Phase 8 - Final Evaluation and Comparison (2026-04-02 16:00:59)

**Objective**: Compare all trained models on the untouched test split and choose the production final_model.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_lr.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_rf.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_xgb.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\models\model_lgbm.pkl

**Methods applied**
- Generated predictions and probabilities on the untouched test set
- Computed accuracy, macro F1, weighted F1, per-class metrics, confusion matrices, ROC-AUC, and false alarm rates
- Ranked models by the existing macro-F1-first selection rule
- Calibrated only the winning model for production inference and tuned minority-class thresholds on a held-out validation split

**Metrics/plots generated**
- Per-model confusion matrices
- ROC comparison plot
- Metric comparison chart
- Training time comparison chart
- Per-class F1 heatmap

**Decisions and justification**
- The production final_model is selected by measured evaluation metrics; current winner: LGBM.
- Macro F1 remains the leading ranking signal because multiclass robustness matters more than raw accuracy alone.

**Challenges and resolution**
- None

**Observations and insights**
- Production final_model selected: LGBM
- Calibrated final_model accuracy: 0.7473
- Calibrated final_model macro F1: 0.5269
- Tuned thresholds: {'fuzzers': 0.3, 'backdoor': 0.15, 'analysis': 0.2, 'worms': 0.15}
- Offline comparison artifacts remain intact for academic reporting.

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\model_metrics.json
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\classification_reports.md
- C:\Users\ramil\OneDrive\Desktop\New folder\models\final_model.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\model_metric_comparison.png
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\per_class_f1_heatmap.png

**Next step**: Run bounded SHAP explainability on the selected final_model.
## Phase 8B - External Repo Benchmark Audit (2026-04-02 16:01:17)

**Objective**: Reproduce the public repo benchmark and measure how much of its score depends on target-derived leakage features.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\.codex-tmp\IoT-Network-Intrusion-Detection-System-UNSW-NB15\datasets\multi_data.csv

**Methods applied**
- Loaded the repo-prepared multi_data.csv file
- Reproduced the published RandomForest benchmark with the repo split
- Reran the same benchmark after removing attack_cat_* leakage columns from the feature matrix

**Metrics/plots generated**
- Repo benchmark leakage audit Markdown report
- Accuracy and macro-F1 comparison plot

**Decisions and justification**
- The external repo benchmark is documented for comparison, but it is not used as the production model selection source.
- Production inference remains limited to raw non-leaky features that are actually available at prediction time.

**Challenges and resolution**
- None

**Observations and insights**
- Repo-style benchmark accuracy with leakage: 97.27%
- Repo-style benchmark accuracy without attack_cat leakage columns: 89.21%

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\repo_benchmark_audit.json
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\repo_benchmark_audit.md
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\repo_benchmark_accuracy_comparison.png

**Next step**: Generate SHAP for the final production model and keep the SaaS inference path honest.
## Phase 9 - SHAP Explainability (2026-04-02 16:01:46)

**Objective**: Generate bounded-cost explainability artifacts for the production final_model.

**Inputs used**
- C:\Users\ramil\OneDrive\Desktop\New folder\models\final_model.pkl

**Methods applied**
- Sampled up to 300 training rows in a class-aware way
- Computed SHAP values for the production final_model
- Generated summary, bar, and waterfall explainability plots

**Metrics/plots generated**
- SHAP summary plot
- SHAP feature-importance bar plot
- SHAP waterfall example

**Decisions and justification**
- SHAP is generated only for the production final_model so inference explainability stays single-model and predictable.
- The bounded sample keeps explainability practical for SaaS-oriented production runs.

**Challenges and resolution**
- None

**Observations and insights**
- The explainability package complements the evaluation metrics with feature-level reasoning.
- Interpretation should still be treated as sample-based evidence rather than a full-population guarantee.

**Outputs saved**
- C:\Users\ramil\OneDrive\Desktop\New folder\outputs\shap_values.pkl
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\reports\shap_summary.md
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\shap_summary.png
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\shap_bar.png
- C:\Users\ramil\OneDrive\Desktop\New folder\artifacts\plots\shap_waterfall_example.png

**Next step**: Review saved artifacts and confirm the local-only pipeline is reproducible.
