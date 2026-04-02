from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import joblib
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

from src.config import get_config
from src.data_io import cache_parquet, load_datasets, validate_schema
from src.eda import (
    save_boxplots,
    save_categorical_frequency_plots,
    save_correlation_heatmap,
    save_label_distribution,
    save_missing_values_plot,
    save_numeric_distribution_plots,
)
from src.evaluate import evaluate_final_model, evaluate_models, select_best_model
from src.explain import run_shap_analysis
from src.inference import explain_records, load_production_artifacts, predict_records
from src.preprocess import prepare_features, save_preprocessing_artifacts
from src.repo_audit import run_repo_benchmark_audit
from src.train import TrainedModel, train_calibrated_final_model, train_models
from src.utils.logging_utils import (
    RuntimeTracker,
    append_phase_journal,
    append_run_summary,
    ensure_directories,
    init_project_documents,
    log_progress,
    save_json,
    write_text,
)


def config_markdown(config) -> str:
    return "\n".join(
        [
            "| Key | Value |",
            "| --- | --- |",
            f"| Project root | `{config.project_root}` |",
            f"| Train CSV | `{config.train_csv}` |",
            f"| Test CSV | `{config.test_csv}` |",
            f"| Target column | `{config.target_column}` |",
            f"| Random state | `{config.random_state}` |",
            f"| Mode | `{config.mode}` |",
            f"| Default production model | `{config.default_production_model}` |",
            f"| Feature cap | `{config.feature_cap}` |",
            f"| CV folds | `{config.cv_splits}` |",
            f"| SHAP sample size | `{config.shap_sample_size}` |",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IDS ML pipeline")
    parser.add_argument("--mode", choices=["production", "train_all"], default="production")
    parser.add_argument("--retrain-final-model", action="store_true")
    return parser.parse_args()


def resolve_final_model_name(config) -> tuple[str, str]:
    if config.model_metrics_path.exists():
        metrics_report = json.loads(config.model_metrics_path.read_text(encoding="utf-8"))
        return select_best_model(metrics_report), "evaluation_metrics"
    if config.final_model_manifest_path.exists():
        manifest = json.loads(config.final_model_manifest_path.read_text(encoding="utf-8"))
        manifest_model = manifest.get("final_model")
        if manifest_model:
            return manifest_model, "existing_manifest"
    return config.default_production_model, "default_fallback"


def save_final_model_manifest(config, final_model_name: str, mode: str, selected_features: list[str], class_names: list[str], source: str) -> dict:
    manifest = {
        "final_model": final_model_name,
        "final_model_path": str(config.final_model_path),
        "mode": mode,
        "selection_source": source,
        "selected_features": selected_features,
        "class_names": class_names,
        "uses_scaled_features": final_model_name == "lr",
    }
    save_json(config.final_model_manifest_path, manifest)
    return manifest


def load_feature_info(config) -> dict:
    return joblib.load(config.models_dir / "feature_info.pkl")


def update_feature_info_thresholds(config, threshold_metadata: dict | None) -> dict:
    feature_info = load_feature_info(config)
    if not threshold_metadata:
        return feature_info
    metadata = threshold_metadata
    feature_info["class_thresholds"] = metadata.get("thresholds_by_name", {})
    feature_info["threshold_tuning"] = {
        "validation_macro_f1": metadata.get("validation_macro_f1"),
        "thresholds_by_index": metadata.get("thresholds_by_index", {}),
    }
    joblib.dump(feature_info, config.models_dir / "feature_info.pkl")
    return feature_info


def resolve_final_trained_model(config, final_model_name: str, prepared) -> tuple[TrainedModel, str]:
    manifest = None
    if config.final_model_manifest_path.exists():
        manifest = json.loads(config.final_model_manifest_path.read_text(encoding="utf-8"))

    if not config.force_retrain_final_model:
        if config.final_model_path.exists() and manifest and manifest.get("final_model") == final_model_name:
            estimator = joblib.load(config.final_model_path)
            return TrainedModel(name=final_model_name, estimator=estimator, cv_scores=[], train_seconds=0.0), "loaded_final_model"

    trained = train_calibrated_final_model(
        final_model_name,
        prepared.X_train_selected,
        prepared.X_train_scaled,
        prepared.y_train,
        len(prepared.class_names),
        prepared.class_names,
        config,
        save_path=str(config.final_model_path),
    )
    return trained, "fresh_production_training"


def main() -> None:
    args = parse_args()
    config = get_config(mode=args.mode, force_retrain_final_model=args.retrain_final_model)
    ensure_directories(
        [
            config.artifacts_dir,
            config.plots_dir,
            config.reports_dir,
            config.logs_dir,
            config.models_dir,
            config.outputs_dir,
        ]
    )

    journal_path = config.logs_dir / "project_journal.md"
    run_summary_path = config.reports_dir / "run_summary.md"
    init_project_documents(journal_path, run_summary_path, config_markdown(config))
    tracker = RuntimeTracker()

    package_versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }
    save_json(config.reports_dir / "environment_versions.json", package_versions)
    append_phase_journal(
        journal_path,
        "Phase 1 - Setup and Run Governance",
        "Initialize folders, configuration, and runtime metadata for the local pipeline.",
        [str(config.project_root)],
        ["Created artifacts/models/outputs folders", "Initialized journal and run summary", "Captured environment metadata"],
        ["Environment summary table", "Package version report"],
        [
            "Using local Python files as the source of truth keeps the workflow reproducible without depending on notebook tooling.",
            "A fixed configuration object prevents drift in model and reporting settings.",
        ],
        [],
        ["Workspace scaffolding is ready for dataset-driven phases."],
        [str(journal_path), str(run_summary_path), str(config.reports_dir / "environment_versions.json")],
        "Load and validate the predefined train/test datasets.",
    )
    append_run_summary(
        run_summary_path,
        "Phase 1",
        "Local workspace folders were created, environment metadata was captured, and the run configuration was initialized.",
    )

    with tracker.track("phase_2_data_ingestion"):
        train_df, test_df = load_datasets(config)
        schema_report = validate_schema(train_df, test_df, config)
        parquet_status = {
            "train_cached": cache_parquet(train_df, config.train_parquet),
            "test_cached": cache_parquet(test_df, config.test_parquet),
        }
    save_json(config.reports_dir / "data_profile.json", {**schema_report, **parquet_status})
    save_label_distribution(train_df, test_df, config.target_column, config.plots_dir / "label_distribution.png")
    save_missing_values_plot(train_df, config.plots_dir / "missing_values_train.png")
    append_phase_journal(
        journal_path,
        "Phase 2 - Data Ingestion and Profiling",
        "Load the predefined UNSW-NB15 train/test split, validate schema integrity, and profile the target distribution.",
        [str(config.train_csv), str(config.test_csv)],
        ["Loaded CSV files with pandas", "Validated schema and target presence", "Attempted parquet caching for repeated local reads"],
        ["Train/test label distribution plot", "Missing values summary plot", "Schema profile JSON"],
        [
            "The predefined train/test split is preserved exactly to avoid leakage and keep the project bounded.",
            "Parquet caching is enabled when available because it reduces repeated local read time without changing semantics.",
        ],
        [] if schema_report["schema_match"] else ["Train/test columns did not match; pipeline should not continue until fixed."],
        [
            f"Schema match: {schema_report['schema_match']}",
            f"Target present in both splits: {schema_report['target_present']}",
            f"Binary label retained for reference but excluded from multiclass target modeling: {schema_report['binary_label_present']}",
        ],
        [
            str(config.reports_dir / "data_profile.json"),
            str(config.plots_dir / "label_distribution.png"),
            str(config.plots_dir / "missing_values_train.png"),
        ],
        "Clean the data and prepare train-only feature engineering.",
    )
    append_run_summary(
        run_summary_path,
        "Phase 2",
        f"Loaded {schema_report['train_rows']} training rows and {schema_report['test_rows']} testing rows with a schema match of {schema_report['schema_match']}.",
    )
    if not schema_report["schema_match"] or not schema_report["target_present"]:
        raise RuntimeError("Dataset schema validation failed.")

    with tracker.track("phase_3_5_preprocessing_and_feature_reduction"):
        prepared, preprocessing_report = prepare_features(train_df, test_df, config)
    save_preprocessing_artifacts(prepared, config)
    save_json(config.reports_dir / "feature_report.json", preprocessing_report)
    save_missing_values_plot(train_df.replace([float("inf"), float("-inf")], pd.NA), config.plots_dir / "missing_before_cleaning.png")
    save_missing_values_plot(
        pd.DataFrame(prepared.X_train_selected, columns=prepared.selected_feature_names),
        config.plots_dir / "missing_after_cleaning.png",
    )
    append_phase_journal(
        journal_path,
        "Phase 3 and 5 - Cleaning, Preprocessing, and Feature Reduction",
        "Clean the train/test splits, encode the multiclass target, reduce redundant features, and save preprocessing artifacts.",
        [str(config.train_csv), str(config.test_csv)],
        [
            "Replaced infinite values with NaN",
            "Dropped exact duplicate rows",
            "Applied median and most-frequent imputations through a train-fitted preprocessor",
            "Removed highly correlated numeric features above the fixed threshold",
            f"Encoded the target with LabelEncoder and selected the top {config.feature_cap} transformed features",
            "Scaled the selected feature matrix for Logistic Regression",
        ],
        [
            "Missingness plots before and after preprocessing",
            "Feature report with selected features, dropped correlations, and class mapping",
        ],
        [
            "Target modeling uses attack_cat because it aligns with the multiclass IDS objective; the binary label column is excluded from features to prevent leakage.",
            f"Feature selection is capped at {config.feature_cap} to preserve more minority-class signal while keeping compute bounded.",
            "All fitting decisions were learned from training data only, then applied to test data unchanged.",
        ],
        [],
        [
            f"Dropped correlated features: {len(prepared.dropped_correlated_features)}",
            f"Selected transformed features: {len(prepared.selected_feature_names)}",
            f"Classes modeled: {', '.join(prepared.class_names)}",
        ],
        [
            str(config.models_dir / "label_encoder.pkl"),
            str(config.models_dir / "preprocessor.pkl"),
            str(config.models_dir / "selector.pkl"),
            str(config.models_dir / "scaler.pkl"),
            str(config.models_dir / "feature_info.pkl"),
            str(config.reports_dir / "feature_report.json"),
        ],
        "Generate compact EDA visuals on the cleaned and bounded feature space.",
    )
    append_run_summary(
        run_summary_path,
        "Phase 3 and 5",
        f"Removed {len(prepared.dropped_correlated_features)} correlated numeric features and retained {len(prepared.selected_feature_names)} transformed features for modeling.",
    )

    with tracker.track("phase_4_eda"):
        cleaned_train = train_df.copy()
        cleaned_train[config.target_column] = cleaned_train[config.target_column].astype(str).str.strip().str.lower()
        features_only = cleaned_train.drop(columns=[config.target_column, config.binary_label_column], errors="ignore")
        numeric_cols = features_only.select_dtypes(include="number").columns.tolist()
        categorical_cols = [col for col in features_only.columns if col not in numeric_cols]
        save_numeric_distribution_plots(cleaned_train, numeric_cols, config.plots_dir / "eda_numeric_distributions.png")
        save_boxplots(cleaned_train, numeric_cols, config.plots_dir / "eda_boxplots.png")
        save_correlation_heatmap(cleaned_train, numeric_cols, config.plots_dir / "eda_correlation_heatmap.png")
        if categorical_cols:
            save_categorical_frequency_plots(cleaned_train, categorical_cols, config.plots_dir / "eda_categorical_frequencies.png")
    eda_summary = "\n".join(
        [
            "# EDA Summary",
            "",
            f"- Train class count: {len(cleaned_train[config.target_column].unique())}",
            f"- Numeric feature count before reduction: {len(numeric_cols)}",
            f"- Categorical feature count before encoding: {len(categorical_cols)}",
            f"- Correlated numeric features removed: {len(prepared.dropped_correlated_features)}",
            "- EDA was intentionally limited to class balance, numeric spread, outliers, correlation, and categorical frequency patterns.",
        ]
    )
    write_text(config.reports_dir / "eda_summary.md", eda_summary)
    append_phase_journal(
        journal_path,
        "Phase 4 - Decision-Focused EDA",
        "Produce only the exploratory visuals needed to justify preprocessing and model choices.",
        [str(config.train_csv)],
        ["Generated numeric distribution, boxplot, correlation, and categorical frequency visuals"],
        [
            "Numeric distribution panel",
            "Top variance boxplots",
            "Correlation heatmap",
            "Categorical frequency plot",
            "EDA summary Markdown",
        ],
        [
            "EDA was intentionally constrained to decision-relevant analysis so the project stays finishable.",
            "Correlation and feature spread findings feed directly into the feature reduction strategy.",
        ],
        [],
        [
            "The training split contains both numeric and categorical predictors, so a mixed preprocessing pipeline is required.",
            "Outlier presence supports using tree models alongside a scaled linear baseline.",
        ],
        [
            str(config.reports_dir / "eda_summary.md"),
            str(config.plots_dir / "eda_numeric_distributions.png"),
            str(config.plots_dir / "eda_boxplots.png"),
            str(config.plots_dir / "eda_correlation_heatmap.png"),
        ],
        "Train the four bounded-complexity models and compare their cross-validation behavior.",
    )
    append_run_summary(
        run_summary_path,
        "Phase 4",
        "EDA visuals were generated for class balance, numeric spread, outliers, correlation, and categorical frequencies.",
    )

    if config.mode == "train_all":
        with tracker.track("phase_6_7_training"):
            trained_models = train_models(
                prepared.X_train_selected,
                prepared.X_train_scaled,
                prepared.y_train,
                len(prepared.class_names),
                config,
            )
        save_json(
            config.reports_dir / "training_cv_summary.json",
            {
                name: {"cv_scores": model.cv_scores, "train_seconds": model.train_seconds}
                for name, model in trained_models.items()
            },
        )
        append_phase_journal(
            journal_path,
            "Phase 6 and 7 - Evaluation Setup and Model Training",
            "Train all configured models for offline comparison and persist them for evaluation.",
            [str(config.models_dir / "scaler.pkl"), str(config.models_dir / "feature_info.pkl")],
            [
                "Used 3-fold StratifiedKFold on training data only",
                "Trained Logistic Regression on scaled selected features",
                "Trained Random Forest, XGBoost, and LightGBM on the selected feature matrix",
                "Applied internal validation splits for XGBoost and LightGBM final fitting",
            ],
            ["Cross-validation score summary", "Training time comparison data"],
            [
                "train_all mode preserves the experimental comparison pipeline for academic reporting.",
                "The offline comparison path remains separate from the production inference artifact.",
            ],
            [],
            [
                "All requested comparison models were trained and serialized.",
                "Training times and CV behavior are available for side-by-side comparison.",
            ],
            [
                str(config.models_dir / "model_lr.pkl"),
                str(config.models_dir / "model_rf.pkl"),
                str(config.models_dir / "model_xgb.pkl"),
                str(config.models_dir / "model_lgbm.pkl"),
                str(config.reports_dir / "training_cv_summary.json"),
            ],
            "Evaluate every trained model on the untouched predefined test set.",
        )
        append_run_summary(
            run_summary_path,
            "Phase 6 and 7",
            "train_all mode trained the full comparison set and preserved the offline evaluation path.",
        )

        with tracker.track("phase_8_evaluation"):
            metrics_report, classification_markdown = evaluate_models(
                trained_models,
                prepared.X_test_selected,
                prepared.X_test_scaled,
                prepared.y_test,
                prepared.class_names,
                config.plots_dir,
            )
        final_model_name = select_best_model(metrics_report)
        save_json(config.reports_dir / "model_metrics.json", metrics_report)
        write_text(config.reports_dir / "classification_reports.md", classification_markdown)
        selection_source = "fresh_offline_evaluation"
        with tracker.track("phase_8a_final_model_calibration"):
            final_trained = train_calibrated_final_model(
                final_model_name,
                prepared.X_train_selected,
                prepared.X_train_scaled,
                prepared.y_train,
                len(prepared.class_names),
                prepared.class_names,
                config,
                save_path=str(config.final_model_path),
            )
        feature_info = update_feature_info_thresholds(config, final_trained.metadata)
        calibrated_final_metrics = evaluate_final_model(
            final_trained.estimator,
            final_model_name,
            prepared.X_test_selected,
            prepared.X_test_scaled,
            prepared.y_test,
            prepared.class_names,
            thresholds_by_index=(final_trained.metadata or {}).get("thresholds_by_index"),
        )
        append_phase_journal(
            journal_path,
            "Phase 8 - Final Evaluation and Comparison",
            "Compare all trained models on the untouched test split and choose the production final_model.",
            [
                str(config.models_dir / "model_lr.pkl"),
                str(config.models_dir / "model_rf.pkl"),
                str(config.models_dir / "model_xgb.pkl"),
                str(config.models_dir / "model_lgbm.pkl"),
            ],
            [
                "Generated predictions and probabilities on the untouched test set",
                "Computed accuracy, macro F1, weighted F1, per-class metrics, confusion matrices, ROC-AUC, and false alarm rates",
                "Ranked models by the existing macro-F1-first selection rule",
                "Calibrated only the winning model for production inference and tuned minority-class thresholds on a held-out validation split",
            ],
            [
                "Per-model confusion matrices",
                "ROC comparison plot",
                "Metric comparison chart",
                "Training time comparison chart",
                "Per-class F1 heatmap",
            ],
            [
                f"The production final_model is selected by measured evaluation metrics; current winner: {final_model_name.upper()}.",
                "Macro F1 remains the leading ranking signal because multiclass robustness matters more than raw accuracy alone.",
            ],
            [],
            [
                f"Production final_model selected: {final_model_name.upper()}",
                f"Calibrated final_model accuracy: {calibrated_final_metrics['accuracy']:.4f}",
                f"Calibrated final_model macro F1: {calibrated_final_metrics['macro_f1']:.4f}",
                f"Tuned thresholds: {feature_info.get('class_thresholds', {})}",
                "Offline comparison artifacts remain intact for academic reporting.",
            ],
            [
                str(config.reports_dir / "model_metrics.json"),
                str(config.reports_dir / "classification_reports.md"),
                str(config.final_model_path),
                str(config.plots_dir / "model_metric_comparison.png"),
                str(config.plots_dir / "per_class_f1_heatmap.png"),
            ],
            "Run bounded SHAP explainability on the selected final_model.",
        )
        append_run_summary(
            run_summary_path,
            "Phase 8",
            f"train_all mode completed offline evaluation, selected {final_model_name.upper()} as the production final_model, and saved a calibrated production artifact.",
        )
    else:
        final_model_name, selection_source = resolve_final_model_name(config)
        with tracker.track("phase_6_7_training"):
            final_trained, production_source = resolve_final_trained_model(config, final_model_name, prepared)
        feature_info = update_feature_info_thresholds(config, final_trained.metadata)
        calibrated_final_metrics = evaluate_final_model(
            final_trained.estimator,
            final_model_name,
            prepared.X_test_selected,
            prepared.X_test_scaled,
            prepared.y_test,
            prepared.class_names,
            thresholds_by_index=(final_trained.metadata or {}).get("thresholds_by_index") or feature_info.get("threshold_tuning", {}).get("thresholds_by_index"),
        )
        append_phase_journal(
            journal_path,
            "Phase 6 and 7 - Production Final Model Preparation",
            "Resolve, train, or load only the production final_model without running multi-model comparison.",
            [str(config.models_dir / "scaler.pkl"), str(config.models_dir / "feature_info.pkl"), str(config.model_metrics_path)],
            [
                "Resolved the production final_model using the existing ranking policy",
                "Skipped multi-model training in production mode",
                "Loaded or trained only the selected calibrated final_model and saved it as models/final_model.pkl",
            ],
            ["Final model manifest", "Production model artifact"],
            [
                f"Production mode defaults to a single-model SaaS path, using {final_model_name.upper()} as the resolved final_model.",
                "Existing offline evaluation artifacts are reused when available to avoid redundant training.",
            ],
            [],
            [
                f"Final model selection source: {selection_source}",
                f"Production artifact source: {production_source}",
                f"Calibrated final_model macro F1: {calibrated_final_metrics['macro_f1']:.4f}",
                f"Tuned thresholds: {feature_info.get('class_thresholds', {})}",
            ],
            [str(config.final_model_path)],
            "Reuse existing offline evaluation artifacts if present, then generate SHAP for the final_model.",
        )
        append_run_summary(
            run_summary_path,
            "Phase 6 and 7",
            f"production mode resolved {final_model_name.upper()} as the final_model and prepared models/final_model.pkl without running multi-model training.",
        )

        if config.model_metrics_path.exists():
            append_phase_journal(
                journal_path,
                "Phase 8 - Offline Evaluation Artifacts Retained",
                "Retain existing comparison metrics and plots without rerunning the offline evaluation pipeline.",
                [str(config.model_metrics_path)],
                ["Loaded the existing model comparison outputs as the source of final_model selection"],
                ["Existing comparison reports and plots retained"],
                ["Production mode avoids redoing offline evaluation work when artifacts already exist."],
                [],
                [f"Existing evaluation artifacts preserved for final_model {final_model_name.upper()}."],
                [str(config.model_metrics_path), str(config.reports_dir / 'classification_reports.md')],
                "Run bounded SHAP explainability on the selected final_model.",
            )
            append_run_summary(
                run_summary_path,
                "Phase 8",
                f"production mode retained the existing offline evaluation artifacts and kept {final_model_name.upper()} as the final_model.",
        )
        final_trained.name = final_model_name

    with tracker.track("phase_8b_repo_benchmark_audit"):
        repo_audit_payload = run_repo_benchmark_audit(config)
    if repo_audit_payload is not None:
        append_phase_journal(
            journal_path,
            "Phase 8B - External Repo Benchmark Audit",
            "Reproduce the public repo benchmark and measure how much of its score depends on target-derived leakage features.",
            [str(config.repo_multi_data_path)],
            [
                "Loaded the repo-prepared multi_data.csv file",
                "Reproduced the published RandomForest benchmark with the repo split",
                "Reran the same benchmark after removing attack_cat_* leakage columns from the feature matrix",
            ],
            [
                "Repo benchmark leakage audit Markdown report",
                "Accuracy and macro-F1 comparison plot",
            ],
            [
                "The external repo benchmark is documented for comparison, but it is not used as the production model selection source.",
                "Production inference remains limited to raw non-leaky features that are actually available at prediction time.",
            ],
            [],
            [
                f"Repo-style benchmark accuracy with leakage: {repo_audit_payload['with_leakage']['accuracy'] * 100:.2f}%",
                f"Repo-style benchmark accuracy without attack_cat leakage columns: {repo_audit_payload['without_attack_cat_dummies']['accuracy'] * 100:.2f}%",
            ],
            [
                str(config.repo_audit_json_path),
                str(config.repo_audit_md_path),
                str(config.repo_audit_plot_path),
            ],
            "Generate SHAP for the final production model and keep the SaaS inference path honest.",
        )
        append_run_summary(
            run_summary_path,
            "Phase 8B",
            "The external repo benchmark was reproduced and audited; its 97% result depends heavily on target-derived attack_cat dummy columns that are not valid production inputs.",
        )
    else:
        append_run_summary(
            run_summary_path,
            "Phase 8B",
            "Repo benchmark audit was skipped because the cloned comparison repository was not present in the workspace.",
        )

    final_manifest = save_final_model_manifest(
        config,
        final_model_name,
        config.mode,
        prepared.selected_feature_names,
        prepared.class_names,
        selection_source,
    )
    with tracker.track("phase_9_shap"):
        shap_payload = run_shap_analysis(
            final_trained.estimator,
            final_model_name,
            prepared.X_train_selected if final_model_name != "lr" else prepared.X_train_scaled,
            prepared.y_train,
            prepared.selected_feature_names,
            prepared.class_names,
            config,
        )
    shap_summary_md = "\n".join(
        [
            "# SHAP Summary",
            "",
            f"- Final model explained: {final_model_name.upper()}",
            f"- Sample size used: {len(shap_payload['sample_indices'])}",
            f"- Feature count explained: {len(shap_payload['feature_names'])}",
            "- SHAP was intentionally limited to a bounded stratified sample for the production final_model.",
        ]
    )
    write_text(config.reports_dir / "shap_summary.md", shap_summary_md)
    append_phase_journal(
        journal_path,
        "Phase 9 - SHAP Explainability",
        "Generate bounded-cost explainability artifacts for the production final_model.",
        [str(config.final_model_path)],
        [
            "Sampled up to 300 training rows in a class-aware way",
            "Computed SHAP values for the production final_model",
            "Generated summary, bar, and waterfall explainability plots",
        ],
        ["SHAP summary plot", "SHAP feature-importance bar plot", "SHAP waterfall example"],
        [
            "SHAP is generated only for the production final_model so inference explainability stays single-model and predictable.",
            "The bounded sample keeps explainability practical for SaaS-oriented production runs.",
        ],
        [],
        [
            "The explainability package complements the evaluation metrics with feature-level reasoning.",
            "Interpretation should still be treated as sample-based evidence rather than a full-population guarantee.",
        ],
        [
            str(config.outputs_dir / "shap_values.pkl"),
            str(config.reports_dir / "shap_summary.md"),
            str(config.plots_dir / "shap_summary.png"),
            str(config.plots_dir / "shap_bar.png"),
            str(config.plots_dir / "shap_waterfall_example.png"),
        ],
        "Review saved artifacts and confirm the local-only pipeline is reproducible.",
    )
    append_run_summary(
        run_summary_path,
        "Phase 9",
        f"SHAP explainability was generated for the production final_model {final_model_name.upper()} using a bounded stratified sample.",
    )

    production_artifacts = load_production_artifacts(config.project_root)
    sample_input = train_df.drop(columns=[config.target_column, config.binary_label_column], errors="ignore").head(1)
    sample_prediction = predict_records(sample_input, production_artifacts)
    sample_explanation = explain_records(sample_input, production_artifacts, top_n=5)
    final_feature_info = load_feature_info(config)

    save_json(
        config.reports_dir / "final_summary.json",
        {
            "final_model": final_model_name,
            "final_model_path": str(config.final_model_path),
            "mode": config.mode,
            "selected_features": prepared.selected_feature_names,
            "class_names": prepared.class_names,
            "artifacts_root": str(config.artifacts_dir),
            "models_root": str(config.models_dir),
            "outputs_root": str(config.outputs_dir),
            "manifest_path": str(config.final_model_manifest_path),
            "repo_audit_report_path": str(config.repo_audit_md_path) if config.repo_audit_md_path.exists() else None,
            "interaction_feature_names": final_feature_info.get("interaction_feature_names", []),
            "class_thresholds": final_feature_info.get("class_thresholds", {}),
            "threshold_tuning": final_feature_info.get("threshold_tuning", {}),
            "calibrated_final_model_metrics": calibrated_final_metrics,
            "sample_prediction": sample_prediction,
            "sample_explanation": sample_explanation,
        },
    )
    save_json(config.reports_dir / "runtime_profile.json", tracker.timings)
    log_progress("Pipeline completed successfully")


if __name__ == "__main__":
    main()
