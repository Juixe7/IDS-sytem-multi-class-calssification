from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import Config


PLOT_EXPLANATIONS = {
    "model_metric_comparison.png": "Official predefined-split comparison across the offline evaluation models.",
    "training_time_comparison.png": "Training-time comparison for the evaluated models.",
    "roc_curve_comparison.png": "One-vs-rest ROC comparison on the untouched predefined test split.",
    "per_class_f1_heatmap.png": "Per-class F1 shows where the multiclass IDS pipeline performs well and where it struggles.",
    "confusion_matrix_rf.png": "Confusion matrix for the current production model.",
    "repo_benchmark_accuracy_comparison.png": "Audit of the cloned repo benchmark showing leakage vs non-leaky performance.",
    "shap_summary.png": "Global SHAP summary for the current final model.",
    "shap_bar.png": "Mean absolute SHAP importance for the final model.",
    "shap_waterfall_example.png": "Single-sample SHAP waterfall illustration.",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def load_feature_artifacts(project_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root) if project_root is not None else Config().project_root
    config = Config(project_root=root)
    feature_info = joblib.load(config.models_dir / "feature_info.pkl")
    preprocessor = joblib.load(config.models_dir / "preprocessor.pkl")

    default_row = pd.read_csv(config.train_csv, nrows=1)
    default_row = default_row.drop(columns=[config.target_column, config.binary_label_column], errors="ignore")
    defaults = default_row.iloc[0].to_dict() if not default_row.empty else {}

    categories = {}
    categorical_cols = feature_info.get("categorical_cols", [])
    try:
        encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        for column, values in zip(categorical_cols, encoder.categories_):
            categories[column] = [str(value) for value in values.tolist()]
    except Exception:
        categories = {column: [] for column in categorical_cols}

    return {
        "expected_columns": list(getattr(preprocessor, "feature_names_in_", [])),
        "feature_info": feature_info,
        "defaults": defaults,
        "categories": categories,
    }


def build_empty_record(feature_bundle: dict[str, Any]) -> dict[str, Any]:
    record = {}
    defaults = feature_bundle.get("defaults", {})
    categories = feature_bundle.get("categories", {})
    numeric_cols = set(feature_bundle.get("feature_info", {}).get("numeric_cols", []))
    for column in feature_bundle.get("expected_columns", []):
        if column in defaults and pd.notna(defaults[column]):
            record[column] = defaults[column]
        elif column in categories and categories[column]:
            record[column] = categories[column][0]
        elif column in numeric_cols:
            record[column] = 0.0
        else:
            record[column] = ""
    return record


def ensure_frame_columns(df: pd.DataFrame, feature_bundle: dict[str, Any]) -> pd.DataFrame:
    frame = df.copy()
    for column, value in build_empty_record(feature_bundle).items():
        if column not in frame.columns:
            frame[column] = value
    ordered = feature_bundle.get("expected_columns", [])
    if ordered:
        frame = frame[ordered]
    return frame


def prediction_rows_to_frame(predictions: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for index, item in enumerate(predictions):
        row = {
            "row_index": index,
            "predicted_label": item.get("predicted_label"),
            "predicted_index": item.get("predicted_index"),
            "confidence": item.get("confidence"),
        }
        probabilities = item.get("probabilities", {})
        for class_name, probability in probabilities.items():
            row[f"prob_{class_name}"] = probability
        rows.append(row)
    return pd.DataFrame(rows)


def list_existing_plots(project_root: str | Path | None = None) -> list[Path]:
    root = Path(project_root) if project_root is not None else Config().project_root
    config = Config(project_root=root)
    if not config.plots_dir.exists():
        return []
    return sorted(config.plots_dir.glob("*.png"))
