from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / ".mplconfig"))

import joblib
import matplotlib
import numpy as np
import pandas as pd
import json
import shap
from sklearn.calibration import CalibratedClassifierCV

from src.config import Config

matplotlib.use("Agg")


@dataclass
class ProductionArtifacts:
    model_name: str
    final_model: object
    label_encoder: object
    preprocessor: object
    selector: object
    scaler: object
    feature_info: dict[str, Any]


def _thresholded_predictions(probabilities: np.ndarray, thresholds_by_name: dict[str, float], class_names: list[str]) -> np.ndarray:
    if not thresholds_by_name:
        return np.argmax(probabilities, axis=1).astype(int)

    thresholds_by_index = {
        class_names.index(class_name): float(threshold)
        for class_name, threshold in thresholds_by_name.items()
        if class_name in class_names
    }
    predictions = np.argmax(probabilities, axis=1).astype(int)
    threshold_items = list(thresholds_by_index.items())
    for row_idx in range(probabilities.shape[0]):
        eligible: list[tuple[float, float, int]] = []
        for class_index, threshold in threshold_items:
            probability = float(probabilities[row_idx, class_index])
            if probability >= threshold:
                eligible.append((probability - threshold, probability, class_index))
        if eligible:
            eligible.sort(reverse=True)
            predictions[row_idx] = int(eligible[0][2])
    return predictions


def _resolve_explainable_estimator(final_model: object) -> object:
    if isinstance(final_model, CalibratedClassifierCV):
        calibrated_classifiers = getattr(final_model, "calibrated_classifiers_", [])
        if calibrated_classifiers:
            return getattr(calibrated_classifiers[0], "estimator", final_model)
    return final_model


def _to_dataframe(records: pd.DataFrame | dict | list[dict]) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records.copy()
    if isinstance(records, dict):
        return pd.DataFrame([records])
    return pd.DataFrame(records)


def load_production_artifacts(project_root: str | Path | None = None) -> ProductionArtifacts:
    root = Path(project_root) if project_root is not None else Config().project_root
    config = Config(project_root=root)
    manifest = None
    if config.final_model_manifest_path.exists():
        manifest = json.loads(config.final_model_manifest_path.read_text(encoding="utf-8"))
    model_name = manifest["final_model"] if manifest else config.default_production_model

    return ProductionArtifacts(
        model_name=model_name,
        final_model=joblib.load(config.final_model_path),
        label_encoder=joblib.load(config.models_dir / "label_encoder.pkl"),
        preprocessor=joblib.load(config.models_dir / "preprocessor.pkl"),
        selector=joblib.load(config.models_dir / "selector.pkl"),
        scaler=joblib.load(config.models_dir / "scaler.pkl"),
        feature_info=joblib.load(config.models_dir / "feature_info.pkl"),
    )


def _prepare_features(records: pd.DataFrame | dict | list[dict], artifacts: ProductionArtifacts) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    frame = _to_dataframe(records)
    expected_columns = list(getattr(artifacts.preprocessor, "feature_names_in_", []))
    for column in expected_columns:
        if column not in frame.columns:
            frame[column] = np.nan
    if expected_columns:
        frame = frame[expected_columns]

    processed = artifacts.preprocessor.transform(frame)
    processed = np.asarray(processed, dtype=np.float32)
    selected = artifacts.selector.transform(processed)
    selected = np.asarray(selected, dtype=np.float32)
    scaled = artifacts.scaler.transform(selected)
    scaled = np.asarray(scaled, dtype=np.float32)
    return selected, scaled, frame


def predict_records(records: pd.DataFrame | dict | list[dict], artifacts: ProductionArtifacts | None = None) -> list[dict[str, Any]]:
    artifacts = artifacts or load_production_artifacts()
    selected, scaled, _ = _prepare_features(records, artifacts)
    X_infer = scaled if artifacts.model_name == "lr" else selected

    probabilities = artifacts.final_model.predict_proba(X_infer) if hasattr(artifacts.final_model, "predict_proba") else None
    class_thresholds = artifacts.feature_info.get("class_thresholds", {})
    if probabilities is not None:
        predicted_indices = _thresholded_predictions(
            probabilities,
            class_thresholds,
            list(artifacts.label_encoder.classes_),
        )
    else:
        predicted_indices = artifacts.final_model.predict(X_infer)
    predicted_labels = artifacts.label_encoder.inverse_transform(predicted_indices)

    results = []
    for row_idx, label in enumerate(predicted_labels):
        item = {
            "predicted_label": str(label),
            "predicted_index": int(predicted_indices[row_idx]),
        }
        if probabilities is not None:
            item["confidence"] = float(np.max(probabilities[row_idx]))
            item["probabilities"] = {
                class_name: float(probabilities[row_idx][class_idx])
                for class_idx, class_name in enumerate(artifacts.label_encoder.classes_)
            }
        results.append(item)
    return results


def explain_records(
    records: pd.DataFrame | dict | list[dict],
    artifacts: ProductionArtifacts | None = None,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    artifacts = artifacts or load_production_artifacts()
    selected, scaled, _ = _prepare_features(records, artifacts)
    X_infer = scaled if artifacts.model_name == "lr" else selected
    feature_names = artifacts.feature_info.get("selected_feature_names", [])
    explainable_estimator = _resolve_explainable_estimator(artifacts.final_model)

    if artifacts.model_name in {"rf", "xgb", "lgbm"}:
        explainer = shap.TreeExplainer(explainable_estimator)
        raw_shap = explainer.shap_values(X_infer, check_additivity=False)
        if isinstance(raw_shap, list):
            shap_by_class = [np.asarray(values, dtype=np.float32) for values in raw_shap]
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
            shap_by_class = [np.asarray(raw_shap[:, :, idx], dtype=np.float32) for idx in range(raw_shap.shape[2])]
        else:
            shap_by_class = [np.asarray(raw_shap, dtype=np.float32)]
        expected_value = np.array(getattr(explainer, "expected_value", []))
    else:
        explainer = shap.Explainer(explainable_estimator, X_infer)
        explanation = explainer(X_infer)
        if explanation.values.ndim == 3:
            shap_by_class = [np.asarray(explanation.values[:, :, idx], dtype=np.float32) for idx in range(explanation.values.shape[2])]
        else:
            shap_by_class = [np.asarray(explanation.values, dtype=np.float32)]
        expected_value = np.array(explanation.base_values)

    predictions = predict_records(records, artifacts)
    explanations: list[dict[str, Any]] = []
    for row_idx, prediction in enumerate(predictions):
        class_index = prediction["predicted_index"]
        class_slot = min(class_index, len(shap_by_class) - 1)
        row_values = shap_by_class[class_slot][row_idx]
        ranked = np.argsort(np.abs(row_values))[::-1][:top_n]
        top_features = [
            {
                "feature": feature_names[idx] if idx < len(feature_names) else str(idx),
                "shap_value": float(row_values[idx]),
                "feature_value": float(X_infer[row_idx][idx]),
            }
            for idx in ranked
        ]
        explanations.append(
            {
                "prediction": prediction,
                "expected_value": (
                    float(expected_value[class_slot]) if expected_value.ndim else float(expected_value)
                )
                if expected_value.size
                else 0.0,
                "top_features": top_features,
            }
        )
    return explanations
