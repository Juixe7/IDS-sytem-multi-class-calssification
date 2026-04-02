from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / ".mplconfig"))

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.calibration import CalibratedClassifierCV

from src.config import Config

matplotlib.use("Agg")


def stratified_sample_indices(y: np.ndarray, max_samples: int, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    unique_classes = np.unique(y)
    per_class = max(1, max_samples // len(unique_classes))
    indices: list[int] = []
    for cls in unique_classes:
        class_indices = np.where(y == cls)[0]
        take = min(len(class_indices), per_class)
        chosen = rng.choice(class_indices, size=take, replace=False)
        indices.extend(chosen.tolist())
    return np.array(sorted(indices))


def _resolve_explainable_estimator(estimator: object) -> object:
    if isinstance(estimator, CalibratedClassifierCV):
        calibrated_classifiers = getattr(estimator, "calibrated_classifiers_", [])
        if calibrated_classifiers:
            return getattr(calibrated_classifiers[0], "estimator", estimator)
    return estimator


def run_shap_analysis(
    estimator: object,
    model_name: str,
    X_reference: np.ndarray,
    y_reference: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    config: Config,
) -> dict:
    effective_limit = min(config.shap_sample_size, 100)
    sample_idx = stratified_sample_indices(y_reference, effective_limit, config.random_state)
    X_sample = np.asarray(X_reference[sample_idx], dtype=np.float32)
    y_sample = y_reference[sample_idx]
    print(f"[START] shap.{model_name} on {len(sample_idx)} samples", flush=True)

    explainable_estimator = _resolve_explainable_estimator(estimator)
    explainer = shap.TreeExplainer(explainable_estimator)
    raw_shap = explainer.shap_values(X_sample, check_additivity=False)

    if isinstance(raw_shap, list):
        shap_by_class = [np.asarray(values, dtype=np.float32) for values in raw_shap]
    elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
        shap_by_class = [np.asarray(raw_shap[:, :, idx], dtype=np.float32) for idx in range(raw_shap.shape[2])]
    else:
        shap_by_class = [np.asarray(raw_shap, dtype=np.float32)]

    aggregated_shap = np.mean(np.abs(np.stack(shap_by_class, axis=0)), axis=0)
    print("[INFO] SHAP values computed; generating plots", flush=True)

    plt.figure()
    shap.summary_plot(aggregated_shap, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(config.plots_dir / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(aggregated_shap, X_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(config.plots_dir / "shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    predicted = estimator.predict(X_sample)
    example_index = int(np.argmax(predicted == y_sample)) if np.any(predicted == y_sample) else 0
    predicted_class = int(predicted[example_index]) if np.ndim(predicted) else 0
    class_for_waterfall = min(predicted_class, len(shap_by_class) - 1)
    expected_value = np.array(getattr(explainer, "expected_value", []))
    base_value = (
        float(expected_value[class_for_waterfall]) if expected_value.size > 1 else float(expected_value.ravel()[0])
        if expected_value.size
        else 0.0
    )
    waterfall_explanation = shap.Explanation(
        values=shap_by_class[class_for_waterfall][example_index],
        base_values=base_value,
        data=X_sample[example_index],
        feature_names=feature_names,
    )
    plt.figure()
    shap.plots.waterfall(waterfall_explanation, max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(config.plots_dir / "shap_waterfall_example.png", dpi=300, bbox_inches="tight")
    plt.close()

    payload = {
        "model_name": model_name,
        "sample_indices": sample_idx.tolist(),
        "feature_names": feature_names,
        "class_names": class_names,
        "expected_value": expected_value.tolist(),
        "effective_shap_sample_size": int(len(sample_idx)),
        "waterfall_class_index": class_for_waterfall,
    }
    joblib.dump(
        {"payload": payload, "aggregated_shap": aggregated_shap, "shap_by_class": shap_by_class},
        config.outputs_dir / "shap_values.pkl",
    )
    print(f"[END] shap.{model_name}", flush=True)
    return payload
