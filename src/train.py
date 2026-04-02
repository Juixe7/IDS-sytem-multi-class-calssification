from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import product

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from lightgbm import early_stopping as lgb_early_stopping
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from src.config import Config


@dataclass
class TrainedModel:
    name: str
    estimator: object
    cv_scores: list[dict]
    train_seconds: float
    metadata: dict | None = None


def get_feature_matrix(model_name: str, X_selected: np.ndarray, X_scaled: np.ndarray) -> np.ndarray:
    return X_scaled if model_name == "lr" else X_selected


def build_estimators(class_count: int, config: Config) -> dict[str, object]:
    xgb_params = dict(config.xgb_params)
    xgb_params["num_class"] = class_count
    return {
        "lr": LogisticRegression(**config.logistic_params),
        "rf": RandomForestClassifier(**config.rf_params),
        "xgb": XGBClassifier(**xgb_params),
        "lgbm": LGBMClassifier(**config.lgbm_params),
    }


def _class_distribution(y: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(y, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def _build_controlled_smote_strategy(y_train: np.ndarray, ratio: float) -> dict[int, int]:
    distribution = _class_distribution(y_train)
    if not distribution:
        return {}
    majority_count = max(distribution.values())
    target_count = max(1, int(np.ceil(majority_count * ratio)))
    strategy = {
        cls: target_count
        for cls, count in distribution.items()
        if count < target_count
    }
    return strategy


def _apply_thresholds(probabilities: np.ndarray, thresholds: dict[int, float] | None) -> np.ndarray:
    if not thresholds:
        return np.argmax(probabilities, axis=1).astype(int)

    base_predictions = np.argmax(probabilities, axis=1).astype(int)
    final_predictions = base_predictions.copy()
    threshold_items = list(thresholds.items())
    for row_idx in range(probabilities.shape[0]):
        eligible: list[tuple[float, float, int]] = []
        for class_index, threshold in threshold_items:
            probability = float(probabilities[row_idx, class_index])
            if probability >= threshold:
                eligible.append((probability - threshold, probability, class_index))
        if eligible:
            eligible.sort(reverse=True)
            final_predictions[row_idx] = int(eligible[0][2])
    return final_predictions


def _apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Config,
    log_prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    print(f"[INFO] {log_prefix}.before_smote: X={X_train.shape}, y={y_train.shape}", flush=True)
    print(f"[INFO] {log_prefix}.class_distribution_before={_class_distribution(y_train)}", flush=True)
    smote_ratio = 0.3
    smote_strategy = _build_controlled_smote_strategy(y_train, smote_ratio)
    if not smote_strategy:
        print(f"[INFO] {log_prefix}.smote_strategy=0.3 (controlled) - no resampling needed", flush=True)
        return np.asarray(X_train, dtype=np.float32), np.asarray(y_train, dtype=y_train.dtype)
    print(f"[INFO] {log_prefix}.smote_strategy=0.3 (controlled)", flush=True)
    print(f"[INFO] {log_prefix}.smote_targets={smote_strategy}", flush=True)
    smote = SMOTE(
        sampling_strategy=smote_strategy,
        random_state=config.random_state,
        k_neighbors=3,
    )
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    X_resampled = np.asarray(X_resampled, dtype=np.float32)
    y_resampled = np.asarray(y_resampled, dtype=y_train.dtype)
    print(f"[INFO] {log_prefix}.after_smote: X={X_resampled.shape}, y={y_resampled.shape}", flush=True)
    print(f"[INFO] {log_prefix}.class_distribution_after={_class_distribution(y_resampled)}", flush=True)
    return X_resampled, y_resampled


def _compute_xgb_sample_weight(y_train: np.ndarray, log_prefix: str) -> np.ndarray:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_map = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
    sample_weight = np.asarray([class_weight_map[int(label)] for label in y_train], dtype=np.float32)
    print(f"[INFO] {log_prefix}.xgb_class_weight_map={class_weight_map}", flush=True)
    print(f"[INFO] {log_prefix}.xgb_sample_weight_applied=True", flush=True)
    return sample_weight


def _threshold_anchor_grid(class_names: list[str]) -> tuple[dict[str, int], list[tuple[dict[int, float], dict[str, float]]]]:
    anchors = {
        "fuzzers": [0.15, 0.20, 0.25, 0.30],
        "backdoor": [0.15, 0.20, 0.25, 0.30],
        "analysis": [0.20, 0.25, 0.30, 0.35],
        "worms": [0.10, 0.15, 0.20, 0.25],
    }
    class_index_by_name = {name: idx for idx, name in enumerate(class_names)}
    available_names = {name: class_index_by_name[name] for name in anchors if name in class_index_by_name}
    threshold_options: list[tuple[dict[int, float], dict[str, float]]] = []
    ordered_names = list(available_names.keys())
    if not ordered_names:
        return {}, threshold_options
    for candidate_values in product(*(anchors[name] for name in ordered_names)):
        by_index = {
            int(available_names[name]): float(value)
            for name, value in zip(ordered_names, candidate_values)
        }
        by_name = {
            name: float(value)
            for name, value in zip(ordered_names, candidate_values)
        }
        threshold_options.append((by_index, by_name))
    return available_names, threshold_options


def _tune_class_thresholds(
    estimator: object,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    class_names: list[str],
    log_prefix: str,
) -> dict:
    if not hasattr(estimator, "predict_proba"):
        return {
            "thresholds_by_index": {},
            "thresholds_by_name": {},
            "validation_macro_f1": None,
        }

    _, threshold_options = _threshold_anchor_grid(class_names)
    if not threshold_options:
        return {
            "thresholds_by_index": {},
            "thresholds_by_name": {},
            "validation_macro_f1": None,
        }

    probabilities = estimator.predict_proba(X_validation)
    best_macro_f1 = -np.inf
    best_thresholds_by_index: dict[int, float] = {}
    best_thresholds_by_name: dict[str, float] = {}
    for thresholds_by_index, thresholds_by_name in threshold_options:
        preds = _apply_thresholds(probabilities, thresholds_by_index)
        macro_f1 = float(f1_score(y_validation, preds, average="macro"))
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_thresholds_by_index = thresholds_by_index
            best_thresholds_by_name = thresholds_by_name

    print(f"[INFO] {log_prefix}.threshold_validation_macro_f1={best_macro_f1:.4f}", flush=True)
    print(f"[INFO] {log_prefix}.thresholds_by_name={best_thresholds_by_name}", flush=True)
    return {
        "thresholds_by_index": best_thresholds_by_index,
        "thresholds_by_name": best_thresholds_by_name,
        "validation_macro_f1": best_macro_f1,
    }


def _fit_estimator(
    model_name: str,
    estimator: object,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Config,
    log_prefix: str,
    enable_early_stopping: bool = True,
) -> object:
    if model_name in {"xgb", "lgbm"} and enable_early_stopping:
        X_inner_train, X_inner_eval, y_inner_train, y_inner_eval = train_test_split(
            X_train,
            y_train,
            test_size=0.1,
            stratify=y_train,
            random_state=config.random_state,
        )
        X_inner_resampled, y_inner_resampled = _apply_smote(
            X_inner_train,
            y_inner_train,
            config,
            f"{log_prefix}.inner_train",
        )
        if model_name == "xgb":
            sample_weight = _compute_xgb_sample_weight(y_inner_resampled, f"{log_prefix}.inner_train")
            estimator.fit(
                X_inner_resampled,
                y_inner_resampled,
                sample_weight=sample_weight,
                eval_set=[(X_inner_eval, y_inner_eval)],
                verbose=False,
                early_stopping_rounds=20,
            )
        else:
            estimator.fit(
                X_inner_resampled,
                y_inner_resampled,
                eval_set=[(X_inner_eval, y_inner_eval)],
                callbacks=[lgb_early_stopping(20, verbose=False)],
            )
        return estimator

    X_resampled, y_resampled = _apply_smote(X_train, y_train, config, log_prefix)
    if model_name == "xgb":
        sample_weight = _compute_xgb_sample_weight(y_resampled, log_prefix)
        estimator.fit(X_resampled, y_resampled, sample_weight=sample_weight, verbose=False)
    else:
        estimator.fit(X_resampled, y_resampled)
    return estimator


def _manual_cv(model_name: str, estimator: object, X: np.ndarray, y: np.ndarray, config: Config) -> list[dict]:
    skf = StratifiedKFold(n_splits=config.cv_splits, shuffle=True, random_state=config.random_state)
    scores: list[dict] = []
    print(f"[START] cv.{model_name}", flush=True)
    for fold_index, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        fold_start = time.perf_counter()
        X_fold_train, X_fold_valid = X[train_idx], X[valid_idx]
        y_fold_train, y_fold_valid = y[train_idx], y[valid_idx]
        fold_model = clone(estimator)
        fold_model = _fit_estimator(
            model_name,
            fold_model,
            X_fold_train,
            y_fold_train,
            config,
            f"cv.{model_name}.fold_{fold_index}",
        )
        preds = fold_model.predict(X_fold_valid)
        scores.append(
            {
                "fold": fold_index,
                "accuracy": float(accuracy_score(y_fold_valid, preds)),
                "macro_f1": float(f1_score(y_fold_valid, preds, average="macro")),
            }
        )
        print(
            f"[INFO] cv.{model_name}.fold_{fold_index}: {time.perf_counter() - fold_start:.2f}s "
            f"(acc={scores[-1]['accuracy']:.4f}, macro_f1={scores[-1]['macro_f1']:.4f})",
            flush=True,
        )
    print(f"[END] cv.{model_name}", flush=True)
    return scores


def train_calibrated_final_model(
    model_name: str,
    X_train_selected: np.ndarray,
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    class_count: int,
    class_names: list[str],
    config: Config,
    save_path: str | None = None,
) -> TrainedModel:
    base_estimator = build_estimators(class_count, config)[model_name]
    features = get_feature_matrix(model_name, X_train_selected, X_train_scaled)
    print(f"[START] train.{model_name}.calibrated_final", flush=True)
    X_train_core, X_threshold_val, y_train_core, y_threshold_val = train_test_split(
        features,
        y_train,
        test_size=0.1,
        stratify=y_train,
        random_state=config.random_state,
    )
    X_resampled, y_resampled = _apply_smote(
        X_train_core,
        y_train_core,
        config,
        f"train.{model_name}.calibrated_final",
    )
    calibrated_model = CalibratedClassifierCV(
        estimator=base_estimator,
        method="isotonic",
        cv=config.cv_splits,
    )
    start = time.perf_counter()
    fit_kwargs: dict[str, np.ndarray] = {}
    if model_name == "xgb":
        fit_kwargs["sample_weight"] = _compute_xgb_sample_weight(
            y_resampled,
            f"train.{model_name}.calibrated_final",
        )
    calibrated_model.fit(X_resampled, y_resampled, **fit_kwargs)
    threshold_metadata = _tune_class_thresholds(
        calibrated_model,
        X_threshold_val,
        y_threshold_val,
        class_names,
        f"train.{model_name}.calibrated_final",
    )
    train_seconds = time.perf_counter() - start
    if save_path is not None:
        joblib.dump(calibrated_model, save_path)
    print(
        f"[INFO] train.{model_name}.calibrated_final.calibration_used=True",
        flush=True,
    )
    print(f"[END] train.{model_name}.calibrated_final: final_fit={train_seconds:.2f}s", flush=True)
    return TrainedModel(
        name=model_name,
        estimator=calibrated_model,
        cv_scores=[],
        train_seconds=train_seconds,
        metadata=threshold_metadata,
    )


def train_models(
    X_train_selected: np.ndarray,
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    class_count: int,
    config: Config,
) -> dict[str, TrainedModel]:
    estimators = {
        name: (estimator, get_feature_matrix(name, X_train_selected, X_train_scaled))
        for name, estimator in build_estimators(class_count, config).items()
    }
    trained: dict[str, TrainedModel] = {}
    for name, (estimator, features) in estimators.items():
        print(f"[START] train.{name}", flush=True)
        cv_scores = _manual_cv(name, estimator, features, y_train, config)
        start = time.perf_counter()
        estimator = _fit_estimator(name, estimator, features, y_train, config, f"train.{name}")
        train_seconds = time.perf_counter() - start
        trained[name] = TrainedModel(name=name, estimator=estimator, cv_scores=cv_scores, train_seconds=train_seconds, metadata={})
        joblib.dump(estimator, config.models_dir / f"model_{name}.pkl")
        print(f"[END] train.{name}: final_fit={train_seconds:.2f}s", flush=True)
    return trained


def train_single_model(
    model_name: str,
    X_train_selected: np.ndarray,
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    class_count: int,
    config: Config,
    save_path: str | None = None,
    run_cv: bool = False,
) -> TrainedModel:
    estimator = build_estimators(class_count, config)[model_name]
    features = get_feature_matrix(model_name, X_train_selected, X_train_scaled)
    print(f"[START] train.{model_name}.production", flush=True)
    cv_scores = _manual_cv(model_name, estimator, features, y_train, config) if run_cv else []
    start = time.perf_counter()
    estimator = _fit_estimator(model_name, estimator, features, y_train, config, f"train.{model_name}.production")
    train_seconds = time.perf_counter() - start
    if save_path is not None:
        joblib.dump(estimator, save_path)
    print(f"[END] train.{model_name}.production: final_fit={train_seconds:.2f}s", flush=True)
    return TrainedModel(name=model_name, estimator=estimator, cv_scores=cv_scores, train_seconds=train_seconds, metadata={})
