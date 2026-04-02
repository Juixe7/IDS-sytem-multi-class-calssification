from __future__ import annotations

import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.config import Config


@dataclass
class PreparedData:
    X_train_selected: np.ndarray
    X_test_selected: np.ndarray
    X_train_scaled: np.ndarray
    X_test_scaled: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    label_encoder: LabelEncoder
    preprocessor: ColumnTransformer
    selector: SelectKBest
    scaler: StandardScaler
    selected_feature_names: list[str]
    dropped_correlated_features: list[str]
    interaction_feature_names: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    class_names: list[str]


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def clean_dataframe(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, dict]:
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip()
    if target_col in cleaned.columns:
        cleaned[target_col] = cleaned[target_col].astype(str).str.strip().str.lower()
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    id_removed = False
    if "id" in cleaned.columns:
        cleaned = cleaned.drop(columns=["id"])
        id_removed = True
    before_rows = int(cleaned.shape[0])
    duplicates = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates()
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in cleaned.columns if col not in numeric_cols]
    for col in categorical_cols:
        cleaned[col] = cleaned[col].astype(str).str.strip()
    return cleaned, {
        "before_rows": before_rows,
        "after_rows": int(cleaned.shape[0]),
        "id_removed": id_removed,
        "duplicates_removed": duplicates,
        "missing_by_column": cleaned.isna().sum().to_dict(),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }


def _drop_high_correlation_features(df: pd.DataFrame, numeric_cols: list[str], threshold: float) -> list[str]:
    if not numeric_cols:
        return []
    corr = df[numeric_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return [column for column in upper.columns if any(upper[column] > threshold)]


def _contains_id_feature(feature_names: list[str]) -> bool:
    return any(name == "id" or name.endswith("__id") for name in feature_names)


def add_interaction_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    engineered = df.copy()
    new_feature_names: list[str] = []

    if "sload" in engineered.columns and "dload" in engineered.columns:
        engineered["sload_dload_ratio"] = engineered["sload"] / (engineered["dload"] + 1e-6)
        new_feature_names.append("sload_dload_ratio")

    if "ct_src_ltm" in engineered.columns and "ct_dst_ltm" in engineered.columns:
        engineered["src_dst_ltm_ratio"] = engineered["ct_src_ltm"] / (engineered["ct_dst_ltm"] + 1e-6)
        new_feature_names.append("src_dst_ltm_ratio")

    if "sttl" in engineered.columns and "dttl" in engineered.columns:
        engineered["ttl_diff"] = engineered["sttl"] - engineered["dttl"]
        new_feature_names.append("ttl_diff")

    return engineered, new_feature_names


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame, config: Config) -> tuple[PreparedData, dict]:
    stage_start = time.perf_counter()
    print("[START] preprocess.clean_and_prepare", flush=True)
    train_clean, train_report = clean_dataframe(train_df, config.target_column)
    test_clean, test_report = clean_dataframe(test_df, config.target_column)

    y_train_raw = train_clean[config.target_column].copy()
    y_test_raw = test_clean[config.target_column].copy()

    X_train = train_clean.drop(columns=[config.target_column, config.binary_label_column], errors="ignore")
    X_test = test_clean.drop(columns=[config.target_column, config.binary_label_column], errors="ignore")
    X_train, interaction_feature_names = add_interaction_features(X_train)
    X_test, _ = add_interaction_features(X_test)
    print(f"[INFO] Interaction features added: {interaction_feature_names}", flush=True)

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X_train.columns if col not in numeric_cols]
    dropped_corr = _drop_high_correlation_features(X_train, numeric_cols, config.correlation_threshold)
    X_train = X_train.drop(columns=dropped_corr, errors="ignore")
    X_test = X_test.drop(columns=dropped_corr, errors="ignore")

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X_train.columns if col not in numeric_cols]

    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _make_one_hot_encoder()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
    transform_start = time.perf_counter()
    print("[INFO] Fitting train-only preprocessing pipeline", flush=True)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    X_train_processed = np.asarray(X_train_processed, dtype=np.float32)
    X_test_processed = np.asarray(X_test_processed, dtype=np.float32)
    print(f"[INFO] Preprocessing transform complete in {time.perf_counter() - transform_start:.2f}s", flush=True)

    feature_names = preprocessor.get_feature_names_out().tolist()
    transformed_feature_count = X_train_processed.shape[1]
    print(f"[INFO] Features before selection: {transformed_feature_count}", flush=True)
    print(f"[INFO] Feature cap set to: {config.feature_cap}", flush=True)
    print(f"[INFO] Config feature cap: {config.feature_cap}", flush=True)
    print(f"[INFO] Raw feature columns contain id: {'id' in X_train.columns}", flush=True)
    print(f"[INFO] Transformed feature names contain id: {_contains_id_feature(feature_names)}", flush=True)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    k = min(config.feature_cap, transformed_feature_count)
    selector = SelectKBest(score_func=f_classif, k=k)
    select_start = time.perf_counter()
    X_train_selected = selector.fit_transform(X_train_processed, y_train)
    X_test_selected = selector.transform(X_test_processed)
    X_train_selected = np.asarray(X_train_selected, dtype=np.float32)
    X_test_selected = np.asarray(X_test_selected, dtype=np.float32)
    print(f"[INFO] Feature selection complete in {time.perf_counter() - select_start:.2f}s", flush=True)
    print(f"[INFO] Features after selection: {X_train_selected.shape[1]}", flush=True)
    selected_feature_names = [name for name, keep in zip(feature_names, selector.get_support()) if keep]
    print(f"[INFO] Selected feature names contain id: {_contains_id_feature(selected_feature_names)}", flush=True)
    print(f"[INFO] Train duplicates removed after id drop: {train_report['duplicates_removed']}", flush=True)
    print(f"[INFO] Test duplicates removed after id drop: {test_report['duplicates_removed']}", flush=True)

    scale_start = time.perf_counter()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    X_train_scaled = np.asarray(X_train_scaled, dtype=np.float32)
    X_test_scaled = np.asarray(X_test_scaled, dtype=np.float32)
    print(f"[INFO] Scaling complete in {time.perf_counter() - scale_start:.2f}s", flush=True)

    prepared = PreparedData(
        X_train_selected=X_train_selected,
        X_test_selected=X_test_selected,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        label_encoder=label_encoder,
        preprocessor=preprocessor,
        selector=selector,
        scaler=scaler,
        selected_feature_names=selected_feature_names,
        dropped_correlated_features=dropped_corr,
        interaction_feature_names=interaction_feature_names,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        class_names=label_encoder.classes_.tolist(),
    )
    report = {
        "train_cleaning": train_report,
        "test_cleaning": test_report,
        "dropped_correlated_features": dropped_corr,
        "interaction_feature_names": interaction_feature_names,
        "selected_feature_names": selected_feature_names,
        "class_mapping": {cls: int(idx) for idx, cls in enumerate(label_encoder.classes_)},
        "features_before_selection": int(transformed_feature_count),
        "features_after_selection": int(X_train_selected.shape[1]),
        "final_train_shape": list(X_train_selected.shape),
        "final_test_shape": list(X_test_selected.shape),
    }
    print(f"[END] preprocess.clean_and_prepare: {time.perf_counter() - stage_start:.2f}s", flush=True)
    return prepared, report


def save_preprocessing_artifacts(prepared: PreparedData, config: Config) -> None:
    joblib.dump(prepared.label_encoder, config.models_dir / "label_encoder.pkl")
    joblib.dump(prepared.preprocessor, config.models_dir / "preprocessor.pkl")
    joblib.dump(prepared.selector, config.models_dir / "selector.pkl")
    joblib.dump(prepared.scaler, config.models_dir / "scaler.pkl")
    joblib.dump(
        {
            "selected_feature_names": prepared.selected_feature_names,
            "dropped_correlated_features": prepared.dropped_correlated_features,
            "interaction_feature_names": prepared.interaction_feature_names,
            "numeric_cols": prepared.numeric_cols,
            "categorical_cols": prepared.categorical_cols,
            "class_names": prepared.class_names,
            "class_thresholds": {},
            "threshold_tuning": {},
        },
        config.models_dir / "feature_info.pkl",
    )
