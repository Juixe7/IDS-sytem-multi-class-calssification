from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = normalized.columns.str.strip()
    return normalized


def load_datasets(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = normalize_columns(pd.read_csv(config.train_csv))
    test_df = normalize_columns(pd.read_csv(config.test_csv))
    return train_df, test_df


def cache_parquet(df: pd.DataFrame, path: Path) -> bool:
    try:
        df.to_parquet(path, index=False)
        return True
    except Exception:
        return False


def validate_schema(train_df: pd.DataFrame, test_df: pd.DataFrame, config: Config) -> dict:
    train_cols = list(train_df.columns)
    test_cols = list(test_df.columns)
    schema_match = train_cols == test_cols
    target_present = config.target_column in train_df.columns and config.target_column in test_df.columns
    binary_label_present = (
        config.binary_label_column in train_df.columns and config.binary_label_column in test_df.columns
    )
    return {
        "schema_match": schema_match,
        "target_present": target_present,
        "binary_label_present": binary_label_present,
        "train_rows": int(train_df.shape[0]),
        "test_rows": int(test_df.shape[0]),
        "columns": train_cols,
    }
