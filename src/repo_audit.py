from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.config import Config
from src.utils.logging_utils import save_json, write_text

matplotlib.use("Agg")


def _fit_repo_rf(X_train, X_test, y_train, y_test) -> dict:
    model = RandomForestClassifier(n_estimators=100, random_state=50, n_jobs=1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted")),
    }


def run_repo_benchmark_audit(config: Config) -> dict | None:
    repo_multi_path = config.repo_multi_data_path
    if not repo_multi_path.exists():
        return None

    print("[START] repo_benchmark_audit", flush=True)
    df = pd.read_csv(repo_multi_path)
    unnamed_cols = [column for column in df.columns if column.lower().startswith("unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    X = df.drop(columns=["label"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=100,
        stratify=y,
    )

    leakage_columns = [column for column in X.columns if column.startswith("attack_cat_")]
    leaked_result = _fit_repo_rf(X_train, X_test, y_train, y_test)
    safe_result = _fit_repo_rf(
        X_train.drop(columns=leakage_columns),
        X_test.drop(columns=leakage_columns),
        y_train,
        y_test,
    )

    payload = {
        "repo_dataset_path": str(repo_multi_path),
        "rows": int(df.shape[0]),
        "feature_count_with_leakage": int(X.shape[1]),
        "feature_count_without_leakage": int(X.shape[1] - len(leakage_columns)),
        "leakage_columns": leakage_columns,
        "with_leakage": leaked_result,
        "without_attack_cat_dummies": safe_result,
        "accuracy_gap": float(leaked_result["accuracy"] - safe_result["accuracy"]),
        "macro_f1_gap": float(leaked_result["macro_f1"] - safe_result["macro_f1"]),
        "interpretation": {
            "headline": "The repo benchmark is inflated by target-derived attack_cat dummy columns being present in the feature matrix.",
            "production_note": "Those attack_cat_* columns are not available at inference time, so they must not be used in a deployable SaaS model.",
        },
    }
    save_json(config.repo_audit_json_path, payload)

    report_md = "\n".join(
        [
            "# Repo Benchmark Audit",
            "",
            "## What Was Reproduced",
            "",
            f"- Source file: `{repo_multi_path}`",
            "- Model: RandomForestClassifier(n_estimators=100, random_state=50, n_jobs=1)",
            "- Split: train_test_split(test_size=0.30, random_state=100, stratify=label)",
            "",
            "## Findings",
            "",
            f"- Accuracy with target-derived `attack_cat_*` features left in: **{leaked_result['accuracy'] * 100:.2f}%**",
            f"- Macro F1 with target-derived `attack_cat_*` features left in: **{leaked_result['macro_f1']:.4f}**",
            f"- Accuracy after removing those leaked target-derived features: **{safe_result['accuracy'] * 100:.2f}%**",
            f"- Macro F1 after removing those leaked target-derived features: **{safe_result['macro_f1']:.4f}**",
            "",
            "## Interpretation",
            "",
            "- The public 97% benchmark is reproducible on the repo-prepared dataset.",
            "- A large part of that score comes from `attack_cat_*` columns that are deterministic functions of the label being predicted.",
            "- Those columns are valid only after the label is already known, so they are not legitimate raw-input features for a SaaS inference pipeline.",
            "- The production pipeline should keep using non-leaky raw network features only.",
        ]
    )
    write_text(config.repo_audit_md_path, report_md)

    comparison_df = pd.DataFrame(
        [
            {"scenario": "Repo style\n(with leakage)", "accuracy": leaked_result["accuracy"], "macro_f1": leaked_result["macro_f1"]},
            {
                "scenario": "Repo style\n(without attack_cat dummies)",
                "accuracy": safe_result["accuracy"],
                "macro_f1": safe_result["macro_f1"],
            },
        ]
    ).melt(id_vars="scenario", var_name="metric", value_name="value")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=comparison_df, x="scenario", y="value", hue="metric", ax=ax)
    ax.set_title("Repo Benchmark Audit: Leakage vs Safe Features")
    ax.set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(config.repo_audit_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[END] repo_benchmark_audit", flush=True)
    return payload
