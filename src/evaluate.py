from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from src.train import TrainedModel

matplotlib.use("Agg")


def _apply_thresholds(probabilities: np.ndarray, thresholds_by_index: dict[int, float] | None) -> np.ndarray:
    if not thresholds_by_index:
        return np.argmax(probabilities, axis=1).astype(int)

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


def _macro_far(cm: np.ndarray) -> tuple[float, dict[str, float]]:
    per_class = {}
    total = cm.sum()
    for idx in range(cm.shape[0]):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        tn = total - tp - fp - fn
        denom = fp + tn
        per_class[str(idx)] = float(fp / denom) if denom else 0.0
    return float(np.mean(list(per_class.values()))), per_class


def evaluate_models(
    trained_models: dict[str, TrainedModel],
    X_test_selected: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    plots_dir: Path,
) -> tuple[dict, str]:
    sns.set_theme(style="whitegrid")
    metrics_report: dict = {}
    markdown_parts = ["# Classification Reports", ""]
    roc_fig, roc_ax = plt.subplots(figsize=(10, 8))
    y_test_bin = label_binarize(y_test, classes=np.arange(len(class_names)))

    for model_name, trained in trained_models.items():
        X_eval = X_test_scaled if model_name == "lr" else X_test_selected
        estimator = trained.estimator
        preds = estimator.predict(X_eval)
        proba = estimator.predict_proba(X_eval) if hasattr(estimator, "predict_proba") else None
        cm = confusion_matrix(y_test, preds)
        macro_far, per_class_far = _macro_far(cm)
        report = classification_report(y_test, preds, target_names=class_names, output_dict=True, zero_division=0)
        roc_auc = None
        if proba is not None:
            try:
                roc_auc = float(roc_auc_score(y_test_bin, proba, multi_class="ovr", average="macro"))
                fpr, tpr, _ = roc_curve(y_test_bin.ravel(), proba.ravel())
                roc_ax.plot(fpr, tpr, label=f"{model_name.upper()} (AUC={roc_auc:.3f})")
            except Exception:
                roc_auc = None

        metrics_report[model_name] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "macro_f1": float(f1_score(y_test, preds, average="macro")),
            "weighted_f1": float(f1_score(y_test, preds, average="weighted")),
            "roc_auc_ovr": roc_auc,
            "macro_far": macro_far,
            "per_class_far": per_class_far,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "cv_scores": trained.cv_scores,
            "train_seconds": trained.train_seconds,
        }

        markdown_parts.extend(
            [
                f"## {model_name.upper()}",
                "",
                f"- Accuracy: {metrics_report[model_name]['accuracy']:.4f}",
                f"- Macro F1: {metrics_report[model_name]['macro_f1']:.4f}",
                f"- Weighted F1: {metrics_report[model_name]['weighted_f1']:.4f}",
                f"- Macro FAR: {metrics_report[model_name]['macro_far']:.4f}",
                f"- ROC-AUC OvR: {roc_auc:.4f}" if roc_auc is not None else "- ROC-AUC OvR: unavailable",
                "",
                "```",
                classification_report(y_test, preds, target_names=class_names, zero_division=0),
                "```",
                "",
            ]
        )

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=class_names, yticklabels=class_names)
        ax.set_title(f"Confusion Matrix - {model_name.upper()}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.tight_layout()
        fig.savefig(plots_dir / f"confusion_matrix_{model_name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    roc_ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    roc_ax.set_title("ROC Curve Comparison")
    roc_ax.set_xlabel("False Positive Rate")
    roc_ax.set_ylabel("True Positive Rate")
    roc_ax.legend()
    roc_fig.tight_layout()
    roc_fig.savefig(plots_dir / "roc_curve_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(roc_fig)

    comparison_df = pd.DataFrame(
        [
            {
                "model": name.upper(),
                "accuracy": values["accuracy"],
                "macro_f1": values["macro_f1"],
                "weighted_f1": values["weighted_f1"],
                "macro_far": values["macro_far"],
                "train_seconds": values["train_seconds"],
            }
            for name, values in metrics_report.items()
        ]
    )
    melted = comparison_df.melt(id_vars="model", var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=melted[melted["metric"] != "train_seconds"], x="metric", y="value", hue="model", ax=ax)
    ax.set_title("Model Metric Comparison")
    fig.tight_layout()
    fig.savefig(plots_dir / "model_metric_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=comparison_df, x="model", y="train_seconds", ax=ax, palette="rocket")
    ax.set_title("Training Time Comparison")
    ax.set_ylabel("Seconds")
    fig.tight_layout()
    fig.savefig(plots_dir / "training_time_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    per_class_f1 = pd.DataFrame(
        {
            model_name.upper(): {
                cls: values["classification_report"].get(cls, {}).get("f1-score", 0.0) for cls in class_names
            }
            for model_name, values in metrics_report.items()
        }
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(per_class_f1, annot=True, cmap="YlGnBu", ax=ax)
    ax.set_title("Per-Class F1 Heatmap")
    fig.tight_layout()
    fig.savefig(plots_dir / "per_class_f1_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return metrics_report, "\n".join(markdown_parts)


def select_best_model(metrics_report: dict) -> str:
    ranked = sorted(
        metrics_report.items(),
        key=lambda item: (item[1]["macro_f1"], item[1]["accuracy"], -item[1]["macro_far"]),
        reverse=True,
    )
    return ranked[0][0]


def evaluate_final_model(
    estimator: object,
    model_name: str,
    X_test_selected: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    thresholds_by_index: dict[int, float] | None = None,
) -> dict:
    X_eval = X_test_scaled if model_name == "lr" else X_test_selected
    if thresholds_by_index and hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(X_eval)
        preds = _apply_thresholds(probabilities, thresholds_by_index)
    else:
        preds = estimator.predict(X_eval)
    report = classification_report(y_test, preds, target_names=class_names, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro")),
        "weighted_f1": float(f1_score(y_test, preds, average="weighted")),
        "classification_report": report,
    }
