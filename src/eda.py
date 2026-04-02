from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")


sns.set_theme(style="whitegrid")


def save_label_distribution(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    train_counts = train_df[target_col].value_counts().sort_values(ascending=False)
    test_counts = test_df[target_col].value_counts().sort_values(ascending=False)
    sns.barplot(x=train_counts.values, y=train_counts.index, ax=axes[0], palette="Blues_r")
    axes[0].set_title("Train Label Distribution")
    axes[0].set_xlabel("Count")
    axes[0].set_ylabel(target_col)
    sns.barplot(x=test_counts.values, y=test_counts.index, ax=axes[1], palette="Greens_r")
    axes[1].set_title("Test Label Distribution")
    axes[1].set_xlabel("Count")
    axes[1].set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_missing_values_plot(df: pd.DataFrame, output_path: Path, top_n: int = 15) -> None:
    missing_pct = df.isna().mean().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=missing_pct.values * 100, y=missing_pct.index, ax=ax, palette="magma")
    ax.set_title("Top Missing Columns")
    ax.set_xlabel("Missing Percentage")
    ax.set_ylabel("Column")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_numeric_distribution_plots(df: pd.DataFrame, numeric_cols: list[str], output_path: Path, top_n: int = 6) -> None:
    selected = numeric_cols[:top_n]
    fig, axes = plt.subplots(len(selected), 1, figsize=(12, 4 * len(selected)))
    axes = [axes] if len(selected) == 1 else axes
    for ax, col in zip(axes, selected):
        sns.histplot(df[col], bins=40, kde=True, ax=ax, color="#2a9d8f")
        ax.set_title(f"Distribution: {col}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_boxplots(df: pd.DataFrame, numeric_cols: list[str], output_path: Path, top_n: int = 6) -> None:
    variances = df[numeric_cols].var(numeric_only=True).sort_values(ascending=False).head(top_n)
    melted = df[variances.index].melt(var_name="feature", value_name="value")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=melted, x="feature", y="value", ax=ax, palette="Set2")
    ax.set_title("Top Variance Numeric Features")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str], output_path: Path, top_n: int = 15) -> None:
    selected = df[numeric_cols].var(numeric_only=True).sort_values(ascending=False).head(top_n).index
    corr = df[selected].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap (Top Variance Numeric Features)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_categorical_frequency_plots(
    df: pd.DataFrame, categorical_cols: list[str], output_path: Path, top_categories: int = 10
) -> None:
    selected = categorical_cols[: min(3, len(categorical_cols))]
    fig, axes = plt.subplots(len(selected), 1, figsize=(12, 4 * len(selected)))
    axes = [axes] if len(selected) == 1 else axes
    for ax, col in zip(axes, selected):
        counts = df[col].astype(str).value_counts().head(top_categories)
        sns.barplot(x=counts.values, y=counts.index, ax=ax, palette="viridis")
        ax.set_title(f"Top Categories: {col}")
        ax.set_xlabel("Count")
        ax.set_ylabel(col)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
