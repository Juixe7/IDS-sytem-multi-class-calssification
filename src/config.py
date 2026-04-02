from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)
    plots_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)
    train_csv: Path = field(init=False)
    test_csv: Path = field(init=False)
    train_parquet: Path = field(init=False)
    test_parquet: Path = field(init=False)
    final_model_path: Path = field(init=False)
    model_metrics_path: Path = field(init=False)
    final_model_manifest_path: Path = field(init=False)
    repo_clone_dir: Path = field(init=False)
    repo_multi_data_path: Path = field(init=False)
    repo_audit_json_path: Path = field(init=False)
    repo_audit_md_path: Path = field(init=False)
    repo_audit_plot_path: Path = field(init=False)
    random_state: int = 42
    target_column: str = "attack_cat"
    binary_label_column: str = "label"
    correlation_threshold: float = 0.95
    feature_cap: int = 75
    shap_sample_size: int = 300
    cv_splits: int = 3
    mode: str = "production"
    default_production_model: str = "lgbm"
    force_retrain_final_model: bool = False
    logistic_params: dict = field(
        default_factory=lambda: {
            "max_iter": 400,
            "multi_class": "multinomial",
            "solver": "lbfgs",
            "class_weight": "balanced",
            "random_state": 42,
        }
    )
    rf_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 100,
            "max_depth": 20,
            "class_weight": "balanced",
            "n_jobs": 1,
            "random_state": 42,
        }
    )
    xgb_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 150,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "random_state": 42,
            "n_jobs": 1,
        }
    )
    lgbm_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 400,
            "num_leaves": 64,
            "max_depth": -1,
            "learning_rate": 0.05,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": 1,
            "verbose": -1,
        }
    )

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "unsw"
        self.artifacts_dir = self.project_root / "artifacts"
        self.plots_dir = self.artifacts_dir / "plots"
        self.reports_dir = self.artifacts_dir / "reports"
        self.logs_dir = self.artifacts_dir / "logs"
        self.models_dir = self.project_root / "models"
        self.outputs_dir = self.project_root / "outputs"
        self.train_csv = self.data_dir / "UNSW_NB15_training-set.csv"
        self.test_csv = self.data_dir / "UNSW_NB15_testing-set.csv"
        self.train_parquet = self.data_dir / "train.parquet"
        self.test_parquet = self.data_dir / "test.parquet"
        self.final_model_path = self.models_dir / "final_model.pkl"
        self.model_metrics_path = self.reports_dir / "model_metrics.json"
        self.final_model_manifest_path = self.reports_dir / "final_model_manifest.json"
        self.repo_clone_dir = self.project_root / ".codex-tmp" / "IoT-Network-Intrusion-Detection-System-UNSW-NB15"
        self.repo_multi_data_path = self.repo_clone_dir / "datasets" / "multi_data.csv"
        self.repo_audit_json_path = self.reports_dir / "repo_benchmark_audit.json"
        self.repo_audit_md_path = self.reports_dir / "repo_benchmark_audit.md"
        self.repo_audit_plot_path = self.plots_dir / "repo_benchmark_accuracy_comparison.png"


def get_config(mode: str | None = None, force_retrain_final_model: bool = False) -> Config:
    config = Config()
    if mode is not None:
        config.mode = mode
    config.force_retrain_final_model = force_retrain_final_model
    return config
