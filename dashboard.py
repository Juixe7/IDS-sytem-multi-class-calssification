from __future__ import annotations
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import streamlit as st

from src.config import Config
from src.dashboard_service import DashboardService
from src.dashboard_utils import (
    PLOT_EXPLANATIONS,
    build_empty_record,
    ensure_frame_columns,
    list_existing_plots,
    load_feature_artifacts,
    load_json,
    load_text,
    prediction_rows_to_frame,
)


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG = Config(project_root=PROJECT_ROOT)


@st.cache_resource(show_spinner=False)
def get_service() -> DashboardService:
    return DashboardService(PROJECT_ROOT)


@st.cache_data(show_spinner=False)
def get_feature_bundle() -> dict:
    return load_feature_artifacts(PROJECT_ROOT)


@st.cache_data(show_spinner=False)
def get_report_bundle() -> dict:
    return {
        "final_summary": load_json(CONFIG.reports_dir / "final_summary.json"),
        "final_manifest": load_json(CONFIG.final_model_manifest_path),
        "metrics": load_json(CONFIG.model_metrics_path),
        "repo_audit": load_json(CONFIG.repo_audit_json_path),
        "repo_audit_md": load_text(CONFIG.repo_audit_md_path),
        "shap_summary_md": load_text(CONFIG.reports_dir / "shap_summary.md"),
    }


def render_status_banner(service: DashboardService) -> None:
    if service.connection_mode() == "api":
        st.success(service.mode_label())
    else:
        st.warning(service.mode_label())
    st.caption("The dashboard prefers the local API, then falls back to direct local inference if the API is unavailable.")


def render_metric_cards(report_bundle: dict) -> None:
    manifest = report_bundle["final_manifest"]
    metrics = report_bundle["metrics"]
    final_model = manifest.get("final_model", "rf")
    final_metrics = metrics.get(final_model, {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final Model", str(final_model).upper())
    col2.metric("Accuracy", f"{final_metrics.get('accuracy', 0.0) * 100:.2f}%")
    col3.metric("Macro F1", f"{final_metrics.get('macro_f1', 0.0):.4f}")
    col4.metric("Weighted F1", f"{final_metrics.get('weighted_f1', 0.0):.4f}")


def render_overview(service: DashboardService, report_bundle: dict) -> None:
    st.title("UNSW IDS Dashboard")
    st.subheader("Overview")
    render_status_banner(service)
    render_metric_cards(report_bundle)

    final_summary = report_bundle["final_summary"]
    manifest = report_bundle["final_manifest"]
    metrics = report_bundle["metrics"]
    metadata = service.get_metadata()

    st.warning("The cloned repo's 97% benchmark is not the production benchmark. It depends on target-derived leakage features and is shown separately for audit purposes.")

    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown("### Production Status")
        st.json(
            {
                "final_model": manifest.get("final_model"),
                "mode": final_summary.get("mode"),
                "selected_feature_count": metadata.get("selected_feature_count"),
                "class_names": metadata.get("class_names"),
            }
        )
    with right:
        st.markdown("### Artifact Status")
        plot_count = len(list_existing_plots(PROJECT_ROOT))
        st.write(f"Saved plots available: `{plot_count}`")
        st.write(f"Final model artifact: `{CONFIG.final_model_path.exists()}`")
        st.write(f"Repo benchmark audit: `{CONFIG.repo_audit_json_path.exists()}`")
        st.write(f"SHAP summary report: `{(CONFIG.reports_dir / 'shap_summary.md').exists()}`")

    st.markdown("### Official Model Comparison")
    if metrics:
        overview_table = pd.DataFrame(
            [
                {
                    "model": name.upper(),
                    "accuracy": values.get("accuracy"),
                    "macro_f1": values.get("macro_f1"),
                    "weighted_f1": values.get("weighted_f1"),
                    "roc_auc_ovr": values.get("roc_auc_ovr"),
                    "train_seconds": values.get("train_seconds"),
                }
                for name, values in metrics.items()
            ]
        )
        st.dataframe(overview_table, use_container_width=True, hide_index=True)
    else:
        st.info("Model metrics are not available yet.")


def render_evaluation(report_bundle: dict) -> None:
    st.title("UNSW IDS Dashboard")
    st.subheader("Evaluation")

    st.markdown("### Official Predefined-Split Results")
    official_plots = [
        "model_metric_comparison.png",
        "training_time_comparison.png",
        "roc_curve_comparison.png",
        "per_class_f1_heatmap.png",
        "confusion_matrix_rf.png",
    ]
    for plot_name in official_plots:
        plot_path = CONFIG.plots_dir / plot_name
        if plot_path.exists():
            st.image(str(plot_path), use_column_width=True)
            st.caption(PLOT_EXPLANATIONS.get(plot_name, "Saved evaluation plot."))
        else:
            st.warning(f"Missing plot: {plot_name}")

    st.markdown("### External Repo Benchmark Audit")
    repo_audit = report_bundle["repo_audit"]
    if repo_audit:
        audit_cols = st.columns(2)
        audit_cols[0].metric("Repo Style Accuracy (With Leakage)", f"{repo_audit['with_leakage']['accuracy'] * 100:.2f}%")
        audit_cols[1].metric(
            "Repo Style Accuracy (Without Leakage)",
            f"{repo_audit['without_attack_cat_dummies']['accuracy'] * 100:.2f}%",
        )
        audit_plot = CONFIG.plots_dir / "repo_benchmark_accuracy_comparison.png"
        if audit_plot.exists():
            st.image(str(audit_plot), use_column_width=True)
            st.caption(PLOT_EXPLANATIONS.get(audit_plot.name, "Repo benchmark audit plot."))
        repo_audit_md = report_bundle["repo_audit_md"]
        if repo_audit_md:
            st.markdown(repo_audit_md)
    else:
        st.info("Repo benchmark audit artifacts are not available.")


def _render_manual_record_form(feature_bundle: dict, form_key: str) -> dict | None:
    defaults = build_empty_record(feature_bundle)
    numeric_cols = set(feature_bundle["feature_info"].get("numeric_cols", []))
    categories = feature_bundle.get("categories", {})
    core_cols = feature_bundle.get("expected_columns", [])
    initial_cols = core_cols[:18]
    advanced_cols = core_cols[18:]

    with st.form(form_key):
        st.markdown("### Manual Single-Record Input")
        submitted_values: dict[str, object] = {}
        form_columns = st.columns(3)
        for index, column in enumerate(initial_cols):
            container = form_columns[index % 3]
            with container:
                if column in categories and categories[column]:
                    default_value = str(defaults.get(column, categories[column][0]))
                    options = categories[column]
                    default_index = options.index(default_value) if default_value in options else 0
                    submitted_values[column] = st.selectbox(column, options=options, index=default_index, key=f"{form_key}_{column}")
                else:
                    submitted_values[column] = st.number_input(
                        column,
                        value=float(defaults.get(column, 0.0) or 0.0),
                        key=f"{form_key}_{column}",
                    )
        with st.expander("Advanced fields", expanded=False):
            advanced_columns = st.columns(3)
            for index, column in enumerate(advanced_cols):
                container = advanced_columns[index % 3]
                with container:
                    if column in categories and categories[column]:
                        default_value = str(defaults.get(column, categories[column][0]))
                        options = categories[column]
                        default_index = options.index(default_value) if default_value in options else 0
                        submitted_values[column] = st.selectbox(
                            column,
                            options=options,
                            index=default_index,
                            key=f"{form_key}_{column}",
                        )
                    else:
                        submitted_values[column] = st.number_input(
                            column,
                            value=float(defaults.get(column, 0.0) or 0.0),
                            key=f"{form_key}_{column}",
                        )
        submit = st.form_submit_button("Run Prediction")
    if submit:
        return submitted_values
    return None


def render_predict(service: DashboardService, feature_bundle: dict) -> None:
    st.title("UNSW IDS Dashboard")
    st.subheader("Predict")
    render_status_banner(service)

    manual_tabs = st.tabs(["Single Record", "CSV Batch Upload"])

    with manual_tabs[0]:
        manual_record = _render_manual_record_form(feature_bundle, "predict_form")
        if manual_record is not None:
            record_frame = ensure_frame_columns(pd.DataFrame([manual_record]), feature_bundle)
            predictions = service.predict(record_frame)
            prediction_table = prediction_rows_to_frame(predictions)
            if predictions:
                top_prediction = predictions[0]
                cols = st.columns(3)
                cols[0].metric("Predicted Class", top_prediction.get("predicted_label", "n/a"))
                cols[1].metric("Confidence", f"{top_prediction.get('confidence', 0.0) * 100:.2f}%")
                cols[2].metric("Predicted Index", str(top_prediction.get("predicted_index", "n/a")))
            st.dataframe(prediction_table, use_container_width=True, hide_index=True)
            if predictions and predictions[0].get("probabilities"):
                probability_frame = pd.DataFrame(
                    [
                        {"class_name": class_name, "probability": probability}
                        for class_name, probability in predictions[0]["probabilities"].items()
                    ]
                ).sort_values("probability", ascending=False)
                st.markdown("#### Per-Class Probabilities")
                st.dataframe(probability_frame, use_container_width=True, hide_index=True)

    with manual_tabs[1]:
        uploaded = st.file_uploader("Upload a CSV of raw UNSW feature rows", type=["csv"], key="predict_csv")
        if uploaded is not None:
            batch_frame = pd.read_csv(uploaded)
            batch_frame = ensure_frame_columns(batch_frame, feature_bundle)
            st.markdown("#### Preview")
            st.dataframe(batch_frame.head(10), use_container_width=True)
            if st.button("Run Batch Predictions", key="run_batch_predictions"):
                predictions = service.predict(batch_frame)
                prediction_frame = prediction_rows_to_frame(predictions)
                merged = pd.concat([batch_frame.reset_index(drop=True), prediction_frame], axis=1)
                st.dataframe(merged.head(25), use_container_width=True, hide_index=True)
                csv_bytes = merged.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Batch Predictions",
                    data=csv_bytes,
                    file_name="unsw_ids_batch_predictions.csv",
                    mime="text/csv",
                )
                st.session_state["dashboard_last_batch_frame"] = batch_frame


def render_explain(service: DashboardService, feature_bundle: dict, report_bundle: dict) -> None:
    st.title("UNSW IDS Dashboard")
    st.subheader("Explain")
    render_status_banner(service)

    explain_tabs = st.tabs(["Single Record Explanation", "Explain Uploaded Row", "Saved SHAP Visuals"])

    with explain_tabs[0]:
        manual_record = _render_manual_record_form(feature_bundle, "explain_form")
        if manual_record is not None:
            frame = ensure_frame_columns(pd.DataFrame([manual_record]), feature_bundle)
            explanations = service.explain(frame, top_n=5)
            if explanations:
                explanation = explanations[0]
                st.markdown("### Prediction Summary")
                st.json(explanation["prediction"])
                st.markdown("### Top SHAP Drivers")
                st.dataframe(pd.DataFrame(explanation["top_features"]), use_container_width=True, hide_index=True)

    with explain_tabs[1]:
        batch_frame = st.session_state.get("dashboard_last_batch_frame")
        if batch_frame is None:
            uploaded = st.file_uploader("Upload a CSV to pick a row for explanation", type=["csv"], key="explain_csv")
            if uploaded is not None:
                batch_frame = ensure_frame_columns(pd.read_csv(uploaded), feature_bundle)
                st.session_state["dashboard_last_batch_frame"] = batch_frame
        if batch_frame is not None:
            st.dataframe(batch_frame.head(10), use_container_width=True)
            selected_index = st.number_input(
                "Row index to explain",
                min_value=0,
                max_value=max(len(batch_frame) - 1, 0),
                value=0,
                step=1,
            )
            if st.button("Explain Selected Row", key="explain_selected_row"):
                selected_row = batch_frame.iloc[[int(selected_index)]]
                explanations = service.explain(selected_row, top_n=5)
                if explanations:
                    explanation = explanations[0]
                    st.json(explanation["prediction"])
                    st.dataframe(pd.DataFrame(explanation["top_features"]), use_container_width=True, hide_index=True)
        else:
            st.info("Upload a CSV on the Predict page or here to explain a specific row.")

    with explain_tabs[2]:
        shap_summary = report_bundle["shap_summary_md"]
        if shap_summary:
            st.markdown(shap_summary)
        for plot_name in ["shap_summary.png", "shap_bar.png", "shap_waterfall_example.png"]:
            plot_path = CONFIG.plots_dir / plot_name
            if plot_path.exists():
                st.image(str(plot_path), use_column_width=True)
                st.caption(PLOT_EXPLANATIONS.get(plot_name, "Saved SHAP plot."))
            else:
                st.warning(f"Missing SHAP plot: {plot_name}")


def main() -> None:
    st.set_page_config(
        page_title="UNSW IDS SaaS Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    service = get_service()
    feature_bundle = get_feature_bundle()
    report_bundle = get_report_bundle()

    st.sidebar.title("UNSW IDS SaaS")
    st.sidebar.caption(service.mode_label())
    page = st.sidebar.radio("Navigate", ["Overview", "Evaluation", "Predict", "Explain"])

    if page == "Overview":
        render_overview(service, report_bundle)
    elif page == "Evaluation":
        render_evaluation(report_bundle)
    elif page == "Predict":
        render_predict(service, feature_bundle)
    else:
        render_explain(service, feature_bundle, report_bundle)


if __name__ == "__main__":
    main()
