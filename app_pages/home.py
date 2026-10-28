"""Executive dashboard with verified dataset and experiment results."""

from pathlib import Path

import streamlit as st

from src.app_services import (
    PROJECT_ROOT,
    RF_METRICS_PATH,
    XGB_METRICS_PATH,
    get_model,
    load_dataset,
    try_load_metrics,
)
from src.data_loader import FEATURE_COLUMNS


st.title("Explainable AI for Credit Risk Prediction")
st.html(
    """
    <section class="xai-hero" aria-label="Project overview">
      <p class="xai-eyebrow">Risk analytics · interpreted with SHAP</p>
      <h2 class="xai-hero-title">A clearer view of credit-risk model behavior</h2>
      <p class="xai-hero-copy">
        Explore a verified UCI dataset, compare measured model results, and
        inspect how the trained classifier produces an output. Built for
        learning—not for real credit decisions.
      </p>
    </section>
    """
)

if st.button(
    "Run an applicant assessment",
    type="primary",
    icon=":material/arrow_forward:",
):
    st.switch_page("app_pages/predict.py")

dataset = None
try:
    dataset = load_dataset()
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc), title="Dataset unavailable")

model = None
try:
    model = get_model()
except (FileNotFoundError, ValueError, TypeError) as exc:
    st.error(str(exc), title="XGBoost model unavailable")

rf_report = None
xgb_report = None
try:
    rf_report = try_load_metrics(RF_METRICS_PATH)
    xgb_report = try_load_metrics(XGB_METRICS_PATH)
except (OSError, ValueError) as exc:
    st.error(str(exc), title="Saved model metrics could not be loaded")

xgb_metrics = xgb_report["metrics"] if xgb_report is not None else {}
xgb_accuracy = xgb_metrics.get("accuracy")

st.subheader("Project snapshot")
first, second, third, fourth = st.columns(4)
snapshot_metrics = (
    ("Records", "Unavailable" if dataset is None else f"{len(dataset):,}"),
    ("Input attributes", f"{len(FEATURE_COLUMNS)}"),
    (
        "Inference model",
        "Unavailable"
        if model is None
        else type(model.named_steps["classifier"]).__name__,
    ),
    (
        "XGBoost test accuracy",
        "Unavailable" if xgb_accuracy is None else f"{xgb_accuracy:.1%}",
    ),
)
for column, (label, value) in zip((first, second, third, fourth), snapshot_metrics):
    with column:
        with st.container(border=True):
            st.metric(label, value)
st.caption(
    "Dataset and model values are read from the validated source, trusted "
    "local model artifact, and saved held-out evaluation reports."
)

if dataset is not None:
    st.subheader("Dataset composition")
    distribution, performance = st.columns((1, 1.35), gap="large")
    with distribution, st.container(border=True):
        st.markdown("#### Observed risk labels")
        class_counts = dataset["credit_risk"].value_counts().rename_axis("Risk label")
        st.bar_chart(
            class_counts,
            x_label="Documented UCI risk label",
            y_label="Records",
            sort=False,
            alt="Record counts for the documented good-risk and bad-risk labels.",
        )
        st.caption("Source code 1 = good-risk · source code 2 = bad-risk.")
    with performance, st.container(border=True):
        st.markdown("#### Measured test-set performance")
        if rf_report is None:
            st.info("Random Forest results are unavailable. Train the baseline model first.")
        if xgb_report is None:
            st.info("XGBoost results are unavailable. Train the primary model first.")

        for name, report in (
            ("Random Forest", rf_report),
            ("XGBoost", xgb_report),
        ):
            if report is None:
                continue
            st.markdown(f"**{name}**")
            columns = st.columns(5)
            for column, metric in zip(
                columns, ("accuracy", "precision", "recall", "f1_score", "roc_auc")
            ):
                value = report["metrics"].get(metric)
                column.metric(
                    metric.replace("_", " ").title(),
                    "N/A" if value is None else f"{value:.3f}",
                )
        st.caption(
            "Precision, recall, and F1 use binary averaging with bad-risk as "
            "the positive class. Results are descriptive for this test split."
        )

st.subheader("Global model explanation")
importance_path = (
    PROJECT_ROOT / "reports" / "figures" / "xgboost_shap_importance.png"
)
if importance_path.is_file():
    with st.container(border=True):
        st.image(
            importance_path,
            alt=(
                "Global XGBoost feature importance using mean absolute SHAP "
                "values for the bad-risk raw-margin output."
            ),
        )
        st.caption(
            "Mean absolute SHAP magnitude summarizes the trained XGBoost "
            "model's bad-risk raw-margin contributions; it does not establish "
            "causality or show the direction of each effect."
        )
else:
    st.info("Global SHAP plot unavailable. Generate it with `python -m src.explain`.")

st.subheader("How the workflow fits together")
st.html(
    """
    <div class="xai-flow" role="list" aria-label="Prediction workflow">
      <div class="xai-flow-step" role="listitem">
        <span class="xai-flow-number">01 · INPUT</span>
        <span class="xai-flow-title">Applicant data</span>
      </div>
      <div class="xai-flow-step" role="listitem">
        <span class="xai-flow-number">02 · MODEL</span>
        <span class="xai-flow-title">Fitted XGBoost pipeline</span>
      </div>
      <div class="xai-flow-step" role="listitem">
        <span class="xai-flow-number">03 · OUTPUT</span>
        <span class="xai-flow-title">Risk category</span>
      </div>
      <div class="xai-flow-step" role="listitem">
        <span class="xai-flow-number">04 · EXPLAIN</span>
        <span class="xai-flow-title">SHAP contribution view</span>
      </div>
    </div>
    """
)

st.subheader("Latest prediction in this session")
latest = st.session_state.get("latest_prediction")
if latest is None:
    st.info(
        "No applicant prediction has been submitted in this browser session yet. "
        "A summary will appear here after a real prediction."
    )
else:
    risk_label = latest["risk_label"]
    bad_risk_probability = latest["bad_risk_probability"]
    with st.container(border=True):
        st.html(
            f"""
            <div class="xai-session-summary">
              <strong>Latest model output:</strong> {risk_label} · estimated
              bad-risk probability {bad_risk_probability:.1%}
            </div>
            """
        )
        st.caption(
            "This is the most recent prediction stored in this session, not a "
            "history across users or sessions."
        )
        st.page_link(
            "app_pages/explain.py",
            label="Review its SHAP explanation",
            icon=":material/psychology:",
        )

st.warning(
    "Educational prototype only. This historical academic dataset and model "
    "are not suitable for real lending, credit-approval, or financial-advice "
    "decisions. SHAP describes model behavior; it does not prove causation.",
    icon=":material/info:",
)
