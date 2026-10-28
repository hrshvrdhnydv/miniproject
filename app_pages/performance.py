"""Measured Random Forest and XGBoost test-set performance."""

import pandas as pd
import streamlit as st

from src.app_services import RF_METRICS_PATH, XGB_METRICS_PATH, try_load_metrics


st.title("Model performance")
st.write(
    "Metrics below are loaded from saved experiments evaluated on the same "
    "held-out test split. Precision, recall and F1 use binary averaging with "
    "`bad-risk` as the positive class. ROC-AUC uses the estimated probability "
    "for bad-risk."
)

reports = {
    "Random Forest": try_load_metrics(RF_METRICS_PATH),
    "XGBoost": try_load_metrics(XGB_METRICS_PATH),
}
available = {name: report for name, report in reports.items() if report is not None}
if len(available) < 2:
    missing = [name for name, report in reports.items() if report is None]
    st.warning(
        "The following saved result(s) are unavailable: "
        + ", ".join(missing)
        + ". Run the corresponding training command before comparing."
    )

if available:
    metric_names = ("accuracy", "precision", "recall", "f1_score", "roc_auc")
    comparison = pd.DataFrame(
        {
            name: report["metrics"]
            for name, report in available.items()
        }
    ).reindex(metric_names)
    comparison.index = [name.replace("_", " ").title() for name in comparison.index]
    st.subheader("Measured test metrics")
    st.dataframe(comparison, width="stretch")
    st.caption(
        "Higher values are not automatically preferable for every deployment "
        "objective. Model selection requires considering error costs, "
        "calibration, fairness and external validation."
    )

    columns = st.columns(len(available))
    for column, (name, report) in zip(columns, available.items()):
        st.subheader(f"{name} confusion matrix")
        image_path = (
            RF_METRICS_PATH.parent / report["figure"]
            if name == "Random Forest"
            else XGB_METRICS_PATH.parent / report["figure"]
        )
        if image_path.is_file():
            column.image(
                image_path,
                alt=f"{name} confusion matrix, actual classes by predicted classes.",
            )
        else:
            matrix = report["confusion_matrix"]
            counts = matrix["counts"]
            labels = matrix["labels_in_row_and_column_order"]
            column.dataframe(
                pd.DataFrame(counts, index=labels, columns=labels),
                width="stretch",
            )
        column.caption(
            "Rows are actual classes; columns are predicted classes. "
            "False negatives and false positives have different implications."
        )

    st.subheader("Plain-language metric definitions")
    st.markdown(
        "- **Accuracy:** share of all test records classified correctly.\n"
        "- **Precision:** among records predicted bad-risk, share that were bad-risk.\n"
        "- **Recall:** among actual bad-risk records, share found by the model.\n"
        "- **F1-score:** harmonic balance of bad-risk precision and recall.\n"
        "- **ROC-AUC:** ranking ability across thresholds using bad-risk scores; "
        "it is not probability calibration."
    )

st.info(
    "These measurements apply to the historical UCI test set only. They are "
    "not evidence of suitability, fairness, or performance in real lending."
)
