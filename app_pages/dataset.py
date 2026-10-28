"""Dataset schema, class distribution and feature analytics."""

import pandas as pd
import streamlit as st

from src.app_services import load_dataset, load_training_features
from src.data_loader import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


st.title("Dataset analytics")
st.write(
    "The application uses the original symbolic UCI Statlog German Credit "
    "data: 1,000 records, 20 input attributes and a final class field. UCI "
    "documents class 1 as good and class 2 as bad."
)

try:
    data = load_dataset()
    training = load_training_features()
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc), title="Dataset unavailable")
    st.stop()

first, second, third, fourth = st.columns(4)
first.metric("Records", f"{len(data):,}")
second.metric("Input attributes", "20")
third.metric("Categorical", len(CATEGORICAL_FEATURES))
fourth.metric("Numerical", len(NUMERICAL_FEATURES))

st.subheader("Class distribution")
class_counts = data["credit_risk"].value_counts().rename_axis("credit_risk")
st.bar_chart(
    class_counts,
    x_label="Documented risk label",
    y_label="Number of records",
    sort=False,
    alt="Number of original dataset records in good-risk and bad-risk classes.",
)
st.caption("Counts are loaded from the official source file, not generated examples.")

st.subheader("Verified input columns and data types")
schema = pd.DataFrame(
    {
        "attribute": data.drop(columns=["credit_risk"]).columns,
        "data_type": [
            "categorical" if column in CATEGORICAL_FEATURES else "integer"
            for column in data.drop(columns=["credit_risk"]).columns
        ],
        "observed_unique_values": [
            int(data[column].nunique())
            for column in data.drop(columns=["credit_risk"]).columns
        ],
        "missing_values": [
            int(data[column].isna().sum())
            for column in data.drop(columns=["credit_risk"]).columns
        ],
    }
)
st.dataframe(schema, width="stretch", hide_index=True)

st.subheader("Explore feature distributions")
feature = st.selectbox(
    "Choose an input attribute",
    options=list(CATEGORICAL_FEATURES) + list(NUMERICAL_FEATURES),
)
if feature in CATEGORICAL_FEATURES:
    counts = data[feature].value_counts().sort_index()
    st.bar_chart(
        counts,
        x_label=feature.replace("_", " ").title(),
        y_label="Record count",
        sort=False,
        alt=f"Observed category counts for {feature}.",
    )
else:
    histogram = (
        data[feature]
        .value_counts(bins=12, sort=False)
        .sort_index()
        .rename_axis("observed interval")
    )
    st.bar_chart(
        histogram,
        x_label=feature.replace("_", " ").title(),
        y_label="Record count",
        sort=False,
        alt=f"Observed numeric distribution for {feature}.",
    )
    st.caption(
        f"Training split range: {int(training[feature].min())} to "
        f"{int(training[feature].max())}. Values are integer-coded as in UCI."
    )

st.subheader("Dataset limitations")
st.markdown(
    "- This is a small, historical dataset and may not represent current "
    "applicants or current lending practices.\n"
    "- Sensitive and proxy attributes are present; predictive performance does "
    "not establish fairness.\n"
    "- A held-out split from one dataset is not external validation.\n"
    "- Target labels and class balance reflect the historical dataset, not "
    "future repayment outcomes in a new population."
)
