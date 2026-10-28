"""Applicant input and real XGBoost prediction page."""

import pandas as pd
import streamlit as st

from src.app_services import (
    get_explainer,
    get_model,
    load_dataset,
    load_training_features,
    numeric_training_ranges,
)
from src.data_loader import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from src.explain import make_individual_explanation, plot_individual_explanation
from src.prediction import predict_credit_risk


st.title("Credit risk prediction")
st.write(
    "Enter values from the verified German Credit schema. The model's "
    "probabilities are learned estimates on this historical dataset, not "
    "calibrated lending odds or a credit decision."
)

try:
    model = get_model()
    dataset = load_dataset()
    training = load_training_features()
    ranges = numeric_training_ranges(training)
except (FileNotFoundError, ValueError, TypeError) as exc:
    st.error(str(exc), title="Prediction model unavailable")
    st.stop()

labels = {
    "checking_status": "Checking account status",
    "duration_months": "Credit duration (months)",
    "credit_history": "Credit history",
    "purpose": "Credit purpose",
    "credit_amount": "Credit amount",
    "savings_status": "Savings status",
    "employment_since": "Employment duration",
    "installment_rate": "Installment rate",
    "personal_status_sex": "Personal status and sex",
    "other_debtors": "Other debtors / guarantors",
    "residence_since": "Years at current residence",
    "property": "Property",
    "age_years": "Age (years)",
    "other_installment_plans": "Other installment plans",
    "housing": "Housing",
    "existing_credits": "Existing credits",
    "job": "Job category",
    "num_dependents": "Number of dependents",
    "telephone": "Telephone",
    "foreign_worker": "Foreign worker status",
}

field_groups = {
    "Credit profile": (
        "checking_status",
        "duration_months",
        "credit_history",
        "purpose",
        "credit_amount",
        "savings_status",
        "installment_rate",
    ),
    "Personal & employment": (
        "employment_since",
        "personal_status_sex",
        "residence_since",
        "property",
        "age_years",
        "housing",
        "job",
        "num_dependents",
        "foreign_worker",
    ),
    "Other financial commitments": (
        "other_debtors",
        "other_installment_plans",
        "existing_credits",
        "telephone",
    ),
}

with st.form("applicant_form", clear_on_submit=False):
    st.subheader("Applicant profile")
    st.caption(
        "Fields are grouped for readability. Category choices and numeric "
        "ranges come from the verified dataset and saved training split."
    )
    values: dict[str, object] = {}
    for group_name, features in field_groups.items():
        with st.container(border=True):
            st.markdown(f"#### {group_name}")
            left, right = st.columns(2)
            for index, feature in enumerate(features):
                container = left if index % 2 == 0 else right
                with container:
                    if feature in CATEGORICAL_FEATURES:
                        options = sorted(dataset[feature].unique().tolist())
                        values[feature] = st.selectbox(
                            labels[feature],
                            options=options,
                            help=(
                                "Category code from the original UCI German "
                                "Credit dataset. No user-entered categories are accepted."
                            ),
                            key=f"applicant_{feature}",
                        )
                    else:
                        minimum, maximum = ranges[feature]
                        default = int(training[feature].median())
                        default = min(max(default, minimum), maximum)
                        values[feature] = st.number_input(
                            labels[feature],
                            min_value=minimum,
                            max_value=maximum,
                            value=default,
                            step=1,
                            help="Allowed range is taken from the saved training rows.",
                            key=f"applicant_{feature}",
                        )
    submitted = st.form_submit_button(
        "Predict credit risk", type="primary", width="stretch"
    )

if submitted:
    applicant = pd.DataFrame(
        [{feature: values[feature] for feature in CATEGORICAL_FEATURES + NUMERICAL_FEATURES}]
    )
    # Keep the original 20-feature order expected by the fitted transformer.
    from src.data_loader import FEATURE_COLUMNS

    applicant = applicant.loc[:, list(FEATURE_COLUMNS)]
    try:
        with st.spinner("Generating prediction and SHAP explanation..."):
            result = predict_credit_risk(model, applicant, ranges)
            explainer = get_explainer()
            explanation = make_individual_explanation(
                inference_pipeline=model,
                explainer=explainer,
                applicant=applicant,
            )
        st.session_state.latest_applicant = applicant
        st.session_state.latest_prediction = {
            "risk_label": result.risk_label,
            "bad_risk_probability": result.bad_risk_probability,
            "good_risk_probability": result.good_risk_probability,
        }
        st.subheader("Model output")
        if result.risk_label == "bad-risk":
            st.error("Predicted category: bad-risk", icon=":material/warning:")
        else:
            st.success("Predicted category: good-risk", icon=":material/check_circle:")
        first, second = st.columns(2)
        first.metric(
            "Model-estimated probability for bad-risk",
            f"{result.bad_risk_probability:.1%}",
        )
        second.metric(
            "Model-estimated probability for good-risk",
            f"{result.good_risk_probability:.1%}",
        )
        st.caption(
            "These are the classifier's probability outputs, not calibrated "
            "real-world odds and not a lending recommendation."
        )

        st.subheader("Why did this model output occur?")
        st.caption(
            "This SHAP waterfall explains the bad-risk raw margin (log-odds "
            "scale). Positive contributions push the model output toward "
            "bad-risk; negative contributions push it toward good-risk. SHAP "
            "describes model behavior, not causation."
        )
        figure = plot_individual_explanation(explanation)
        st.pyplot(
            figure,
            clear_figure=True,
            alt="SHAP waterfall for this applicant, explaining the bad-risk raw margin.",
        )
    except (FileNotFoundError, ValueError, TypeError) as exc:
        st.error(str(exc), title="Could not generate the prediction")
else:
    st.info(
        "Complete the fields and submit the form to run the real saved model. "
        "The app does not show example or hard-coded predictions."
    )
