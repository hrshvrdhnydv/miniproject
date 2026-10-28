"""Global and latest-applicant SHAP explanations."""

from pathlib import Path

import streamlit as st

from src.app_services import get_explainer, get_model, load_training_features
from src.explain import (
    EXPLAINED_CLASS,
    make_individual_explanation,
    plot_individual_explanation,
    save_global_explanation_plots,
)


st.title("Explainable AI")
st.write(
    "SHAP attributes parts of the fitted XGBoost model's output to the "
    "transformed input features. These values describe model behavior, not "
    "causal effects or a reason to approve or reject an application."
)
st.info(
    f"Output explained: **{EXPLAINED_CLASS} raw margin (log-odds scale)**. "
    "Positive values push the model margin toward bad-risk; negative values "
    "push it away from bad-risk.",
    title="How to read these explanations",
)

figures_dir = Path(__file__).resolve().parents[1] / "reports" / "figures"
importance_path = figures_dir / "xgboost_shap_importance.png"
beeswarm_path = figures_dir / "xgboost_shap_beeswarm.png"

if not importance_path.is_file() or not beeswarm_path.is_file():
    st.warning(
        "Global SHAP plots are missing. Generate them with "
        "`python -m src.explain` from the project root."
    )
else:
    first, second = st.columns(2)
    with first:
        st.subheader("Global feature importance")
        st.image(
            importance_path,
            alt="Mean absolute SHAP values for top transformed XGBoost features.",
        )
        st.caption(
            "Mean absolute SHAP magnitude measures average contribution size "
            "in the explained raw-margin output; it does not show direction."
        )
    with second:
        st.subheader("SHAP beeswarm")
        st.image(
            beeswarm_path,
            alt="Distribution and direction of SHAP values across the sampled training applicants.",
        )
        st.caption(
            "Each point is a training applicant. Horizontal position is its "
            "SHAP value; color indicates the transformed feature value."
        )

st.subheader("Latest individual prediction")
applicant = st.session_state.get("latest_applicant")
prediction = st.session_state.get("latest_prediction")
if applicant is None or prediction is None:
    st.info(
        "No submitted applicant is stored in this session. Use the Credit risk "
        "prediction page to generate an explanation."
    )
else:
    st.write(f"Most recent model category: **{prediction['risk_label']}**.")
    try:
        model = get_model()
        explainer = get_explainer()
        individual = make_individual_explanation(model, explainer, applicant)
        figure = plot_individual_explanation(individual)
        st.pyplot(
            figure,
            clear_figure=True,
            alt="Latest applicant SHAP waterfall explaining the bad-risk raw margin.",
        )
    except (FileNotFoundError, ValueError, TypeError) as exc:
        st.error(str(exc), title="Could not render the individual explanation")

st.subheader("Rebuild global explanations")
if st.button("Regenerate global SHAP plots", type="secondary"):
    try:
        with st.spinner("Explaining a bounded training sample..."):
            model = get_model()
            training = load_training_features()
            generated = save_global_explanation_plots(
                model, training, output_dir=figures_dir
            )
        st.success(f"Saved updated global plots: {generated}")
    except (FileNotFoundError, ValueError, TypeError) as exc:
        st.error(str(exc), title="SHAP generation failed")
