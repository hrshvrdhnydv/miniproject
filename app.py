"""Streamlit entry point for the educational German Credit XAI project."""

import streamlit as st

from src.app_services import get_model, load_dataset
from src.ui import inject_styles, render_brand_header, render_sidebar_brand

st.set_page_config(
    page_title="Credit Risk XAI",
    page_icon=":material/insights:",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()
render_brand_header()
render_sidebar_brand()

if "latest_applicant" not in st.session_state:
    st.session_state.latest_applicant = None
if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None

pages = {
    "Project": [
        st.Page("app_pages/home.py", title="Home dashboard", icon=":material/home:"),
        st.Page(
            "app_pages/predict.py",
            title="Credit risk prediction",
            icon=":material/assessment:",
        ),
    ],
    "Insights": [
        st.Page(
            "app_pages/explain.py",
            title="Explainable AI",
            icon=":material/psychology:",
        ),
        st.Page(
            "app_pages/performance.py",
            title="Model performance",
            icon=":material/monitoring:",
        ),
        st.Page(
            "app_pages/dataset.py",
            title="Dataset analytics",
            icon=":material/analytics:",
        ),
    ],
}

page = st.navigation(pages, position="sidebar")
with st.sidebar:
    st.divider()
    try:
        dataset = load_dataset()
        st.metric("Verified dataset", f"{len(dataset):,} records")
        st.caption("20 inputs · UCI class 1 = good, class 2 = bad")
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Dataset unavailable: {exc}")

    try:
        model = get_model()
        model_name = type(model.named_steps["classifier"]).__name__
        st.metric("Inference model", model_name)
        st.caption("Loaded from the trusted local project artifact.")
    except (FileNotFoundError, ValueError, TypeError) as exc:
        st.warning(f"Model unavailable: {exc}")

    st.divider()
    st.caption(
        "Academic prototype only — not a lending, credit-approval, "
        "or financial-advice system."
    )

page.run()
