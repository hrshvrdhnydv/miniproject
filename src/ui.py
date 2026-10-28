"""Shared visual theme helpers for the Streamlit application."""

from pathlib import Path

import streamlit as st


STYLESHEET_PATH = Path(__file__).resolve().parents[1] / "assets" / "styles.css"


def inject_styles() -> None:
    """Load the project stylesheet once at the top of the app entry point."""
    if not STYLESHEET_PATH.is_file():
        raise FileNotFoundError(
            f"Application stylesheet is missing: {STYLESHEET_PATH}"
        )
    css = STYLESHEET_PATH.read_text(encoding="utf-8")
    st.html(f"<style>{css}</style>")


def render_brand_header() -> None:
    """Render a compact, responsive brand header using static accessible HTML."""
    st.html(
        """
        <header class="xai-brand" aria-label="Credit risk explainable AI project">
          <div class="xai-brand-mark" aria-hidden="true">CR</div>
          <div class="xai-brand-copy">
            <div class="xai-brand-name">Credit risk intelligence</div>
            <div class="xai-brand-subtitle">Explainable machine learning | UCI German Credit</div>
          </div>
          <div class="xai-brand-tag">ACADEMIC PROTOTYPE</div>
        </header>
        """
    )


def render_sidebar_brand() -> None:
    """Render compact project identity in the shared navigation sidebar."""
    with st.sidebar:
        st.html(
            """
            <header class="xai-sidebar-brand" aria-label="Credit risk XAI">
              <div class="xai-brand-mark" aria-hidden="true">CR</div>
              <div class="xai-brand-copy">
                <div class="xai-brand-name">Credit risk XAI</div>
                <div class="xai-brand-subtitle">Explainable analytics</div>
              </div>
            </header>
            """
        )
