"""Headless smoke test for the Streamlit multipage entry point."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_home_dashboard_renders_without_errors():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(
        timeout=30
    )

    assert not app.exception
    assert app.title[0].value == "Explainable AI for Credit Risk Prediction"
    assert any(metric.label == "Records" and metric.value == "1,000" for metric in app.metric)
    assert any(button.label == "Run an applicant assessment" for button in app.button)
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert any(button.label == "Predict credit risk" for button in app.button)


def test_prediction_and_performance_pages_render_without_errors():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(
        timeout=30
    )
    app.switch_page("app_pages/predict.py").run(timeout=30)
    assert not app.exception
    assert any(button.label == "Predict credit risk" for button in app.button)

    app.switch_page("app_pages/performance.py").run(timeout=30)
    assert not app.exception


def test_explainability_and_dataset_pages_render_without_errors():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(
        timeout=30
    )
    app.switch_page("app_pages/explain.py").run(timeout=30)
    assert not app.exception

    app.switch_page("app_pages/dataset.py").run(timeout=30)
    assert not app.exception


def test_applicant_submission_shows_a_real_prediction_and_explanation():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(
        timeout=30
    )
    app.switch_page("app_pages/predict.py").run(timeout=30)
    app.form_submit_button[0].click().run(timeout=60)

    assert not app.exception
    alerts = list(app.error) + list(app.success)
    assert any("Predicted category:" in item.value for item in alerts)
