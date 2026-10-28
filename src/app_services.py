"""Cached data, model and SHAP resources shared across Streamlit pages."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import FEATURE_COLUMNS, load_german_credit_data
from src.prediction import load_xgboost_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "german.data"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost.joblib"
MODEL_METADATA_PATH = PROJECT_ROOT / "models" / "xgboost_metadata.json"
RF_METRICS_PATH = PROJECT_ROOT / "reports" / "model_metrics.json"
XGB_METRICS_PATH = PROJECT_ROOT / "reports" / "xgboost" / "model_metrics.json"


@st.cache_data(max_entries=2)
def load_dataset() -> pd.DataFrame:
    """Load the validated official data once for app pages."""
    return load_german_credit_data(DATA_PATH)


@st.cache_data(max_entries=2)
def load_training_features() -> pd.DataFrame:
    """Read the exact saved training rows used by both model experiments."""
    path = PROCESSED_DIR / "X_train.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Saved training data is missing: {path}. "
            "Run the preprocessing command from README.md first."
        )
    try:
        training = pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read the saved training file {path}.") from exc
    if list(training.columns) != list(FEATURE_COLUMNS):
        raise ValueError("Saved training data does not match the verified schema.")
    return training


@st.cache_resource(max_entries=2)
def load_cached_model(model_path: str, modified_ns: int):
    """Cache only the model artifact at this project's fixed local path."""
    del modified_ns
    if Path(model_path).resolve() != MODEL_PATH.resolve():
        raise ValueError("The app only loads its fixed, locally trained model.")
    return load_xgboost_pipeline(model_path)


def get_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"XGBoost model is missing: {MODEL_PATH}. "
            "Train it with `python -m src.train_xgboost`."
        )
    return load_cached_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)


@st.cache_resource(max_entries=2)
def _cached_explainer(
    model_path: str,
    modified_ns: int,
    background_signature: str,
):
    del background_signature
    model = load_cached_model(model_path, modified_ns)
    background = load_training_features().sample(
        n=min(60, len(load_training_features())),
        random_state=42,
        replace=False,
    )
    transformed_background = model.named_steps["preprocessor"].transform(background)
    if hasattr(transformed_background, "toarray"):
        transformed_background = transformed_background.toarray()
    columns = model.named_steps["preprocessor"].get_feature_names_out()
    transformed_background = pd.DataFrame(
        transformed_background, columns=columns, index=background.index
    )
    from src.explain import make_tree_explainer

    return make_tree_explainer(
        model.named_steps["classifier"], transformed_background
    )


def get_explainer():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"XGBoost model is missing: {MODEL_PATH}. "
            "Train it with `python -m src.train_xgboost`."
        )
    background = load_training_features().sample(
        n=min(60, len(load_training_features())),
        random_state=42,
        replace=False,
    )
    signature = hashlib.sha256(
        pd.util.hash_pandas_object(background, index=True).to_numpy().tobytes()
    ).hexdigest()
    return _cached_explainer(
        str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns, signature
    )


@st.cache_data(max_entries=6)
def load_metrics(path_text: str, modified_ns: int) -> dict:
    del modified_ns
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation report not found: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read evaluation report {path}.") from exc
    if "metrics" not in report or "confusion_matrix" not in report:
        raise ValueError(f"Evaluation report is incomplete: {path}")
    return report


def try_load_metrics(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return load_metrics(str(path), path.stat().st_mtime_ns)


@st.cache_data(max_entries=2)
def load_model_metadata() -> dict:
    if not MODEL_METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"XGBoost metadata is missing: {MODEL_METADATA_PATH}. "
            "Run `python -m src.train_xgboost`."
        )
    try:
        return json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Could not read XGBoost model metadata.") from exc


def numeric_training_ranges(training: pd.DataFrame) -> dict[str, tuple[int, int]]:
    from src.data_loader import NUMERICAL_FEATURES

    return {
        column: (
            int(training[column].min()),
            int(training[column].max()),
        )
        for column in NUMERICAL_FEATURES
    }
