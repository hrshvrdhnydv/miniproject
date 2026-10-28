"""Validated applicant prediction helpers for the trained XGBoost pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_loader import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    VALID_CATEGORICAL_VALUES,
)


MODEL_CLASSES = {0: "good-risk", 1: "bad-risk"}
POSITIVE_CLASS = "bad-risk"


@dataclass(frozen=True)
class CreditRiskPrediction:
    """Model output for one validated applicant."""

    risk_label: str
    positive_class: str
    bad_risk_probability: float
    good_risk_probability: float


def load_xgboost_pipeline(model_path: str | Path) -> Pipeline:
    """Load a project-generated, local model artifact and validate its shape.

    Joblib files can execute code when loaded. Only load artifacts generated
    by this project's training command; never accept uploaded model files.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained XGBoost model not found: {path}. "
            "Run `python -m src.train_xgboost` from the project directory."
        )
    try:
        model = joblib.load(path)
    except (OSError, EOFError, ImportError, ValueError, pickle.UnpicklingError) as exc:
        raise ValueError(
            f"Could not load the trusted project model artifact at {path}. "
            "Re-create it with `python -m src.train_xgboost`."
        ) from exc

    if not isinstance(model, Pipeline):
        raise TypeError("The XGBoost artifact must be a scikit-learn Pipeline.")
    if list(model.named_steps) != ["preprocessor", "classifier"]:
        raise ValueError(
            "The saved inference Pipeline must contain preprocessor and "
            "classifier steps in that order."
        )
    classifier = model.named_steps["classifier"]
    if not isinstance(classifier, XGBClassifier):
        raise TypeError("The saved Pipeline classifier is not an XGBClassifier.")
    if list(classifier.classes_) != [0, 1]:
        raise ValueError(
            "The saved XGBoost model must use class 0=good-risk and 1=bad-risk."
        )
    if not hasattr(model.named_steps["preprocessor"], "get_feature_names_out"):
        raise ValueError("The saved preprocessor cannot provide transformed names.")
    return model


def validate_applicant(
    applicant: pd.DataFrame,
    numerical_ranges: dict[str, tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Validate one raw applicant row without changing its feature ordering."""
    if not isinstance(applicant, pd.DataFrame):
        raise TypeError("Applicant input must be a pandas DataFrame.")
    if applicant.shape != (1, len(FEATURE_COLUMNS)):
        raise ValueError(
            f"Expected exactly one applicant with {len(FEATURE_COLUMNS)} features."
        )
    if list(applicant.columns) != list(FEATURE_COLUMNS):
        raise ValueError(
            "Applicant feature names or order do not match the verified dataset."
        )
    if applicant.isna().any().any():
        raise ValueError("Applicant data is incomplete; provide every input value.")

    validated = applicant.copy()
    for column in CATEGORICAL_FEATURES:
        value = validated.at[validated.index[0], column]
        if not isinstance(value, str) or value not in VALID_CATEGORICAL_VALUES[column]:
            raise ValueError(
                f"Invalid {column} value {value!r}. Choose a documented "
                f"category code: {sorted(VALID_CATEGORICAL_VALUES[column])}."
            )

    for column in NUMERICAL_FEATURES:
        value = validated.at[validated.index[0], column]
        if isinstance(value, bool):
            raise ValueError(f"{column} must be a whole number.")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{column} must be a whole number.") from exc
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"{column} must be a finite whole number.")
        integer_value = int(numeric_value)
        if integer_value <= 0:
            raise ValueError(f"{column} must be greater than zero.")
        if numerical_ranges is not None:
            if column not in numerical_ranges:
                raise ValueError(f"No verified training range is available for {column}.")
            minimum, maximum = numerical_ranges[column]
            if not minimum <= integer_value <= maximum:
                raise ValueError(
                    f"{column} must be between {minimum} and {maximum}, "
                    "the values observed in the training data."
                )
        validated[column] = integer_value

    return validated.loc[:, list(FEATURE_COLUMNS)]


def predict_credit_risk(
    model: Pipeline,
    applicant: pd.DataFrame,
    numerical_ranges: dict[str, tuple[int, int]] | None = None,
) -> CreditRiskPrediction:
    """Return one real prediction and model-estimated class probabilities."""
    if not isinstance(model, Pipeline):
        raise TypeError("Prediction requires the saved XGBoost inference Pipeline.")
    classifier: Any = model.named_steps.get("classifier")
    if not isinstance(classifier, XGBClassifier):
        raise TypeError("Prediction pipeline does not contain an XGBClassifier.")
    validated = validate_applicant(applicant, numerical_ranges)

    try:
        encoded_prediction = int(model.predict(validated)[0])
        probabilities = np.asarray(model.predict_proba(validated), dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "The trained XGBoost pipeline could not process this applicant input."
        ) from exc

    if encoded_prediction not in MODEL_CLASSES:
        raise ValueError(f"Model returned unknown encoded class {encoded_prediction}.")
    classes = list(classifier.classes_)
    if probabilities.shape != (1, len(classes)):
        raise ValueError(
            f"Expected one probability per model class; got shape {probabilities.shape}."
        )
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Model returned invalid class probability values.")

    good_index = classes.index(0)
    bad_index = classes.index(1)
    good_probability = float(probabilities[0, good_index])
    bad_probability = float(probabilities[0, bad_index])
    if not np.isclose(good_probability + bad_probability, 1.0, atol=1e-5):
        raise ValueError("Model class probabilities do not sum to one.")

    return CreditRiskPrediction(
        risk_label=MODEL_CLASSES[encoded_prediction],
        positive_class=POSITIVE_CLASS,
        bad_risk_probability=bad_probability,
        good_risk_probability=good_probability,
    )
