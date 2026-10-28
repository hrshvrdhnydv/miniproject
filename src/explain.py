"""Reusable SHAP explanations for the fitted XGBoost pipeline.

The explained output is the XGBoost raw margin for encoded class 1
(``bad-risk``). Positive SHAP values raise that margin; negative values lower
it. These explanations describe model behavior, not causal effects.
"""

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_loader import FEATURE_COLUMNS


MAX_EXPLANATION_ROWS = 200
DEFAULT_BACKGROUND_SIZE = 60
MAX_DISPLAY_FEATURES = 20
EXPLAINED_CLASS = "bad-risk (encoded class 1)"

FEATURE_LABELS = {
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


def readable_feature_label(transformed_name: str) -> str:
    """Turn ColumnTransformer output names into labels suitable for plots."""
    feature = transformed_name
    for prefix in ("categorical__", "numerical__"):
        if feature.startswith(prefix):
            feature = feature[len(prefix) :]
            break
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]

    for column in sorted(FEATURE_LABELS, key=len, reverse=True):
        prefix = f"{column}_"
        if feature.startswith(prefix):
            category = feature[len(prefix) :]
            return f"{FEATURE_LABELS[column]}: {category}"
    return feature.replace("_", " ").capitalize()


def _require_model(model: Any) -> XGBClassifier:
    if not isinstance(model, XGBClassifier):
        raise TypeError(
            "SHAP expects the fitted XGBClassifier from the XGBoost inference "
            f"pipeline; received {type(model).__name__}."
        )
    if list(model.classes_) != [0, 1]:
        raise ValueError(
            "Expected XGBoost classes [0, 1] where class 1 is bad-risk."
        )
    return model


def make_tree_explainer(
    model: XGBClassifier, background: pd.DataFrame
) -> shap.TreeExplainer:
    """Create a probability-independent explainer for the binary raw margin."""
    _require_model(model)
    if background.empty:
        raise ValueError("The SHAP background dataset must contain at least one row.")
    if len(background) > MAX_EXPLANATION_ROWS:
        raise ValueError(
            f"SHAP background is limited to {MAX_EXPLANATION_ROWS} rows; "
            f"received {len(background)}."
        )
    if not all(isinstance(name, str) for name in background.columns):
        raise ValueError("Transformed background data must have named columns.")

    # The binary raw margin is one scalar output; class 1 is bad-risk.
    return shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional",
        model_output="raw",
    )


def _as_single_output_explanation(
    explainer: shap.TreeExplainer,
    transformed_data: pd.DataFrame,
) -> shap.Explanation:
    if transformed_data.empty:
        raise ValueError("At least one transformed row is required for SHAP.")
    if len(transformed_data) > MAX_EXPLANATION_ROWS:
        raise ValueError(
            f"SHAP is limited to {MAX_EXPLANATION_ROWS} rows per request; "
            f"received {len(transformed_data)}."
        )
    try:
        explanation = explainer(transformed_data)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "SHAP could not explain the transformed XGBoost input. Verify the "
            "feature order and numeric representation match the fitted model."
        ) from exc

    values = np.asarray(explanation.values)
    feature_count = transformed_data.shape[1]
    if values.ndim == 3:
        # SHAP versions may include an explicit output dimension. Select
        # encoded class 1 while preserving rows x transformed features.
        if values.shape[:2] == (len(transformed_data), feature_count):
            if values.shape[2] < 2:
                values = values[:, :, 0]
            else:
                values = values[:, :, 1]
        elif (
            values.shape[0] == len(transformed_data)
            and values.shape[2] == feature_count
        ):
            if values.shape[1] < 2:
                values = values[:, 0, :]
            else:
                values = values[:, 1, :]
        else:
            raise ValueError(
                f"Unexpected 3D SHAP output shape {values.shape}; expected "
                "rows x features x outputs or rows x outputs x features."
            )
    if values.ndim != 2 or values.shape != (
        len(transformed_data),
        feature_count,
    ):
        raise ValueError(
            f"Unexpected SHAP output shape {values.shape}; expected "
            f"({len(transformed_data)}, {feature_count})."
        )

    base_values = np.asarray(explanation.base_values)
    if base_values.ndim == 0:
        selected_base_values: Any = np.repeat(base_values.item(), len(values))
    elif base_values.ndim == 1:
        if len(base_values) == len(values):
            selected_base_values = base_values
        elif len(base_values) >= 2:
            selected_base_values = np.repeat(base_values[1], len(values))
        else:
            selected_base_values = np.repeat(base_values[0], len(values))
    elif base_values.ndim == 2:
        if base_values.shape[0] != len(values):
            raise ValueError(f"Unexpected SHAP base-value shape {base_values.shape}.")
        selected_base_values = (
            base_values[:, 1] if base_values.shape[1] >= 2 else base_values[:, 0]
        )
    else:
        raise ValueError(f"Unexpected SHAP base-value shape {base_values.shape}.")

    return shap.Explanation(
        values=values,
        base_values=selected_base_values,
        data=transformed_data.to_numpy(),
        feature_names=[
            readable_feature_label(column) for column in transformed_data.columns
        ],
    )


def _transform_raw_features(
    inference_pipeline: Any, raw_features: pd.DataFrame
) -> pd.DataFrame:
    if list(raw_features.columns) != list(FEATURE_COLUMNS):
        raise ValueError(
            "SHAP input must contain the 20 raw feature columns in verified order."
        )
    if raw_features.empty:
        raise ValueError("SHAP input must contain at least one applicant.")
    preprocessor = inference_pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        raise ValueError("The inference pipeline has no fitted preprocessor.")
    transformed = preprocessor.transform(raw_features)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = preprocessor.get_feature_names_out()
    return pd.DataFrame(transformed, columns=feature_names, index=raw_features.index)


def save_global_explanation_plots(
    inference_pipeline: Any,
    training_features: pd.DataFrame,
    output_dir: str | Path = Path("reports") / "figures",
    background_size: int = DEFAULT_BACKGROUND_SIZE,
    max_rows: int = MAX_EXPLANATION_ROWS,
    random_state: int = 42,
) -> dict[str, Path]:
    """Save mean-absolute-SHAP importance and beeswarm plots from train rows."""
    classifier = _require_model(inference_pipeline.named_steps.get("classifier"))
    if list(training_features.columns) != list(FEATURE_COLUMNS):
        raise ValueError("Global SHAP requires the verified raw training features.")
    if training_features.empty:
        raise ValueError("Training features are empty; global SHAP is unavailable.")
    if background_size < 1:
        raise ValueError("background_size must be at least one.")
    if max_rows < 1 or max_rows > MAX_EXPLANATION_ROWS:
        raise ValueError(
            f"max_rows must be between 1 and {MAX_EXPLANATION_ROWS}."
        )

    sample_count = min(len(training_features), max_rows)
    sample = training_features.sample(
        n=sample_count, random_state=random_state, replace=False
    )
    background_count = min(len(sample), background_size)
    background = sample.sample(
        n=background_count, random_state=random_state, replace=False
    )
    transformed_sample = _transform_raw_features(inference_pipeline, sample)
    transformed_background = _transform_raw_features(inference_pipeline, background)
    explainer = make_tree_explainer(classifier, transformed_background)
    explanation = _as_single_output_explanation(explainer, transformed_sample)

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    importance_path = target_dir / "xgboost_shap_importance.png"
    importance = np.mean(np.abs(explanation.values), axis=0)
    top_indexes = np.argsort(importance)[-MAX_DISPLAY_FEATURES:]
    figure, axis = plt.subplots(figsize=(10, 7))
    axis.barh(
        np.asarray(explanation.feature_names)[top_indexes],
        importance[top_indexes],
        color="#176b87",
    )
    axis.set_xlabel("Mean absolute SHAP value (XGBoost raw margin)")
    axis.set_title(f"Global feature importance for {EXPLAINED_CLASS}")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(importance_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    beeswarm_path = target_dir / "xgboost_shap_beeswarm.png"
    try:
        shap.plots.beeswarm(
            explanation,
            max_display=MAX_DISPLAY_FEATURES,
            show=False,
            plot_size=(10, 8),
        )
    except (ValueError, TypeError) as exc:
        raise ValueError("Could not render the XGBoost SHAP beeswarm plot.") from exc
    plt.title(
        f"Feature effects on the {EXPLAINED_CLASS} raw margin\n"
        f"Training sample: {len(sample)} applicants"
    )
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"Explained output: {EXPLAINED_CLASS} raw margin (log-odds scale).")
    print(f"Training rows explained: {len(sample)}; background rows: {len(background)}.")
    print(f"Saved global importance plot: {importance_path}")
    print(f"Saved beeswarm plot: {beeswarm_path}")
    return {"importance": importance_path, "beeswarm": beeswarm_path}


def save_individual_explanation(
    inference_pipeline: Any,
    explainer: shap.TreeExplainer,
    applicant: pd.DataFrame,
    output_path: str | Path = Path("reports")
    / "figures"
    / "latest_applicant_shap.png",
) -> Path:
    """Save one applicant's waterfall plot from their valid raw feature row."""
    explanation = make_individual_explanation(
        inference_pipeline=inference_pipeline,
        explainer=explainer,
        applicant=applicant,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plot_individual_explanation(explanation)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def make_individual_explanation(
    inference_pipeline: Any,
    explainer: shap.TreeExplainer,
    applicant: pd.DataFrame,
) -> shap.Explanation:
    """Return one applicant's validated SHAP explanation."""
    if len(applicant) != 1:
        raise ValueError("An individual SHAP explanation requires exactly one row.")
    transformed = _transform_raw_features(inference_pipeline, applicant)
    return _as_single_output_explanation(explainer, transformed)


def plot_individual_explanation(explanation: shap.Explanation) -> plt.Figure:
    """Build an individual waterfall plot without writing a shared file."""
    try:
        shap.plots.waterfall(explanation[0], max_display=MAX_DISPLAY_FEATURES, show=False)
    except (ValueError, TypeError) as exc:
        raise ValueError("Could not render the individual SHAP waterfall plot.") from exc
    plt.title(f"Applicant explanation: {EXPLAINED_CLASS}")
    plt.tight_layout()
    return plt.gcf()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate global XGBoost SHAP plots using a bounded sample from "
            "the saved training split."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models") / "xgboost.joblib",
        help="Saved XGBoost inference pipeline.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data") / "processed",
        help="Directory containing the existing training split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports") / "figures",
        help="Directory for generated SHAP plots.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=MAX_EXPLANATION_ROWS,
        help=f"Maximum training rows to explain (1-{MAX_EXPLANATION_ROWS}).",
    )
    parser.add_argument(
        "--background-size",
        type=int,
        default=DEFAULT_BACKGROUND_SIZE,
        help=f"Training rows used as SHAP background (default: {DEFAULT_BACKGROUND_SIZE}).",
    )
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(
            f"Fitted XGBoost model not found: {args.model}. "
            "Run `python -m src.train_xgboost` first."
        )
    training_path = args.processed_dir / "X_train.csv"
    if not training_path.is_file():
        raise FileNotFoundError(
            f"Saved training features not found: {training_path}. "
            "Run the preprocessing command first."
        )
    try:
        inference_pipeline = joblib.load(args.model)
        training_features = pd.read_csv(training_path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError("Could not load the saved model or training features.") from exc

    if not isinstance(inference_pipeline, Pipeline):
        raise TypeError("Saved XGBoost artifact must be a scikit-learn Pipeline.")
    if "preprocessor" not in inference_pipeline.named_steps:
        raise ValueError("Saved model Pipeline does not contain its preprocessor.")
    if "classifier" not in inference_pipeline.named_steps:
        raise ValueError("Saved model Pipeline does not contain its classifier.")
    save_global_explanation_plots(
        inference_pipeline=inference_pipeline,
        training_features=training_features,
        output_dir=args.output_dir,
        background_size=args.background_size,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
