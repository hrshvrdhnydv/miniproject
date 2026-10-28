"""Train the reproducible Random Forest baseline for German Credit."""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.data_loader import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluate import evaluate_model


DEFAULT_PROCESSED_DIR = Path("data") / "processed"
DEFAULT_MODEL_PATH = Path("models") / "random_forest.joblib"
DEFAULT_REPORT_DIR = Path("reports")
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_ESTIMATORS = 300
CLASS_LABELS = {"good-risk", "bad-risk"}


def _required_file(path: Path, preprocessing_command: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required preprocessing output is missing: {path}. "
            f"Create the existing split first with: {preprocessing_command}"
        )


def _read_csv(path: Path, preprocessing_command: str) -> pd.DataFrame:
    _required_file(path, preprocessing_command)
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read preprocessing output {path}.") from exc


def _dense_array(values: Any) -> np.ndarray:
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values)


def train_random_forest(
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
) -> dict[str, Any]:
    """Fit on the saved training rows and evaluate on the saved test rows."""
    if random_state < 0:
        raise ValueError("random_state must be a non-negative integer.")
    if n_estimators < 1:
        raise ValueError("n_estimators must be at least 1.")

    processed_path = Path(processed_dir)
    preparation_command = (
        "python -m src.preprocessing --input data/raw/german.data "
        f"--output-dir {processed_path}"
    )
    X_train = _read_csv(
        processed_path / "X_train.csv", preparation_command
    )
    X_test = _read_csv(processed_path / "X_test.csv", preparation_command)
    y_train_frame = _read_csv(
        processed_path / "y_train.csv", preparation_command
    )
    y_test_frame = _read_csv(processed_path / "y_test.csv", preparation_command)
    X_train_transformed = _read_csv(
        processed_path / "X_train_transformed.csv", preparation_command
    )
    X_test_transformed = _read_csv(
        processed_path / "X_test_transformed.csv", preparation_command
    )
    feature_names_path = processed_path / "feature_names.json"
    _required_file(feature_names_path, preparation_command)
    preprocessor_path = processed_path / "preprocessor.joblib"
    _required_file(preprocessor_path, preparation_command)
    metadata_path = processed_path / "preprocessing_metadata.json"
    _required_file(metadata_path, preparation_command)

    if list(X_train.columns) != list(FEATURE_COLUMNS):
        raise ValueError(
            "Saved training features do not match the 20 original attributes."
        )
    if list(X_test.columns) != list(FEATURE_COLUMNS):
        raise ValueError(
            "Saved test features do not match the 20 original attributes."
        )
    if list(y_train_frame.columns) != [TARGET_COLUMN]:
        raise ValueError("Training labels must contain only the credit_risk target.")
    if list(y_test_frame.columns) != [TARGET_COLUMN]:
        raise ValueError("Test labels must contain only the credit_risk target.")
    if len(X_train) != len(y_train_frame) or len(X_test) != len(y_test_frame):
        raise ValueError("Saved feature and label files have inconsistent row counts.")
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Saved train/test split contains an empty partition.")

    y_train = y_train_frame[TARGET_COLUMN]
    y_test = y_test_frame[TARGET_COLUMN]
    if y_train.isna().any() or y_test.isna().any():
        raise ValueError("Saved train/test labels contain missing values.")
    if set(y_train.unique()) != CLASS_LABELS:
        raise ValueError("Training labels must contain good-risk and bad-risk.")
    if set(y_test.unique()) - CLASS_LABELS:
        raise ValueError("Test labels contain an unexpected risk class.")

    feature_names = json.loads(feature_names_path.read_text(encoding="utf-8"))
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError(f"Invalid transformed feature-name file: {feature_names_path}")
    if list(X_train_transformed.columns) != feature_names:
        raise ValueError("Training transformed columns do not match feature_names.json.")
    if list(X_test_transformed.columns) != feature_names:
        raise ValueError("Test transformed columns do not match feature_names.json.")
    if len(X_train_transformed) != len(X_train):
        raise ValueError("Transformed training rows do not match raw training rows.")
    if len(X_test_transformed) != len(X_test):
        raise ValueError("Transformed test rows do not match raw test rows.")

    preprocessor_pipeline = joblib.load(preprocessor_path)
    if not isinstance(preprocessor_pipeline, Pipeline):
        raise ValueError(
            f"Expected the saved scikit-learn preprocessing Pipeline at "
            f"{preprocessor_path}."
        )
    if "preprocessor" not in preprocessor_pipeline.named_steps:
        raise ValueError("Saved preprocessing Pipeline has no preprocessor step.")

    # Confirm that the saved transformer reproduces the saved encoded split.
    # It is loaded as-is and is never fitted on the test data here.
    reproduced_train = _dense_array(preprocessor_pipeline.transform(X_train))
    reproduced_test = _dense_array(preprocessor_pipeline.transform(X_test))
    if not np.allclose(
        reproduced_train, X_train_transformed.to_numpy(), rtol=1e-7, atol=1e-9
    ):
        raise ValueError(
            "Saved preprocessor output differs from X_train_transformed.csv. "
            "Regenerate preprocessing outputs before training."
        )
    if not np.allclose(
        reproduced_test, X_test_transformed.to_numpy(), rtol=1e-7, atol=1e-9
    ):
        raise ValueError(
            "Saved preprocessor output differs from X_test_transformed.csv. "
            "Regenerate preprocessing outputs before training."
        )

    if not np.isfinite(X_train_transformed.to_numpy(dtype=float)).all():
        raise ValueError("Transformed training features contain non-finite values.")
    if not np.isfinite(X_test_transformed.to_numpy(dtype=float)).all():
        raise ValueError("Transformed test features contain non-finite values.")

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    # class_weight is calculated from y_train during fit; no test rows are used.
    classifier.fit(X_train_transformed.to_numpy(), y_train.to_numpy())

    # This deployable pipeline contains the already-fitted preprocessing step
    # from the preprocessing stage and the fitted classifier.
    inference_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor_pipeline.named_steps["preprocessor"]),
            ("classifier", classifier),
        ]
    )

    model_output_path = Path(model_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(inference_pipeline, model_output_path)

    split_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_parameters = {
        "n_estimators": n_estimators,
        "random_state": random_state,
        "class_weight": "balanced",
        "n_jobs": -1,
        "split_random_state": split_metadata.get("random_state"),
        "training_class_counts": {
            str(label): int(count) for label, count in y_train.value_counts().items()
        },
        "preprocessor_source": str(preprocessor_path),
        "preprocessor_fit_scope": split_metadata.get("fit_scope"),
        "transformed_feature_count": len(feature_names),
    }
    model_metadata_path = model_output_path.with_name(
        f"{model_output_path.stem}_metadata.json"
    )
    model_metadata_path.write_text(
        json.dumps(model_parameters, indent=2) + "\n",
        encoding="utf-8",
    )

    return evaluate_model(
        model=inference_pipeline,
        X_test=X_test,
        y_test=y_test,
        report_dir=report_dir,
        model_parameters=model_parameters,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the Random Forest baseline using the saved preprocessing "
            "outputs, then evaluate on the saved test split."
        )
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help=f"Saved preprocessing split directory (default: {DEFAULT_PROCESSED_DIR})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path for the fitted inference pipeline (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help=f"Directory for test metrics and figure (default: {DEFAULT_REPORT_DIR})",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for the Random Forest (default: 42)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=DEFAULT_N_ESTIMATORS,
        help="Number of trees in the forest (default: 300)",
    )
    args = parser.parse_args()

    train_random_forest(
        processed_dir=args.processed_dir,
        model_path=args.model,
        report_dir=args.report_dir,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
    )


if __name__ == "__main__":
    main()
