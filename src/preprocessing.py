"""Create reproducible train/test preprocessing outputs for German Credit."""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data_loader import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    TARGET_LABELS,
    load_german_credit_data,
)


DEFAULT_DATA_PATH = Path("data") / "raw" / "german.data"
DEFAULT_OUTPUT_DIR = Path("data") / "processed"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42


def create_preprocessing_pipeline() -> Pipeline:
    """Build the unfitted transformations used for the 20 input attributes."""
    column_transformer = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_FEATURES),
            ),
            ("numerical", "passthrough", list(NUMERICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline(steps=[("preprocessor", column_transformer)])


def _dense_array(values: Any) -> np.ndarray:
    """Convert sparse or dense transformer output to a NumPy array for CSV."""
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values)


def prepare_preprocessing_outputs(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Path]:
    """Split, fit preprocessing on training data, and save reproducible files."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")
    if not isinstance(random_state, int) or random_state < 0:
        raise ValueError("random_state must be a non-negative integer.")

    data = load_german_credit_data(data_path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError("Validated dataset is missing its target column.")

    # Keep the target separate before splitting; it is never passed to the
    # feature transformer and therefore cannot leak into encoded features.
    X = data.loc[:, list(FEATURE_COLUMNS)].copy()
    y = data[TARGET_COLUMN].copy()

    if TARGET_COLUMN in X.columns or len(X.columns) != len(FEATURE_COLUMNS):
        raise ValueError("Target leakage check failed: invalid feature columns.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline = create_preprocessing_pipeline()
    # Fit only on training rows. The test data is transformed after fitting.
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)
    feature_names = pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out().tolist()

    train_matrix = _dense_array(X_train_transformed)
    test_matrix = _dense_array(X_test_transformed)
    if train_matrix.shape[1] != len(feature_names):
        raise ValueError("Transformed training data and feature names do not match.")
    if test_matrix.shape[1] != len(feature_names):
        raise ValueError("Transformed test data and feature names do not match.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_path / "X_train.csv", index=False)
    X_test.to_csv(output_path / "X_test.csv", index=False)
    y_train.rename(TARGET_COLUMN).to_csv(
        output_path / "y_train.csv", index=False
    )
    y_test.rename(TARGET_COLUMN).to_csv(output_path / "y_test.csv", index=False)
    pd.DataFrame(train_matrix, columns=feature_names).to_csv(
        output_path / "X_train_transformed.csv", index=False
    )
    pd.DataFrame(test_matrix, columns=feature_names).to_csv(
        output_path / "X_test_transformed.csv", index=False
    )

    pipeline_path = output_path / "preprocessor.joblib"
    joblib.dump(pipeline, pipeline_path)

    mapping_path = output_path / "target_mapping.json"
    mapping_document = {
        "target_column": TARGET_COLUMN,
        "source_column_position": 21,
        "source_codes": {str(code): label for code, label in TARGET_LABELS.items()},
        "source_meaning": {
            "1": "Good credit risk",
            "2": "Bad credit risk",
        },
        "source": "UCI Statlog (German Credit Data), original german.data",
        "documentation": "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
    }
    mapping_path.write_text(
        json.dumps(mapping_document, indent=2) + "\n", encoding="utf-8"
    )

    feature_names_path = output_path / "feature_names.json"
    feature_names_path.write_text(
        json.dumps(feature_names, indent=2) + "\n", encoding="utf-8"
    )

    summary_path = output_path / "preprocessing_metadata.json"
    summary = {
        "source_file": str(Path(data_path)),
        "input_record_count": len(data),
        "input_feature_count": len(FEATURE_COLUMNS),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "test_size": test_size,
        "random_state": random_state,
        "target_column": TARGET_COLUMN,
        "target_mapping_file": mapping_path.name,
        "target_counts": {
            str(label): int(count) for label, count in y.value_counts().items()
        },
        "categorical_features": list(CATEGORICAL_FEATURES),
        "numerical_features": list(NUMERICAL_FEATURES),
        "transformed_feature_count": len(feature_names),
        "transformed_feature_names_file": feature_names_path.name,
        "preprocessor_file": pipeline_path.name,
        "fit_scope": "training data only",
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    return {
        "X_train": output_path / "X_train.csv",
        "X_test": output_path / "X_test.csv",
        "y_train": output_path / "y_train.csv",
        "y_test": output_path / "y_test.csv",
        "X_train_transformed": output_path / "X_train_transformed.csv",
        "X_test_transformed": output_path / "X_test_transformed.csv",
        "preprocessor": pipeline_path,
        "target_mapping": mapping_path,
        "feature_names": feature_names_path,
        "metadata": summary_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the original UCI German Credit dataset, split it, and "
            "fit preprocessing on the training data only."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to original german.data (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for preprocessing outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction of records for the test split (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Non-negative random seed for a reproducible split (default: 42)",
    )
    args = parser.parse_args()

    output_files = prepare_preprocessing_outputs(
        data_path=args.input,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print("Preprocessing completed successfully.")
    for name, path in output_files.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
