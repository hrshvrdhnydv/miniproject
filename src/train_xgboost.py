"""Tune and train XGBoost on the saved German Credit training split."""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_loader import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluate import evaluate_model


DEFAULT_PROCESSED_DIR = Path("data") / "processed"
DEFAULT_MODEL_PATH = Path("models") / "xgboost.joblib"
DEFAULT_REPORT_DIR = Path("reports") / "xgboost"
DEFAULT_RANDOM_STATE = 42
CLASS_TO_CODE = {"good-risk": 0, "bad-risk": 1}
CODE_TO_CLASS = {code: label for label, code in CLASS_TO_CODE.items()}


def _read_csv(path: Path, preparation_command: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required preprocessing file not found: {path}. "
            f"Run {preparation_command} first."
        )
    try:
        return pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read preprocessing output {path}.") from exc


def _dense_array(values: Any) -> np.ndarray:
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values)


def train_xgboost(
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    random_state: int = DEFAULT_RANDOM_STATE,
    cv_folds: int = 3,
) -> dict[str, Any]:
    """Cross-validate a small parameter grid using training data only."""
    if random_state < 0:
        raise ValueError("random_state must be a non-negative integer.")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")

    processed_path = Path(processed_dir)
    preparation_command = (
        "python -m src.preprocessing --input data/raw/german.data "
        f"--output-dir {processed_path}"
    )
    X_train = _read_csv(processed_path / "X_train.csv", preparation_command)
    X_test = _read_csv(processed_path / "X_test.csv", preparation_command)
    y_train_frame = _read_csv(processed_path / "y_train.csv", preparation_command)
    y_test_frame = _read_csv(processed_path / "y_test.csv", preparation_command)
    metadata_path = processed_path / "preprocessing_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Preprocessing metadata not found: {metadata_path}. "
            f"Run {preparation_command} first."
        )
    preprocessor_path = processed_path / "preprocessor.joblib"
    if not preprocessor_path.is_file():
        raise FileNotFoundError(
            f"Fitted preprocessing pipeline not found: {preprocessor_path}. "
            f"Run {preparation_command} first."
        )

    if list(X_train.columns) != list(FEATURE_COLUMNS):
        raise ValueError("Saved training data does not contain the verified 20 inputs.")
    if list(X_test.columns) != list(FEATURE_COLUMNS):
        raise ValueError("Saved test data does not contain the verified 20 inputs.")
    if list(y_train_frame.columns) != [TARGET_COLUMN]:
        raise ValueError(f"Training labels must contain only {TARGET_COLUMN!r}.")
    if list(y_test_frame.columns) != [TARGET_COLUMN]:
        raise ValueError(f"Test labels must contain only {TARGET_COLUMN!r}.")
    if len(X_train) != len(y_train_frame) or len(X_test) != len(y_test_frame):
        raise ValueError("Saved feature and target files have different row counts.")

    y_train_text = y_train_frame[TARGET_COLUMN]
    y_test_text = y_test_frame[TARGET_COLUMN]
    if set(y_train_text.unique()) != set(CLASS_TO_CODE):
        raise ValueError("Training labels must be good-risk and bad-risk.")
    if set(y_test_text.unique()) - set(CLASS_TO_CODE):
        raise ValueError("Test labels contain a class not documented by UCI.")

    split_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if split_metadata.get("fit_scope") != "training data only":
        raise ValueError(
            "Preprocessing metadata does not confirm train-only fitting. "
            "Regenerate the preprocessing split before tuning."
        )

    preprocessing_pipeline = joblib.load(preprocessor_path)
    if not isinstance(preprocessing_pipeline, Pipeline):
        raise ValueError(
            f"Expected a saved sklearn Pipeline in {preprocessor_path}."
        )
    if "preprocessor" not in preprocessing_pipeline.named_steps:
        raise ValueError("The saved Pipeline has no named 'preprocessor' step.")

    y_train = y_train_text.map(CLASS_TO_CODE).astype("int8")

    # Put the existing ColumnTransformer into the CV pipeline. GridSearchCV
    # clones it and fits it separately inside each training fold, preventing
    # the fold's validation rows from influencing categorical encoding.
    estimator = Pipeline(
        steps=[
            ("preprocessor", preprocessing_pipeline.named_steps["preprocessor"]),
            (
                "classifier",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    n_estimators=200,
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )
    search = GridSearchCV(
        estimator=estimator,
        param_grid={
            "classifier__max_depth": [2, 3],
            "classifier__learning_rate": [0.05, 0.1],
        },
        scoring="roc_auc",
        cv=StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state,
        ),
        refit=True,
        n_jobs=1,
        error_score="raise",
        return_train_score=False,
    )
    # Only the existing training split is supplied to cross-validation.
    search.fit(X_train, y_train)
    fitted_pipeline = search.best_estimator_
    classifier = fitted_pipeline.named_steps["classifier"]
    if not isinstance(classifier, XGBClassifier):
        raise TypeError("The tuned final pipeline does not contain XGBClassifier.")
    if list(classifier.classes_) != [0, 1]:
        raise ValueError(
            f"Expected binary output classes [0, 1], found {classifier.classes_!r}."
        )

    transformed_names = (
        fitted_pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    )
    transformed_training = fitted_pipeline.named_steps["preprocessor"].transform(
        X_train
    )
    transformed_test = fitted_pipeline.named_steps["preprocessor"].transform(X_test)
    if transformed_training.shape[1] != len(transformed_names):
        raise ValueError("XGBoost transformed feature names do not match its input.")
    if transformed_test.shape[1] != len(transformed_names):
        raise ValueError("Transformed test features do not match the model representation.")
    if not np.isfinite(_dense_array(transformed_training)).all():
        raise ValueError("Transformed training data contains non-finite values.")

    model_output_path = Path(model_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_pipeline, model_output_path)

    model_parameters: dict[str, Any] = {
        "model": "XGBClassifier",
        "objective": "binary:logistic",
        "target_encoding": CLASS_TO_CODE,
        "prediction_labels": CODE_TO_CLASS,
        "positive_class": "bad-risk",
        "binary_positive_code": 1,
        "random_state": random_state,
        "cv_random_state": random_state,
        "cv_folds": cv_folds,
        "cv_scoring": "roc_auc",
        "cv_best_score_training_only": float(search.best_score_),
        "best_params": search.best_params_,
        "n_estimators": int(classifier.n_estimators),
        "training_class_counts": {
            str(label): int(count) for label, count in y_train_text.value_counts().items()
        },
        "split_random_state": split_metadata.get("random_state"),
        "preprocessor_fit_scope": "refit on full saved training split after CV",
        "input_feature_count": len(FEATURE_COLUMNS),
        "transformed_feature_count": len(transformed_names),
        "metric_definitions": {
            "precision_recall_f1": (
                "binary averaging, bad-risk is the positive class"
            ),
            "accuracy_roc_auc": "single scalar metrics; no class averaging",
            "roc_auc": "uses predicted probability for bad-risk",
            "confusion_matrix": "rows are actual; columns are predicted",
        },
        "xgboost_version": xgboost.__version__,
        "scikit_learn_version": sklearn.__version__,
    }
    metadata_output_path = model_output_path.with_name(
        f"{model_output_path.stem}_metadata.json"
    )
    metadata_output_path.write_text(
        json.dumps(model_parameters, indent=2) + "\n", encoding="utf-8"
    )

    metrics = evaluate_model(
        model=fitted_pipeline,
        X_test=X_test,
        y_test=y_test_text,
        report_dir=report_dir,
        model_parameters=model_parameters,
        model_name="XGBClassifier",
        class_mapping=CODE_TO_CLASS,
    )

    # Store model-specific experiment output separately to preserve the
    # existing Random Forest baseline report.
    random_forest_metrics_path = Path("reports") / "model_metrics.json"
    if random_forest_metrics_path.is_file():
        rf_report = json.loads(
            random_forest_metrics_path.read_text(encoding="utf-8")
        )
        comparison = {
            "positive_class": "bad-risk",
            "metric_definitions": model_parameters["metric_definitions"],
            "models": {
                "RandomForestClassifier": rf_report.get("metrics", {}),
                "XGBClassifier": metrics["metrics"],
            },
            "difference_xgboost_minus_random_forest": {
                metric: (
                    None
                    if metrics["metrics"].get(metric) is None
                    or rf_report.get("metrics", {}).get(metric) is None
                    else float(
                        metrics["metrics"][metric]
                        - rf_report["metrics"][metric]
                    )
                )
                for metric in ("accuracy", "precision", "recall", "f1_score", "roc_auc")
            },
            "interpretation": (
                "Differences are descriptive results from this saved test split; "
                "they do not establish general superiority."
            ),
        }
        Path("reports").mkdir(parents=True, exist_ok=True)
        Path("reports/model_comparison.json").write_text(
            json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
        )

    print(f"CV best ROC-AUC (training folds only): {search.best_score_:.4f}")
    print(f"Best CV parameters: {search.best_params_}")
    print(f"Saved end-to-end XGBoost pipeline to: {model_output_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tune and train XGBoost on the saved training split, then evaluate "
            "once on the held-out test split."
        )
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--cv-folds", type=int, default=3)
    args = parser.parse_args()
    train_xgboost(
        processed_dir=args.processed_dir,
        model_path=args.model,
        report_dir=args.report_dir,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
    )


if __name__ == "__main__":
    main()
