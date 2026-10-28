"""Evaluate a fitted credit-risk pipeline on the saved test split."""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.data_loader import FEATURE_COLUMNS, TARGET_COLUMN


CLASS_LABELS = ("good-risk", "bad-risk")
POSITIVE_CLASS = "bad-risk"
DEFAULT_MODEL_PATH = Path("models") / "random_forest.joblib"
DEFAULT_PROCESSED_DIR = Path("data") / "processed"
DEFAULT_REPORT_DIR = Path("reports")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required evaluation file not found: {path}. "
            "Run the preprocessing and training commands first."
        )
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read evaluation file {path}.") from exc


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    model_parameters: dict[str, Any] | None = None,
    model_name: str = "RandomForestClassifier",
    class_mapping: dict[Any, str] | None = None,
    metrics_filename: str = "model_metrics.json",
    figure_filename: str = "confusion_matrix.png",
) -> dict[str, Any]:
    """Calculate test metrics, save JSON and a confusion matrix figure."""
    if X_test.empty or y_test.empty:
        raise ValueError("The test split is empty; evaluation cannot continue.")
    if list(X_test.columns) != list(FEATURE_COLUMNS):
        raise ValueError(
            "Test feature columns do not match the 20 original input attributes."
        )
    if len(X_test) != len(y_test):
        raise ValueError("Test features and labels have different row counts.")
    if y_test.isna().any():
        raise ValueError("Test labels contain missing values.")
    unexpected_labels = sorted(set(y_test.unique()) - set(CLASS_LABELS))
    if unexpected_labels:
        raise ValueError(f"Unexpected test label(s): {unexpected_labels}.")

    predictions = model.predict(X_test)
    if class_mapping is not None:
        try:
            predictions = [class_mapping[label] for label in predictions]
        except KeyError as exc:
            raise ValueError(
                f"The fitted model returned an unmapped class: {exc.args[0]!r}."
            ) from exc

    predicted_labels = set(predictions)
    if predicted_labels - set(CLASS_LABELS):
        raise ValueError("The fitted model returned an unexpected class label.")

    matrix = confusion_matrix(y_test, predictions, labels=list(CLASS_LABELS))
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                pos_label=POSITIVE_CLASS,
                average="binary",
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                pos_label=POSITIVE_CLASS,
                average="binary",
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_test,
                predictions,
                pos_label=POSITIVE_CLASS,
                average="binary",
                zero_division=0,
            )
        ),
        "roc_auc": None,
    }

    roc_auc_reason: str | None = None
    test_classes = set(y_test.unique())
    if len(test_classes) < 2:
        roc_auc_reason = (
            "ROC-AUC was not calculated because the test split contains "
            "only one class."
        )
    elif not hasattr(model, "predict_proba"):
        roc_auc_reason = (
            "ROC-AUC was not calculated because the saved estimator has no "
            "predict_proba method."
        )
    else:
        if hasattr(model, "named_steps") and "classifier" in model.named_steps:
            model_classes = model.named_steps["classifier"].classes_
        else:
            model_classes = model.classes_
        if class_mapping is not None:
            mapped_model_classes = [
                class_mapping.get(label) for label in model_classes
            ]
        else:
            mapped_model_classes = list(model_classes)
        positive_indexes = [
            index
            for index, label in enumerate(mapped_model_classes)
            if label == POSITIVE_CLASS
        ]
        if not positive_indexes:
            raise ValueError(
                f"The fitted model does not contain positive class {POSITIVE_CLASS!r}."
            )
        probabilities = model.predict_proba(X_test)
        positive_scores = probabilities[:, positive_indexes[0]]
        binary_truth = (y_test == POSITIVE_CLASS).astype("int8")
        metrics["roc_auc"] = float(roc_auc_score(binary_truth, positive_scores))

    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_path = output_dir / figure_filename
    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=list(CLASS_LABELS),
    ).plot(ax=axis, cmap="Blues", values_format="d", colorbar=False)
    axis.set_title(f"{model_name} - Credit Risk Confusion Matrix")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)

    report = {
        "model": model_name,
        "positive_class": POSITIVE_CLASS,
        "averaging_method": (
            "binary; precision, recall, and F1 use bad-risk as pos_label. "
            "Accuracy and ROC-AUC are not averaged."
        ),
        "test_rows": int(len(y_test)),
        "test_class_counts": {
            str(label): int(count) for label, count in y_test.value_counts().items()
        },
        "metrics": metrics,
        "roc_auc_unavailable_reason": roc_auc_reason,
        "confusion_matrix": {
            "labels_in_row_and_column_order": list(CLASS_LABELS),
            "rows_are_actual_columns_are_predicted": True,
            "counts": matrix.astype(int).tolist(),
        },
        "model_parameters": model_parameters or {},
        "figure": figure_path.name,
    }
    metrics_path = output_dir / metrics_filename
    metrics_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"{model_name} test-set evaluation (positive class: bad-risk)")
    for metric_name, metric_value in metrics.items():
        formatted_value = (
            "not available" if metric_value is None else f"{metric_value:.4f}"
        )
        print(f"{metric_name}: {formatted_value}")
    if roc_auc_reason:
        print(roc_auc_reason)
    print("Confusion matrix (rows: actual, columns: predicted):")
    print(matrix)
    print(f"Metrics saved to: {metrics_path}")
    print(f"Confusion matrix figure saved to: {figure_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the saved Random Forest on the untouched test split."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Saved end-to-end model path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help=f"Directory containing saved test split (default: {DEFAULT_PROCESSED_DIR})",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help=f"Directory for metrics and figure (default: {DEFAULT_REPORT_DIR})",
    )
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(
            f"Fitted model not found: {args.model}. Run `python -m src.train` first."
        )
    model = joblib.load(args.model)
    model_metadata_path = args.model.with_name(
        f"{args.model.stem}_metadata.json"
    )
    model_parameters: dict[str, Any] = {}
    if model_metadata_path.is_file():
        try:
            model_parameters = json.loads(
                model_metadata_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Could not parse model metadata file {model_metadata_path}."
            ) from exc

    X_test = _read_csv(args.processed_dir / "X_test.csv")
    y_frame = _read_csv(args.processed_dir / "y_test.csv")
    if list(y_frame.columns) != [TARGET_COLUMN]:
        raise ValueError(
            f"Expected the test target file to contain only {TARGET_COLUMN!r}."
        )

    class_mapping: dict[Any, str] | None = None
    model_name = "RandomForestClassifier"
    if (
        hasattr(model, "named_steps")
        and isinstance(model.named_steps.get("classifier"), XGBClassifier)
    ):
        class_mapping = {0: "good-risk", 1: "bad-risk"}
        model_name = "XGBClassifier"

    evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_frame[TARGET_COLUMN],
        report_dir=args.report_dir,
        model_parameters=model_parameters,
        model_name=model_name,
        class_mapping=class_mapping,
    )


if __name__ == "__main__":
    main()
