"""Load and validate the original UCI Statlog German Credit dataset.

The original ``german.data`` file is headerless and whitespace-delimited. It
contains 20 input attributes followed by a class code. The official UCI
documentation defines class code 1 as Good and class code 2 as Bad.
"""

from pathlib import Path

import pandas as pd


EXPECTED_ROW_COUNT = 1000
TARGET_COLUMN = "credit_risk"
RAW_TARGET_COLUMN = "target_code"

# Attribute names follow the order in the official german.doc file.
FEATURE_COLUMNS = (
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment_since",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age_years",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker",
)

ALL_COLUMNS = (*FEATURE_COLUMNS, RAW_TARGET_COLUMN)

CATEGORICAL_FEATURES = (
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment_since",
    "personal_status_sex",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
)

NUMERICAL_FEATURES = (
    "duration_months",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age_years",
    "existing_credits",
    "num_dependents",
)

# These are the categorical codes documented by UCI. A documented category
# does not have to occur in every particular copy of the dataset.
VALID_CATEGORICAL_VALUES = {
    "checking_status": {"A11", "A12", "A13", "A14"},
    "credit_history": {"A30", "A31", "A32", "A33", "A34"},
    "purpose": {
        "A40",
        "A41",
        "A42",
        "A43",
        "A44",
        "A45",
        "A46",
        "A47",
        "A48",
        "A49",
        "A410",
    },
    "savings_status": {"A61", "A62", "A63", "A64", "A65"},
    "employment_since": {"A71", "A72", "A73", "A74", "A75"},
    "personal_status_sex": {"A91", "A92", "A93", "A94", "A95"},
    "other_debtors": {"A101", "A102", "A103"},
    "property": {"A121", "A122", "A123", "A124"},
    "other_installment_plans": {"A141", "A142", "A143"},
    "housing": {"A151", "A152", "A153"},
    "job": {"A171", "A172", "A173", "A174"},
    "telephone": {"A191", "A192"},
    "foreign_worker": {"A201", "A202"},
}

# Keep the original UCI class meaning: 1 is good credit risk; 2 is bad.
TARGET_LABELS = {1: "good-risk", 2: "bad-risk"}


def load_german_credit_data(data_path: str | Path) -> pd.DataFrame:
    """Read the original UCI file and return validated, named columns.

    The returned DataFrame contains the 20 input features and a human-readable
    ``credit_risk`` target. The original numeric target code is removed so it
    cannot accidentally be included as a model input.
    """
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"German Credit dataset file not found: {path}")

    try:
        data = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=list(ALL_COLUMNS),
            dtype=str,
            encoding="ascii",
            engine="python",
            on_bad_lines="error",
        )
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(
            f"Could not read {path} as the original whitespace-delimited "
            "UCI german.data format."
        ) from exc

    if data.shape[1] != len(ALL_COLUMNS):
        raise ValueError(
            f"Expected {len(ALL_COLUMNS)} fields per record "
            f"(20 inputs plus target), found {data.shape[1]} columns."
        )
    if len(data) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ROW_COUNT} dataset records, found {len(data)}."
        )

    missing_count = int(data.isna().sum().sum())
    if missing_count:
        missing_columns = data.columns[data.isna().any()].tolist()
        raise ValueError(
            f"Dataset contains {missing_count} missing value(s) in "
            f"{missing_columns}."
        )

    for column, valid_values in VALID_CATEGORICAL_VALUES.items():
        observed_values = set(data[column].unique())
        invalid_values = sorted(observed_values - valid_values)
        if invalid_values:
            raise ValueError(
                f"Unexpected category code(s) in {column}: {invalid_values}."
            )

    for column in NUMERICAL_FEATURES:
        try:
            numeric_values = pd.to_numeric(data[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Numerical attribute {column!r} contains non-numeric data."
            ) from exc

        if not numeric_values.map(lambda value: float(value).is_integer()).all():
            raise ValueError(
                f"Numerical attribute {column!r} contains non-integer values."
            )
        data[column] = numeric_values.astype("int64")

    try:
        target_codes = pd.to_numeric(data[RAW_TARGET_COLUMN], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("The target column contains non-numeric class codes.") from exc

    if not target_codes.map(lambda value: float(value).is_integer()).all():
        raise ValueError("The target column contains non-integer class codes.")

    target_codes = target_codes.astype("int64")
    invalid_target_codes = sorted(set(target_codes.unique()) - set(TARGET_LABELS))
    if invalid_target_codes:
        raise ValueError(f"Unexpected target code(s): {invalid_target_codes}.")

    duplicate_count = int(data.duplicated().sum())
    if duplicate_count:
        raise ValueError(
            f"Dataset contains {duplicate_count} duplicate complete row(s). "
            "Resolve duplicates before splitting to prevent train/test leakage."
        )

    conflicting_targets = (
        data.groupby(list(FEATURE_COLUMNS), dropna=False)[RAW_TARGET_COLUMN]
        .nunique()
        .gt(1)
    )
    if conflicting_targets.any():
        raise ValueError(
            "Identical feature rows have conflicting target codes; "
            "review the source data before preprocessing."
        )

    data[TARGET_COLUMN] = target_codes.map(TARGET_LABELS)
    return data.drop(columns=[RAW_TARGET_COLUMN])
