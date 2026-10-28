"""Tests that use the verified dataset and trained local project artifacts."""

import pandas as pd
import pytest

from src.app_services import DATA_PATH, MODEL_PATH, PROCESSED_DIR
from src.data_loader import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_german_credit_data,
)
from src.explain import make_individual_explanation, make_tree_explainer
from src.prediction import (
    load_xgboost_pipeline,
    predict_credit_risk,
    validate_applicant,
)
from src.preprocessing import prepare_preprocessing_outputs


def test_verified_dataset_has_original_schema_and_target_mapping():
    dataset = load_german_credit_data(DATA_PATH)

    assert len(dataset) == 1000
    assert list(dataset.columns) == [*FEATURE_COLUMNS, TARGET_COLUMN]
    assert dataset[TARGET_COLUMN].value_counts().to_dict() == {
        "good-risk": 700,
        "bad-risk": 300,
    }
    assert not dataset.isna().any().any()


def test_preprocessing_builds_reproducible_split_and_named_features(tmp_path):
    outputs = prepare_preprocessing_outputs(
        data_path=DATA_PATH,
        output_dir=tmp_path / "processed",
        test_size=0.2,
        random_state=42,
    )
    train = pd.read_csv(outputs["X_train"])
    test = pd.read_csv(outputs["X_test"])
    encoded_train = pd.read_csv(outputs["X_train_transformed"])
    encoded_test = pd.read_csv(outputs["X_test_transformed"])

    assert len(train) == 800
    assert len(test) == 200
    assert list(train.columns) == list(FEATURE_COLUMNS)
    assert encoded_train.columns.tolist() == encoded_test.columns.tolist()
    assert "credit_risk" not in encoded_train.columns


def test_prediction_uses_saved_model_on_a_valid_actual_applicant():
    model = load_xgboost_pipeline(MODEL_PATH)
    training = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    # Use a held-out record as a realistic valid inference row, not a row the
    # classifier saw while fitting.
    applicant = pd.read_csv(PROCESSED_DIR / "X_test.csv").iloc[[0]].copy()
    ranges = {
        column: (int(training[column].min()), int(training[column].max()))
        for column in (
            "duration_months",
            "credit_amount",
            "installment_rate",
            "residence_since",
            "age_years",
            "existing_credits",
            "num_dependents",
        )
    }

    result = predict_credit_risk(model, applicant, ranges)

    assert result.risk_label in {"good-risk", "bad-risk"}
    assert result.positive_class == "bad-risk"
    assert 0.0 <= result.bad_risk_probability <= 1.0
    assert 0.0 <= result.good_risk_probability <= 1.0
    assert result.bad_risk_probability + result.good_risk_probability == pytest.approx(
        1.0, abs=1e-5
    )


def test_prediction_rejects_missing_or_misordered_features():
    training = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    applicant = training.iloc[[0]].copy()

    with pytest.raises(ValueError, match="feature names or order"):
        validate_applicant(applicant.loc[:, list(reversed(FEATURE_COLUMNS))])

    applicant.iloc[0, 0] = None
    with pytest.raises(ValueError, match="incomplete"):
        validate_applicant(applicant)


def test_real_model_can_explain_one_actual_training_row():
    model = load_xgboost_pipeline(MODEL_PATH)
    training = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    applicant = training.iloc[[0]].copy()
    transformed_background = model.named_steps["preprocessor"].transform(
        training.sample(n=20, random_state=42)
    )
    if hasattr(transformed_background, "toarray"):
        transformed_background = transformed_background.toarray()
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    background = pd.DataFrame(transformed_background, columns=feature_names)

    explainer = make_tree_explainer(
        model.named_steps["classifier"], background
    )
    explanation = make_individual_explanation(model, explainer, applicant)

    assert explanation.values.shape == (1, len(feature_names))
    assert len(explanation.feature_names) == len(feature_names)
