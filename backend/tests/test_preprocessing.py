"""Tests for the dataset contract and the in-pipeline preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.dataset import FEATURES, TARGET, split_features_and_target
from ml.preprocessing import (
    ImplausibleValueCleaner,
    as_feature_frame,
    build_pipeline,
    model_input_names,
)
from ml.splits import make_splits


def test_target_is_never_a_feature() -> None:
    assert TARGET not in FEATURES


def test_split_features_and_target_excludes_the_label() -> None:
    frame = pd.DataFrame(
        {TARGET: [0, 1], **{feature: [1.0, 2.0] for feature in FEATURES}}
    )
    X, y = split_features_and_target(frame)
    assert list(X.columns) == list(FEATURES)
    assert TARGET not in X.columns
    assert y.tolist() == [0, 1]


def test_cleaner_masks_placeholder_counter_codes() -> None:
    frame = pd.DataFrame(
        {
            "NumberOfTimes90DaysLate": [0, 3, 96, 98],
            "age": [30, 40, 50, 60],
        }
    )
    cleaned = ImplausibleValueCleaner().fit_transform(frame)
    assert cleaned["NumberOfTimes90DaysLate"].tolist()[:2] == [0, 3]
    assert cleaned["NumberOfTimes90DaysLate"].isna().tolist() == [False, False, True, True]


def test_cleaner_masks_impossible_ages() -> None:
    frame = pd.DataFrame({"age": [0, 17, 18, 95]})
    cleaned = ImplausibleValueCleaner().fit_transform(frame)
    assert cleaned["age"].isna().tolist() == [True, True, False, False]


def test_pipeline_imputes_missing_values_without_dropping_rows() -> None:
    frame = as_feature_frame(
        [
            {feature: 1.0 for feature in FEATURES},
            {**{feature: 3.0 for feature in FEATURES}, "MonthlyIncome": None},
        ]
    )
    frame["age"] = [30.0, 40.0]
    pipeline = build_pipeline(RandomForestClassifier(n_estimators=5, random_state=0))
    pipeline.fit(frame, [0, 1])

    transformed = pipeline[:-1].transform(frame)
    assert len(transformed) == 2
    assert not np.isnan(transformed.to_numpy()).any()


def test_indicator_columns_record_what_was_missing() -> None:
    """A 96/98 code must survive as a 'this was missing' signal, not vanish."""
    rows = [{feature: 1.0 for feature in FEATURES} for _ in range(4)]
    rows[0]["NumberOfTimes90DaysLate"] = 98.0
    rows[1]["MonthlyIncome"] = None
    frame = as_feature_frame(rows)
    frame["age"] = [30.0, 40.0, 50.0, 60.0]

    pipeline = build_pipeline(
        RandomForestClassifier(n_estimators=5, random_state=0), add_indicator=True
    )
    pipeline.fit(frame, [1, 1, 0, 0])
    transformed = pipeline[:-1].transform(frame)

    indicators = [name for name in transformed.columns if "missingindicator" in name]
    assert indicators, "expected indicator columns for the masked values"
    assert len(transformed.columns) > len(FEATURES)
    # The 98-coded row is flagged; an ordinary row is not.
    flagged = transformed[indicators].to_numpy()
    assert flagged[0].sum() == 1
    assert flagged[3].sum() == 0


def test_model_input_names_match_the_fitted_width() -> None:
    frame = as_feature_frame([{feature: float(i) for feature in FEATURES} for i in range(4)])
    frame["age"] = [30.0, 40.0, 50.0, 60.0]

    pipeline = build_pipeline(RandomForestClassifier(n_estimators=5, random_state=0))
    pipeline.fit(frame, [0, 1, 0, 1])

    names = model_input_names(pipeline)
    assert len(names) == len(pipeline.named_steps["model"].feature_importances_)


def test_as_feature_frame_orders_columns_like_training() -> None:
    reversed_record = {feature: 1.0 for feature in reversed(FEATURES)}
    assert list(as_feature_frame([reversed_record]).columns) == list(FEATURES)


def test_as_feature_frame_rejects_an_unknown_feature() -> None:
    with pytest.raises(KeyError):
        as_feature_frame([{"loan_amount": 1000.0}])


def test_splits_are_disjoint_and_preserve_the_class_ratio() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(1000, len(FEATURES))), columns=list(FEATURES))
    y = pd.Series([1] * 100 + [0] * 900)

    splits = make_splits(X, y)

    assert len(splits.X_train) == 600
    assert len(splits.X_validation) == len(splits.X_test) == 200
    indices = [set(part.index) for part in (splits.X_train, splits.X_validation, splits.X_test)]
    assert not indices[0] & indices[1] and not indices[0] & indices[2] and not indices[1] & indices[2]
    for labels in (splits.y_train, splits.y_validation, splits.y_test):
        assert labels.mean() == pytest.approx(0.1, abs=0.005)


def test_splits_are_reproducible_for_a_fixed_seed() -> None:
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(200, len(FEATURES))), columns=list(FEATURES))
    y = pd.Series([1] * 20 + [0] * 180)

    first = make_splits(X, y, seed=7)
    second = make_splits(X, y, seed=7)

    assert first.X_test.index.tolist() == second.X_test.index.tolist()
