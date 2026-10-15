"""Tests for the multi-split evaluation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.dataset import FEATURES
from ml.metrics import classification_metrics, evaluate_split
from ml.preprocessing import build_pipeline
from ml.splits import make_splits


@pytest.fixture(scope="module")
def fitted_on_train():
    """A small forest fitted on the train split of a synthetic dataset.

    Synthetic rather than the real CSV so the test stays fast and runs without
    the Kaggle download.
    """
    rng = np.random.default_rng(0)
    rows = 1200
    X = pd.DataFrame(rng.normal(size=(rows, len(FEATURES))), columns=list(FEATURES))
    # Plausible ages, so the cleaner does not mask the whole column as implausible.
    X["age"] = rng.integers(21, 80, size=rows)
    # Make the label genuinely learnable from the first feature.
    signal = X[FEATURES[0]] + rng.normal(scale=0.5, size=rows)
    y = pd.Series((signal > np.quantile(signal, 0.9)).astype(int))

    splits = make_splits(X, y)
    pipeline = build_pipeline(
        RandomForestClassifier(n_estimators=40, random_state=0, n_jobs=-1)
    )
    pipeline.fit(splits.X_train, splits.y_train)
    return pipeline, splits


def test_evaluate_split_matches_manual_scoring(fitted_on_train) -> None:
    pipeline, splits = fitted_on_train

    from ml.metrics import positive_class_proba

    direct = classification_metrics(
        splits.y_test, positive_class_proba(pipeline, splits.X_test)
    )

    assert evaluate_split(pipeline, splits.X_test, splits.y_test) == direct


def test_every_split_can_be_scored(fitted_on_train) -> None:
    pipeline, splits = fitted_on_train

    scores = {
        "train": evaluate_split(pipeline, splits.X_train, splits.y_train),
        "validation": evaluate_split(pipeline, splits.X_validation, splits.y_validation),
        "test": evaluate_split(pipeline, splits.X_test, splits.y_test),
    }

    for metrics in scores.values():
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_in_sample_score_is_at_least_held_out_score(fitted_on_train) -> None:
    """A full-depth-ish forest should not do worse on data it was fitted on.

    This is what the generalisation-gap report measures; if it were ever
    negative by a wide margin, something would be wrong with the splits.
    """
    pipeline, splits = fitted_on_train

    train = evaluate_split(pipeline, splits.X_train, splits.y_train)
    test = evaluate_split(pipeline, splits.X_test, splits.y_test)

    assert train["roc_auc"] >= test["roc_auc"]


def test_train_and_validation_splits_combine_to_the_expected_size(fitted_on_train) -> None:
    _, splits = fitted_on_train

    combined = splits.X_train_and_validation

    assert len(combined) == len(splits.X_train) + len(splits.X_validation)
    assert len(splits.y_train_and_validation) == len(combined)
    # The test split must not leak into the refitting set.
    assert set(combined.index).isdisjoint(set(splits.X_test.index))
