"""Deterministic train/validation/test splitting.

``train.py`` and ``evaluate.py`` both call :func:`make_splits`, so with the same
seed they see byte-identical splits and the test set stays genuinely held out.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
VALIDATION_FRACTION = 0.2
TEST_FRACTION = 0.2


@dataclass(frozen=True)
class Splits:
    """Stratified 60/20/20 split of the labeled data."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series

    @property
    def X_train_and_validation(self) -> pd.DataFrame:
        """Training rows plus validation rows, for refitting the chosen config."""
        return pd.concat([self.X_train, self.X_validation])

    @property
    def y_train_and_validation(self) -> pd.Series:
        return pd.concat([self.y_train, self.y_validation])


def make_splits(X: pd.DataFrame, y: pd.Series, seed: int = RANDOM_SEED) -> Splits:  # noqa: N803
    """Split into train/validation/test, preserving the class ratio in each part.

    Stratification matters here: positives are only ~6.7% of rows, so an
    unstratified split could leave the parts with noticeably different base
    rates and make the metrics incomparable.
    """
    X_remaining, X_test, y_remaining, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, stratify=y, random_state=seed
    )
    validation_share = VALIDATION_FRACTION / (1 - TEST_FRACTION)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_remaining,
        y_remaining,
        test_size=validation_share,
        stratify=y_remaining,
        random_state=seed,
    )
    return Splits(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )
