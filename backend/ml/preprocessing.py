"""Preprocessing steps shared by training and inference.

Everything here lives inside the scikit-learn ``Pipeline`` that gets saved with
``joblib``, so inference applies exactly the transformations that were fitted on
the training split -- nothing is re-implemented on the serving side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ml.dataset import DELINQUENCY_COUNTERS, FEATURES, PLACEHOLDER_CODES

#: Minimum plausible borrower age. The raw data holds one record with age 0,
#: which cannot be a credit applicant, so it is treated as missing.
MIN_PLAUSIBLE_AGE = 18


class ImplausibleValueCleaner(BaseEstimator, TransformerMixin):
    """Replace documented placeholder/impossible values with ``NaN``.

    Two cases, both verified against the raw file (see docs/methodology.md):

    * The three late-payment counters use 96 and 98 as codes rather than counts.
      A borrower cannot have been 30-59 days late 98 times in two years.
    * ``age`` is 0 for a single record, which is not a possible applicant age.

    Marking them missing hands them to the imputer instead of discarding the
    rows, which keeps every label in the dataset.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "ImplausibleValueCleaner":  # noqa: N803
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        cleaned = X.copy()
        for column in DELINQUENCY_COUNTERS:
            if column in cleaned.columns:
                cleaned[column] = cleaned[column].mask(
                    cleaned[column].isin(PLACEHOLDER_CODES)
                )
        if "age" in cleaned.columns:
            cleaned["age"] = cleaned["age"].mask(cleaned["age"] < MIN_PLAUSIBLE_AGE)
        return cleaned

    def get_feature_names_out(self, input_features=None):
        return np.asarray(
            input_features if input_features is not None else self.feature_names_in_,
            dtype=object,
        )


def build_pipeline(model, add_indicator: bool = False) -> Pipeline:
    """Wrap ``model`` in the project's preprocessing steps.

    All ten predictors are numeric, so a single median imputer covers them; a
    Random Forest needs no scaling or encoding.

    ``add_indicator`` appends a binary "this value was missing" column for each
    feature that had missing values in training. That matters here because
    missingness is not random: the rows whose delinquency counters were coded
    96/98 are 54.65% positive against a 6.68% base rate, and income is absent
    for a fifth of borrowers. Without an indicator, imputing those to the
    median tells the model they were unremarkable, discarding the signal.
    """
    return Pipeline(
        steps=[
            ("clean", ImplausibleValueCleaner()),
            (
                "impute",
                SimpleImputer(
                    strategy="median", add_indicator=add_indicator
                ).set_output(transform="pandas"),
            ),
            ("model", model),
        ]
    )


def model_input_names(pipeline: Pipeline) -> list[str]:
    """Column names the classifier actually sees, after preprocessing.

    With missing-indicator columns this is longer than ``FEATURES``, so
    anything that pairs names with per-feature model output (importances, for
    example) must use this rather than the raw input list.
    """
    return [str(name) for name in pipeline.named_steps["impute"].get_feature_names_out()]


def as_feature_frame(records: list[dict[str, float | None]]) -> pd.DataFrame:
    """Build a model-ready frame from records keyed by dataset column name.

    Columns come out in training order. Unknown keys raise rather than being
    dropped, so a renamed or invented field can never reach the model as a
    silently missing value.
    """
    for record in records:
        unknown = set(record) - set(FEATURES)
        if unknown:
            raise KeyError(f"Unknown feature(s): {sorted(unknown)}")
    return pd.DataFrame(records, columns=list(FEATURES)).astype(float)
