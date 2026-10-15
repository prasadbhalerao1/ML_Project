"""Dataset contract and loading for the Give Me Some Credit training file.

This module is the single source of truth for the dataset schema. Training
imports it to build the model; the API reads the same names back out of the
saved artifact metadata, so the two can never drift.

Dataset: Give Me Some Credit (Kaggle competition)
https://www.kaggle.com/competitions/GiveMeSomeCredit/data
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

#: Name of the label column in the raw CSV.
TARGET = "SeriousDlqin2yrs"

#: Human-readable meaning of a positive label, surfaced in the API and the UI.
TARGET_DEFINITION = (
    "Borrower experienced serious delinquency (90 days past due or worse) "
    "within two years of the observation date"
)

#: Model input columns, in the exact order the pipeline expects them.
FEATURES: tuple[str, ...] = (
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
)

#: Unnamed row index carried by the raw CSV. Dropped: it is a file row number,
#: not borrower information.
INDEX_COLUMN = "Unnamed: 0"

#: Late-payment counters that use 96 and 98 as placeholder codes rather than
#: real counts (see docs/methodology.md).
DELINQUENCY_COUNTERS: tuple[str, ...] = (
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
)

#: Values used as placeholders in the counters above.
PLACEHOLDER_CODES: tuple[int, ...] = (96, 98)

#: Project root, i.e. the directory holding backend/, frontend/ and data/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Default location of the labeled competition training file.
DEFAULT_TRAINING_CSV = PROJECT_ROOT / "data" / "raw" / "GiveMeSomeCredit" / "cs-training.csv"


class DatasetError(RuntimeError):
    """Raised when the dataset is missing or does not match the expected schema."""


def load_training_frame(csv_path: Path | None = None) -> pd.DataFrame:
    """Load the labeled training CSV and validate its schema.

    Returns the frame with the row-index column dropped and every label present.
    The raw file on disk is never modified.
    """
    path = Path(csv_path) if csv_path else DEFAULT_TRAINING_CSV
    if not path.exists():
        raise DatasetError(
            f"Training data not found at {path}. "
            "See README.md for Kaggle download instructions."
        )

    frame = pd.read_csv(path)
    _validate_columns(frame, path)

    if frame[TARGET].isna().any():
        raise DatasetError(
            f"{path} contains unlabeled rows. Train only on the labeled "
            "cs-training.csv, never on the competition's cs-test.csv."
        )

    return frame.drop(columns=[INDEX_COLUMN], errors="ignore")


def split_features_and_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the predictor matrix from the label vector.

    Selecting ``FEATURES`` explicitly guarantees the target cannot leak into the
    inputs, and fixes column order for the saved pipeline.
    """
    return frame.loc[:, list(FEATURES)], frame[TARGET].astype(int)


def _validate_columns(frame: pd.DataFrame, path: Path) -> None:
    missing = [column for column in (TARGET, *FEATURES) if column not in frame.columns]
    if missing:
        raise DatasetError(f"{path} is missing expected columns: {missing}")
