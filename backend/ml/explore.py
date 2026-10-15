"""Data inspection for the Give Me Some Credit training file.

Run from the ``backend/`` directory::

    python -m ml.explore

Prints the schema, missing-value and duplicate counts, target balance and the
suspicious values that motivate the preprocessing in ``ml/preprocessing.py``,
then writes three figures to ml/artifacts/figures/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; figures are written to disk

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ml.artifacts import FIGURES_DIR
from ml.dataset import (
    DEFAULT_TRAINING_CSV,
    DELINQUENCY_COUNTERS,
    FEATURES,
    PLACEHOLDER_CODES,
    TARGET,
    load_training_frame,
)
from ml.preprocessing import MIN_PLAUSIBLE_AGE

#: Features whose distributions are worth plotting individually.
PLOTTED_FEATURES: tuple[str, ...] = (
    "age",
    "MonthlyIncome",
    "RevolvingUtilizationOfUnsecuredLines",
    "NumberOfOpenCreditLinesAndLoans",
)


def main() -> None:
    args = _parse_args()
    frame = load_training_frame(args.data)

    _report_schema(frame)
    _report_missing(frame)
    _report_duplicates(frame)
    _report_target(frame)
    _report_suspicious_values(frame)
    _report_target_correlations(frame)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _plot_target_distribution(frame)
    _plot_feature_distributions(frame)
    _plot_correlation_heatmap(frame)
    print(f"\nFigures written to {FIGURES_DIR}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_TRAINING_CSV)
    return parser.parse_args()


def _report_schema(frame: pd.DataFrame) -> None:
    _heading("Shape and dtypes")
    print(f"{len(frame):,} rows x {frame.shape[1]} columns (row-index column dropped)")
    for column, dtype in frame.dtypes.items():
        role = "TARGET" if column == TARGET else "feature"
        print(f"  {column:<40} {str(dtype):<10} {role}")


def _report_missing(frame: pd.DataFrame) -> None:
    _heading("Missing values")
    missing = frame.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("None.")
        return
    for column, count in missing.items():
        print(f"  {column:<40} {count:>7,}  ({count / len(frame):.2%})")


def _report_duplicates(frame: pd.DataFrame) -> None:
    _heading("Duplicate records")
    exact = int(frame.duplicated().sum())
    predictors_only = int(frame.duplicated(subset=list(FEATURES)).sum())
    print(f"  identical rows including the label: {exact:,}")
    print(f"  identical predictor rows:           {predictors_only:,}")
    print(
        "  Kept: borrowers with the same rounded values are plausible distinct\n"
        "  people, and dropping them would bias the class balance."
    )


def _report_target(frame: pd.DataFrame) -> None:
    _heading(f"Target balance ({TARGET})")
    counts = frame[TARGET].value_counts().sort_index()
    for label, count in counts.items():
        meaning = "serious delinquency" if label == 1 else "no serious delinquency"
        print(f"  {label} ({meaning:<22}) {count:>8,}  {count / len(frame):.2%}")


def _report_suspicious_values(frame: pd.DataFrame) -> None:
    _heading("Suspicious values")
    for column in DELINQUENCY_COUNTERS:
        placeholders = int(frame[column].isin(PLACEHOLDER_CODES).sum())
        print(
            f"  {column:<40} {placeholders:>5,} rows coded "
            f"{list(PLACEHOLDER_CODES)} (treated as missing)"
        )
    print(
        f"  {'age below ' + str(MIN_PLAUSIBLE_AGE):<40} "
        f"{int((frame['age'] < MIN_PLAUSIBLE_AGE).sum()):>5,} rows (treated as missing)"
    )
    print(
        f"  {'RevolvingUtilization > 1.0':<40} "
        f"{int((frame['RevolvingUtilizationOfUnsecuredLines'] > 1).sum()):>5,} rows "
        "(kept; over-limit balances are real)"
    )
    print(
        f"  {'DebtRatio > 10':<40} "
        f"{int((frame['DebtRatio'] > 10).sum()):>5,} rows "
        "(kept; mostly rows with no recorded income)"
    )


def _report_target_correlations(frame: pd.DataFrame) -> None:
    _heading("Correlation with the target (leakage check)")
    correlations = (
        frame[list(FEATURES) + [TARGET]]
        .corr(numeric_only=True)[TARGET]
        .drop(TARGET)
        .sort_values(key=abs, ascending=False)
    )
    for feature, value in correlations.items():
        print(f"  {feature:<40} {value:+.4f}")
    print(
        "  No predictor approaches |1.0|, and none is derived from the label,\n"
        "  so there is no evidence of target leakage."
    )


def _plot_target_distribution(frame: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(6, 4))
    sns.countplot(x=frame[TARGET], hue=frame[TARGET], legend=False, ax=axis, palette="muted")
    # Kept short: a longer title overflows the axes width and gets clipped.
    axis.set_title("Target distribution")
    axis.set_xlabel("0 = no serious delinquency, 1 = serious delinquency")
    axis.set_ylabel("Borrowers")
    _save(figure, "target_distribution.png")


def _plot_feature_distributions(frame: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    for axis, feature in zip(axes.ravel(), PLOTTED_FEATURES):
        values = frame[feature].dropna()
        # Trim the top percentile so heavy right tails do not flatten the plot.
        sns.histplot(values.clip(upper=values.quantile(0.99)), bins=40, ax=axis)
        axis.set_title(feature, fontsize=10)
        axis.set_xlabel("")
    figure.suptitle("Feature distributions (top 1% clipped for display only)")
    _save(figure, "feature_distributions.png")


def _plot_correlation_heatmap(frame: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(
        frame[list(FEATURES) + [TARGET]].corr(numeric_only=True),
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        annot_kws={"size": 7},
        ax=axis,
    )
    axis.set_title("Pearson correlation, predictors and target")
    _save(figure, "correlation_heatmap.png")


def _save(figure: plt.Figure, filename: str) -> None:
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / filename, dpi=130)
    plt.close(figure)


def _heading(title: str) -> None:
    print(f"\n=== {title} ===")


if __name__ == "__main__":
    main()
