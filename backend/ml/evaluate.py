"""Full evaluation report for the saved model artifact.

Run from the ``backend/`` directory::

    python -m ml.evaluate            # per-split metrics, gap, thresholds, figures
    python -m ml.evaluate --cross-validate   # also run 5-fold CV (refits, slower)

This loads the artifact from disk rather than training anything, which also
verifies that the saved pipeline works independently of ``ml/train.py``. It
re-creates the same splits from the seed recorded in ``metadata.json``, so the
test rows here are exactly the ones the model never saw.

Reported:

1. Metrics on train, validation and test, with each split marked in-sample or
   held out, so the generalisation gap is visible rather than implied.
2. The train-vs-test gap on every metric.
3. Stratified 5-fold cross-validation on train+validation (optional), which
   says how much of the single test score is luck of one split.
4. The precision/recall trade-off across thresholds.
5. Feature importances, and the evaluation figures.

Results are written to ``ml/artifacts/evaluation_report.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold

from ml import artifacts
from ml.dataset import DEFAULT_TRAINING_CSV, load_training_frame, split_features_and_target
from ml.metrics import (
    DEFAULT_THRESHOLD,
    class_distribution,
    classification_metrics,
    evaluate_split,
    per_class_report,
    positive_class_proba,
)
from ml.splits import RANDOM_SEED, Splits, make_splits
from ml.train import REPORTED_THRESHOLDS

#: Metrics shown in the per-split table, in display order.
HEADLINE_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")

CV_FOLDS = 5


def main() -> None:
    args = _parse_args()

    loaded = artifacts.load()
    metadata = loaded.metadata
    pipeline = loaded.pipeline
    _print_header(metadata)

    X, y = split_features_and_target(load_training_frame(args.data))
    seed = metadata.get("random_seed", RANDOM_SEED)
    threshold = _decision_threshold(metadata)
    splits = make_splits(X, y, seed=seed)

    per_split = _evaluate_every_split(pipeline, splits, threshold)
    _print_split_distribution(splits)
    _print_per_split_metrics(per_split, threshold)
    gap = _print_generalisation_gap(per_split)

    test_proba = positive_class_proba(pipeline, splits.X_test)
    report = per_class_report(splits.y_test, test_proba, threshold)
    _print_classification_report(report, splits.y_test, test_proba)
    sweep = _print_threshold_table(splits.y_test, test_proba, threshold)
    _print_feature_importances(metadata)

    cross_validation = None
    if args.cross_validate:
        cross_validation = _run_cross_validation(metadata, splits, seed, threshold)

    artifacts.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _plot_curves(splits.y_test, test_proba, per_split["test"])
    _plot_split_comparison(pipeline, splits)
    _plot_confusion_matrix(per_split["test"])
    _plot_feature_importances(metadata)

    _write_report(
        {
            "model_version": metadata["model_version"],
            "evaluated_against": Path(args.data).name,
            "random_seed": seed,
            "class_distribution": {
                name: class_distribution(labels)
                for name, labels in _labelled_splits(splits)
            },
            "decision_threshold": threshold,
            "per_split_metrics": per_split,
            "train_test_gap": gap,
            "test_classification_report": report,
            "test_threshold_sweep": sweep,
            "cross_validation": cross_validation,
        }
    )
    print(f"\nFigures written to {artifacts.FIGURES_DIR}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_TRAINING_CSV)
    parser.add_argument(
        "--cross-validate",
        action="store_true",
        help=f"Additionally run stratified {CV_FOLDS}-fold CV (refits the model {CV_FOLDS} times).",
    )
    return parser.parse_args()


def _labelled_splits(splits: Splits):
    return (
        ("train", splits.y_train),
        ("validation", splits.y_validation),
        ("test", splits.y_test),
    )


def _decision_threshold(metadata: dict[str, Any]) -> float:
    """The threshold chosen at training time, or the conventional default."""
    return float(metadata.get("decision_threshold", {}).get("value", DEFAULT_THRESHOLD))


def _evaluate_every_split(
    pipeline, splits: Splits, threshold: float
) -> dict[str, dict[str, Any]]:
    """Score the saved model on all three splits.

    The saved model was refitted on train + validation, so those two are
    in-sample here; only test is a genuine held-out estimate. Reporting all
    three is what makes the overfitting gap visible.
    """
    return {
        "train": evaluate_split(pipeline, splits.X_train, splits.y_train, threshold),
        "validation": evaluate_split(
            pipeline, splits.X_validation, splits.y_validation, threshold
        ),
        "test": evaluate_split(pipeline, splits.X_test, splits.y_test, threshold),
    }


def _print_classification_report(
    report: dict[str, Any], y_true, proba: np.ndarray
) -> None:
    """Per-class precision/recall/F1 on the test split, plus the averages."""
    _heading(f"Classification report, test split (threshold {report['threshold']:.4f})")
    print(
        f"  {'class':<26} {'precision':>10} {'recall':>8} {'f1':>8} {'support':>9}"
    )
    print("  " + "-" * 63)
    for label, row in report["classes"].items():
        print(
            f"  {label} {row['label']:<24} {row['precision']:>10.4f} "
            f"{row['recall']:>8.4f} {row['f1']:>8.4f} {row['support']:>9,}"
        )
    print()
    total = sum(row["support"] for row in report["classes"].values())
    accuracy = classification_metrics(y_true, proba, report["threshold"])["accuracy"]
    print(f"  {'accuracy':<26} {'':>10} {'':>8} {accuracy:>8.4f} {total:>9,}")
    for name in ("macro_avg", "weighted_avg"):
        average = report[name]
        print(
            f"  {name.replace('_', ' '):<26} {average['precision']:>10.4f} "
            f"{average['recall']:>8.4f} {average['f1']:>8.4f} {total:>9,}"
        )


def _print_header(metadata: dict[str, Any]) -> None:
    configuration = metadata["selected_configuration"]
    print(f"Loaded {metadata['model_name']} v{metadata['model_version']}")
    print(f"  algorithm     {metadata['algorithm']}")
    print(f"  configuration {configuration['name']} {configuration['params']}")
    print(f"  fitted on     {configuration['fitted_on']}")
    print(f"  trained at    {metadata['trained_at']}")
    print(f"  class order   {metadata['class_order']}")


def _print_split_distribution(splits: Splits) -> None:
    _heading("Split sizes and class balance")
    print(f"  {'split':<12} {'rows':>8} {'negative':>10} {'positive':>9} {'positive rate':>14}")
    for name, labels in _labelled_splits(splits):
        stats = class_distribution(labels)
        print(
            f"  {name:<12} {stats['total']:>8,} {stats['negative']:>10,} "
            f"{stats['positive']:>9,} {stats['positive_rate']:>13.2%}"
        )


def _print_per_split_metrics(
    per_split: dict[str, dict[str, Any]], threshold: float
) -> None:
    _heading(f"Metrics by split (threshold {threshold:.4f})")
    print("  train and validation are IN-SAMPLE: the final model was refitted on both.")
    print("  Only 'test' is a held-out estimate.\n")

    header = f"  {'metric':<12}" + "".join(f"{name:>14}" for name in per_split)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for metric in HEADLINE_METRICS:
        row = f"  {metric:<12}"
        for name in per_split:
            row += f"{per_split[name][metric]:>14.4f}"
        print(row)

    print()
    for name, metrics in per_split.items():
        confusion = metrics["confusion_matrix"]
        print(
            f"  {name:<12} TN={confusion['true_negatives']:>6,} "
            f"FP={confusion['false_positives']:>5,} "
            f"FN={confusion['false_negatives']:>5,} "
            f"TP={confusion['true_positives']:>5,}"
        )


def _print_generalisation_gap(per_split: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Train minus test on every headline metric.

    A large positive gap means the forest memorised the training rows. This is
    the number that says whether `min_samples_leaf=20` did its job.
    """
    _heading("Generalisation gap (train - test)")
    gap = {
        metric: float(per_split["train"][metric] - per_split["test"][metric])
        for metric in HEADLINE_METRICS
    }
    for metric, value in gap.items():
        print(f"  {metric:<12} {value:+.4f}")
    return gap


def _print_threshold_table(
    y_true, proba: np.ndarray, selected: float
) -> list[dict[str, Any]]:
    _heading("Threshold trade-off on the test split")
    print(
        f"  {'thr':>7} {'accuracy':>9} {'precision':>10} {'recall':>8} "
        f"{'f1':>8} {'flagged':>9}"
    )
    rows = []
    for threshold in sorted({*REPORTED_THRESHOLDS, round(selected, 4)}):
        row = classification_metrics(y_true, proba, threshold=threshold)
        confusion = row["confusion_matrix"]
        flagged = confusion["true_positives"] + confusion["false_positives"]
        marker = "  <- in use" if abs(threshold - selected) < 1e-9 else ""
        print(
            f"  {threshold:>7.4f} {row['accuracy']:>9.4f} {row['precision']:>10.4f} "
            f"{row['recall']:>8.4f} {row['f1']:>8.4f} {flagged:>9,}{marker}"
        )
        rows.append(row)
    return rows


def _run_cross_validation(
    metadata: dict[str, Any], splits: Splits, seed: int, threshold: float
) -> dict[str, Any]:
    """Stratified K-fold CV of the selected configuration on train + validation.

    The single test score comes from one particular 30,000-row split. CV refits
    the same configuration on K different partitions and reports the spread, so
    we can say whether that score is stable or an artifact of one lucky split.
    The test rows are excluded throughout.
    """
    from ml.train import CANDIDATES  # local import: only needed for this path

    name = metadata["selected_configuration"]["name"]
    candidate = next(c for c in CANDIDATES if c.name == name)

    X = splits.X_train_and_validation
    y = splits.y_train_and_validation

    _heading(f"Stratified {CV_FOLDS}-fold cross-validation on train + validation")
    print(
        f"  configuration '{name}', {len(X):,} rows, test split excluded, "
        f"threshold {threshold:.4f}"
    )

    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    scores: list[dict[str, Any]] = []

    for index, (train_index, validate_index) in enumerate(folds.split(X, y), start=1):
        pipeline = clone(candidate.build(seed))
        pipeline.fit(X.iloc[train_index], y.iloc[train_index])
        metrics = evaluate_split(
            pipeline, X.iloc[validate_index], y.iloc[validate_index], threshold
        )
        scores.append(metrics)
        print(
            f"  fold {index}/{CV_FOLDS}  roc_auc={metrics['roc_auc']:.4f} "
            f"pr_auc={metrics['pr_auc']:.4f} precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f}"
        )

    summary = {
        metric: {
            "mean": float(np.mean([score[metric] for score in scores])),
            "std": float(np.std([score[metric] for score in scores])),
            "min": float(np.min([score[metric] for score in scores])),
            "max": float(np.max([score[metric] for score in scores])),
        }
        for metric in HEADLINE_METRICS
    }

    print(f"\n  {'metric':<12} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
    for metric, stats in summary.items():
        print(
            f"  {metric:<12} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
            f"{stats['min']:>10.4f} {stats['max']:>10.4f}"
        )

    return {"folds": CV_FOLDS, "per_fold": scores, "summary": summary}


def _print_feature_importances(metadata: dict[str, Any]) -> None:
    _heading("Feature importance (mean impurity decrease -- not a causal effect)")
    for entry in metadata["feature_importances"]:
        print(f"  {entry['feature']:<40} {entry['importance']:.4f}")


def _write_report(report: dict[str, Any]) -> None:
    path = artifacts.ARTIFACTS_DIR / "evaluation_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport written to {path}")


def _plot_curves(y_true, proba: np.ndarray, metrics: dict[str, Any]) -> None:
    figure, (roc_axis, pr_axis) = plt.subplots(1, 2, figsize=(11, 4.5))

    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, proba)
    roc_axis.plot(false_positive_rate, true_positive_rate, label=f"AUC = {metrics['roc_auc']:.3f}")
    roc_axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random")
    roc_axis.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test)")
    roc_axis.legend(loc="lower right")

    precision, recall, _ = precision_recall_curve(y_true, proba)
    pr_axis.plot(recall, precision, label=f"PR-AUC = {metrics['pr_auc']:.3f}")
    base_rate = float(np.mean(y_true))
    pr_axis.axhline(base_rate, linestyle="--", color="grey", label=f"Base rate = {base_rate:.3f}")
    pr_axis.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve (test)")
    pr_axis.legend(loc="upper right")

    _save(figure, "test_curves.png")


def _plot_split_comparison(pipeline, splits: Splits) -> None:
    """ROC curves for all three splits on one axis: the gap, made visible."""
    figure, axis = plt.subplots(figsize=(6, 5))
    parts = (
        ("train (in-sample)", splits.X_train, splits.y_train),
        ("validation (in-sample)", splits.X_validation, splits.y_validation),
        ("test (held out)", splits.X_test, splits.y_test),
    )
    for label, X, y in parts:
        proba = positive_class_proba(pipeline, X)
        false_positive_rate, true_positive_rate, _ = roc_curve(y, proba)
        auc = classification_metrics(y, proba)["roc_auc"]
        axis.plot(false_positive_rate, true_positive_rate, label=f"{label} — AUC {auc:.3f}")

    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random")
    axis.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="ROC by split: in-sample vs held out",
    )
    axis.legend(loc="lower right", fontsize=9)
    _save(figure, "roc_by_split.png")


def _plot_confusion_matrix(metrics: dict[str, Any]) -> None:
    confusion = metrics["confusion_matrix"]
    counts = np.array(
        [
            [confusion["true_negatives"], confusion["false_positives"]],
            [confusion["false_negatives"], confusion["true_positives"]],
        ]
    )
    figure, axis = plt.subplots(figsize=(5, 4.2))
    sns.heatmap(
        counts,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        xticklabels=["predicted 0", "predicted 1"],
        yticklabels=["actual 0", "actual 1"],
        ax=axis,
    )
    axis.set_title(f"Test confusion matrix (threshold {metrics['threshold']:.2f})")
    _save(figure, "confusion_matrix.png")


def _plot_feature_importances(metadata: dict[str, Any]) -> None:
    entries = metadata["feature_importances"]
    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    sns.barplot(
        x=[entry["importance"] for entry in entries],
        y=[entry["feature"] for entry in entries],
        hue=[entry["feature"] for entry in entries],
        legend=False,
        palette="crest",
        ax=axis,
    )
    axis.set_title("Random Forest feature importance (impurity decrease, not causal)")
    axis.set_xlabel("Importance")
    _save(figure, "feature_importance.png")


def _save(figure: plt.Figure, filename: str) -> None:
    figure.tight_layout()
    figure.savefig(artifacts.FIGURES_DIR / filename, dpi=130)
    plt.close(figure)


def _heading(title: str) -> None:
    print(f"\n=== {title} ===")


if __name__ == "__main__":
    main()
