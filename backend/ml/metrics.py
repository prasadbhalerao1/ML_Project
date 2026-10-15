"""Evaluation metrics for the binary delinquency classifier."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

#: Probability above which a borrower is labelled positive. 0.5 is the usual
#: default; ``docs/model-evaluation.md`` shows what other thresholds would do.
DEFAULT_THRESHOLD = 0.5


def positive_class_proba(pipeline, X) -> np.ndarray:  # noqa: N803
    """Return P(serious delinquency) for each row.

    The column is located by looking up label ``1`` in ``classes_`` rather than
    assuming ``predict_proba`` column 1, so a different class ordering cannot
    silently invert the probabilities.
    """
    classes = list(pipeline.classes_)
    if 1 not in classes:
        raise ValueError(f"Model has no positive class 1; classes are {classes}.")
    return pipeline.predict_proba(X)[:, classes.index(1)]


def evaluate_split(
    pipeline, X, y, threshold: float = DEFAULT_THRESHOLD  # noqa: N803
) -> dict[str, Any]:
    """Score a fitted pipeline on one split, in one call."""
    return classification_metrics(y, positive_class_proba(pipeline, X), threshold)


def best_f1_threshold(y_true, proba: np.ndarray) -> float:
    """Find the probability cut-off that maximises F1.

    Sweeps every cut-off the data actually distinguishes, via the
    precision-recall curve, rather than a hand-picked grid.

    **This must be called on data the model was not fitted on.** Choosing a
    threshold is a modelling decision: picking it on the test split and then
    reporting that split's metrics would be tuning on the exam paper.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision/recall have one more entry than thresholds; the last point is
    # the degenerate recall=0 end of the curve, which has no threshold.
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = int(np.nanargmax(f1[:-1]))
    return float(thresholds[best])


def classification_metrics(
    y_true, proba: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> dict[str, Any]:
    """Compute threshold-dependent and threshold-free metrics in one dict.

    ``roc_auc`` and ``pr_auc`` rank probabilities and so are unaffected by
    ``threshold``; the rest depend on it.
    """
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }


#: How the two labels read in the dataset's terms.
CLASS_LABELS = {0: "No serious delinquency", 1: "Serious delinquency"}


def per_class_report(
    y_true, proba: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> dict[str, Any]:
    """Precision, recall, F1 and support for *each* class, plus the averages.

    The headline metrics in :func:`classification_metrics` describe the
    positive class only. This is the full picture, equivalent to scikit-learn's
    ``classification_report``: it shows that the model is excellent at the
    majority class and much weaker at the minority one, which a single
    precision number hides.
    """
    y_pred = (proba >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    classes = {
        str(label): {
            "label": CLASS_LABELS[label],
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate((0, 1))
    }
    return {
        "threshold": threshold,
        "classes": classes,
        "macro_avg": _average(classes, weighted=False),
        "weighted_avg": _average(classes, weighted=True),
    }


def _average(classes: dict[str, Any], weighted: bool) -> dict[str, float]:
    """Macro average treats both classes equally; weighted scales by support."""
    rows = list(classes.values())
    weights = [row["support"] for row in rows] if weighted else [1] * len(rows)
    total = sum(weights) or 1
    return {
        metric: float(
            sum(row[metric] * weight for row, weight in zip(rows, weights)) / total
        )
        for metric in ("precision", "recall", "f1")
    }


def class_distribution(y) -> dict[str, Any]:
    """Summarise how many rows carry each label."""
    values = np.asarray(y)
    positives = int((values == 1).sum())
    return {
        "total": int(values.size),
        "negative": int((values == 0).sum()),
        "positive": positives,
        "positive_rate": float(positives / values.size) if values.size else 0.0,
    }


#: Bin count for the predicted-score histogram. 40 bins over [0, 1] is fine
#: enough to show the shape of the overlap without being noisy to draw.
SCORE_BINS = 40


def score_distribution(y_true, proba: np.ndarray, bins: int = SCORE_BINS) -> dict[str, Any]:
    """Histogram of predicted probabilities, split by the actual class.

    This is what shows *why* recall is capped: the two classes' score
    distributions overlap heavily, so no threshold separates them cleanly. A
    confusion matrix at one cut-off cannot show that.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    labels = np.asarray(y_true)
    negative, _ = np.histogram(proba[labels == 0], bins=edges)
    positive, _ = np.histogram(proba[labels == 1], bins=edges)
    return {
        "bin_edges": [round(float(edge), 6) for edge in edges],
        "negative_counts": [int(count) for count in negative],
        "positive_counts": [int(count) for count in positive],
    }


def threshold_curve(
    y_true,
    proba: np.ndarray,
    start: float = 0.02,
    stop: float = 0.98,
    points: int = 49,
    min_flagged: int = 20,
) -> list[dict[str, float]]:
    """Precision, recall and F1 across a dense grid of decision thresholds.

    The four-row summary table hides the *shape* of the trade-off — where F1
    peaks, and where precision and recall cross. This returns enough points to
    draw it.

    Each point also carries the raw confusion counts, so a reader can see what
    a threshold costs in people rather than only in rates.

    The curve stops once fewer than ``min_flagged`` records are predicted
    positive. Past that point precision is computed from a handful of samples
    and swings wildly between 0 and 1 — noise that would read as signal on a
    chart.
    """
    curve: list[dict[str, float]] = []
    for threshold in np.linspace(start, stop, points):
        metrics = classification_metrics(y_true, proba, threshold=float(threshold))
        confusion = metrics["confusion_matrix"]
        flagged = confusion["true_positives"] + confusion["false_positives"]
        if flagged < min_flagged:
            break
        curve.append(
            {
                "threshold": round(float(threshold), 4),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "flagged": int(flagged),
                "true_positives": confusion["true_positives"],
                "false_positives": confusion["false_positives"],
                "false_negatives": confusion["false_negatives"],
            }
        )
    return curve
