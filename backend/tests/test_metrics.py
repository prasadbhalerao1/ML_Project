"""Tests for metric computation and positive-class probability lookup."""

from __future__ import annotations

import numpy as np
import pytest

from ml.metrics import (
    best_f1_threshold,
    score_distribution,
    threshold_curve,
    class_distribution,
    classification_metrics,
    per_class_report,
    positive_class_proba,
)


class _StubPipeline:
    """Minimal stand-in exposing the attributes ``positive_class_proba`` uses."""

    def __init__(self, classes: list[int], probabilities: np.ndarray) -> None:
        self.classes_ = np.array(classes)
        self._probabilities = probabilities

    def predict_proba(self, X):  # noqa: N803
        return self._probabilities


def test_positive_class_proba_follows_class_order_not_column_order() -> None:
    probabilities = np.array([[0.2, 0.8], [0.9, 0.1]])

    usual = positive_class_proba(_StubPipeline([0, 1], probabilities), None)
    reversed_order = positive_class_proba(_StubPipeline([1, 0], probabilities), None)

    assert usual.tolist() == [0.8, 0.1]
    assert reversed_order.tolist() == [0.2, 0.9]


def test_positive_class_proba_rejects_a_model_without_the_positive_class() -> None:
    with pytest.raises(ValueError, match="no positive class"):
        positive_class_proba(_StubPipeline([0, 2], np.array([[1.0, 0.0]])), None)


def test_classification_metrics_on_a_hand_checked_example() -> None:
    y_true = [0, 0, 1, 1]
    proba = np.array([0.1, 0.6, 0.4, 0.9])

    metrics = classification_metrics(y_true, proba, threshold=0.5)

    # Predictions are [0, 1, 0, 1]: one TP, one FP, one FN, one TN.
    assert metrics["confusion_matrix"] == {
        "true_negatives": 1,
        "false_positives": 1,
        "false_negatives": 1,
        "true_positives": 1,
    }
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)


def test_lowering_the_threshold_raises_recall_and_never_lowers_it() -> None:
    y_true = [0, 0, 1, 1]
    proba = np.array([0.1, 0.6, 0.4, 0.9])

    strict = classification_metrics(y_true, proba, threshold=0.5)
    lenient = classification_metrics(y_true, proba, threshold=0.3)

    assert lenient["recall"] > strict["recall"]
    assert lenient["roc_auc"] == strict["roc_auc"]  # ranking metrics ignore threshold


def test_best_f1_threshold_beats_the_conventional_default() -> None:
    """On a skewed score distribution, 0.5 is rarely the F1 optimum."""
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 900 + [1] * 100)
    # Positives score higher on average, but most sit below 0.5.
    proba = np.concatenate(
        [rng.beta(2, 20, size=900), rng.beta(5, 10, size=100)]
    )

    threshold = best_f1_threshold(y_true, proba)

    tuned = classification_metrics(y_true, proba, threshold)["f1"]
    default = classification_metrics(y_true, proba, 0.5)["f1"]
    assert 0.0 < threshold < 1.0
    assert tuned >= default


def test_best_f1_threshold_is_the_maximum_over_a_dense_sweep() -> None:
    rng = np.random.default_rng(1)
    y_true = np.array([0] * 400 + [1] * 100)
    proba = np.concatenate([rng.beta(2, 8, size=400), rng.beta(6, 4, size=100)])

    threshold = best_f1_threshold(y_true, proba)
    best = classification_metrics(y_true, proba, threshold)["f1"]

    swept = max(
        classification_metrics(y_true, proba, candidate)["f1"]
        for candidate in np.linspace(0.01, 0.99, 99)
    )
    assert best >= swept - 1e-9


def test_per_class_report_covers_both_classes_and_averages() -> None:
    y_true = [0, 0, 0, 1]
    proba = np.array([0.1, 0.2, 0.9, 0.8])

    report = per_class_report(y_true, proba, threshold=0.5)

    assert set(report["classes"]) == {"0", "1"}
    assert report["classes"]["0"]["support"] == 3
    assert report["classes"]["1"]["support"] == 1
    # Predictions are [0, 0, 1, 1]: class 1 has one TP and one FP.
    assert report["classes"]["1"]["precision"] == pytest.approx(0.5)
    assert report["classes"]["1"]["recall"] == pytest.approx(1.0)
    # Macro weights both classes equally; weighted follows support.
    assert report["macro_avg"]["f1"] != pytest.approx(report["weighted_avg"]["f1"])


def test_per_class_report_macro_average_is_the_plain_mean() -> None:
    y_true = [0, 0, 1, 1]
    proba = np.array([0.1, 0.6, 0.4, 0.9])

    report = per_class_report(y_true, proba, threshold=0.5)

    expected = (report["classes"]["0"]["f1"] + report["classes"]["1"]["f1"]) / 2
    assert report["macro_avg"]["f1"] == pytest.approx(expected)


def test_class_distribution_counts_each_label() -> None:
    assert class_distribution([0, 0, 0, 1]) == {
        "total": 4,
        "negative": 3,
        "positive": 1,
        "positive_rate": pytest.approx(0.25),
    }


def test_score_distribution_bins_every_record_once() -> None:
    y_true = [0] * 6 + [1] * 4
    proba = np.array([0.01, 0.05, 0.2, 0.4, 0.6, 0.99, 0.3, 0.5, 0.7, 0.95])

    dist = score_distribution(y_true, proba, bins=10)

    assert len(dist["bin_edges"]) == len(dist["negative_counts"]) + 1
    assert sum(dist["negative_counts"]) == 6
    assert sum(dist["positive_counts"]) == 4
    assert dist["bin_edges"][0] == 0.0 and dist["bin_edges"][-1] == 1.0


def test_score_distribution_places_scores_in_the_right_bins() -> None:
    dist = score_distribution([0, 1], np.array([0.05, 0.95]), bins=10)

    assert dist["negative_counts"][0] == 1   # 0.05 -> first bin
    assert dist["positive_counts"][-1] == 1  # 0.95 -> last bin


def test_threshold_curve_is_ordered_and_recall_decreases() -> None:
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 80 + [1] * 20)
    proba = np.concatenate([rng.beta(2, 8, size=80), rng.beta(6, 4, size=20)])

    # min_flagged=0 keeps every point, so the ordering check sees the full grid.
    curve = threshold_curve(y_true, proba, points=20, min_flagged=0)

    thresholds = [p["threshold"] for p in curve]
    recalls = [p["recall"] for p in curve]
    assert len(curve) == 20
    assert thresholds == sorted(thresholds)
    # Raising the cut-off can never flag more positives, so recall is monotone.
    assert all(a >= b for a, b in zip(recalls, recalls[1:]))


def test_threshold_curve_agrees_with_classification_metrics() -> None:
    y_true = [0, 0, 1, 1]
    proba = np.array([0.1, 0.6, 0.4, 0.9])

    point = threshold_curve(y_true, proba, start=0.5, stop=0.5, points=1, min_flagged=0)[0]
    direct = classification_metrics(y_true, proba, threshold=0.5)

    assert point["precision"] == pytest.approx(direct["precision"])
    assert point["recall"] == pytest.approx(direct["recall"])
    assert point["f1"] == pytest.approx(direct["f1"])


def test_threshold_curve_stops_before_precision_degenerates() -> None:
    """Past the point where almost nothing is flagged, precision is noise."""
    rng = np.random.default_rng(3)
    y_true = np.array([0] * 900 + [1] * 100)
    proba = np.concatenate([rng.beta(2, 10, size=900), rng.beta(5, 5, size=100)])

    curve = threshold_curve(y_true, proba, min_flagged=20)

    assert all(point["flagged"] >= 20 for point in curve)
    # Nothing in the retained range collapses to the zero-division fallback.
    assert all(point["precision"] > 0 for point in curve)


def test_threshold_curve_counts_agree_with_the_rates() -> None:
    rng = np.random.default_rng(4)
    y_true = np.array([0] * 400 + [1] * 100)
    proba = np.concatenate([rng.beta(2, 8, size=400), rng.beta(6, 4, size=100)])

    for point in threshold_curve(y_true, proba, points=12):
        assert point["flagged"] == point["true_positives"] + point["false_positives"]
        assert point["precision"] == pytest.approx(
            point["true_positives"] / point["flagged"]
        )
        caught, missed = point["true_positives"], point["false_negatives"]
        assert point["recall"] == pytest.approx(caught / (caught + missed))
