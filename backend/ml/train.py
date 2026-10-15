"""Train the Random Forest delinquency classifier and save the artifact.

Run from the ``backend/`` directory::

    python -m ml.train

Steps: load the labeled Kaggle training file, split it 60/20/20 with
stratification, fit a few Random Forest configurations on the training split,
pick one on the validation split, refit it on train+validation, then score it
once on the held-out test split and save pipeline + metadata to ml/artifacts/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from ml import artifacts
from ml.dataset import (
    DEFAULT_TRAINING_CSV,
    FEATURES,
    TARGET,
    TARGET_DEFINITION,
    load_training_frame,
    split_features_and_target,
)
from ml.metrics import (
    DEFAULT_THRESHOLD,
    best_f1_threshold,
    class_distribution,
    classification_metrics,
    positive_class_proba,
    score_distribution,
    threshold_curve,
)
from ml.preprocessing import build_pipeline, model_input_names
from ml.splits import RANDOM_SEED, Splits, make_splits

MODEL_VERSION = "1.0.0"
MODEL_NAME = "Random Forest delinquency classifier"
DATASET_NAME = "Give Me Some Credit (Kaggle competition)"
DATASET_URL = "https://www.kaggle.com/competitions/GiveMeSomeCredit/data"

#: Metric used to pick the final configuration. Average precision (PR-AUC)
#: focuses on the rare positive class, unlike accuracy or ROC-AUC.
SELECTION_METRIC = "pr_auc"

#: Thresholds reported alongside the final metrics to show the
#: precision/recall trade-off. They do not change the saved model.
REPORTED_THRESHOLDS: tuple[float, ...] = (0.1, 0.2, 0.3, DEFAULT_THRESHOLD)


@dataclass(frozen=True)
class Candidate:
    """One Random Forest configuration, with its preprocessing variant."""

    name: str
    rationale: str
    params: dict[str, Any]
    #: Add binary "value was missing" columns alongside the imputed values.
    add_indicator: bool = False

    def build(self, seed: int) -> Pipeline:
        return build_pipeline(
            RandomForestClassifier(random_state=seed, n_jobs=-1, **self.params),
            add_indicator=self.add_indicator,
        )


#: A deliberately short list. The goal is a correct, well-understood Random
#: Forest, not a hyperparameter search.
CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        name="baseline",
        rationale="Default-depth forest, no class weighting. Reference point.",
        params={"n_estimators": 300, "min_samples_leaf": 1},
    ),
    Candidate(
        name="regularised",
        rationale="Larger leaves to curb the overfitting a full-depth forest shows.",
        params={"n_estimators": 300, "min_samples_leaf": 20},
    ),
    Candidate(
        name="balanced",
        rationale="class_weight='balanced' to counter the 6.7% positive rate.",
        params={
            "n_estimators": 300,
            "min_samples_leaf": 20,
            "class_weight": "balanced",
        },
    ),
    Candidate(
        name="balanced_subsample",
        rationale="Per-tree class weighting, the bootstrap-aware variant.",
        params={
            "n_estimators": 300,
            "min_samples_leaf": 20,
            "class_weight": "balanced_subsample",
        },
    ),
    Candidate(
        name="regularised_with_indicators",
        rationale=(
            "As 'regularised', plus missing-value indicator columns. "
            "Missingness is informative here: the 96/98-coded rows are 54.65% "
            "positive against a 6.68% base rate."
        ),
        params={"n_estimators": 300, "min_samples_leaf": 20},
        add_indicator=True,
    ),
)


def main() -> None:
    args = _parse_args()

    print(f"Loading {args.data}")
    frame = load_training_frame(args.data)
    X, y = split_features_and_target(frame)
    print(f"  {len(frame):,} labeled rows, {len(FEATURES)} predictors")

    splits = make_splits(X, y, seed=args.seed)
    _report_splits(splits)

    validation_scores, validation_probabilities = _evaluate_candidates(
        splits, seed=args.seed
    )
    winner = max(validation_scores, key=lambda row: row[SELECTION_METRIC])
    chosen = next(c for c in CANDIDATES if c.name == winner["candidate"])
    print(
        f"\nSelected '{chosen.name}' on validation "
        f"{SELECTION_METRIC}={winner[SELECTION_METRIC]:.4f}"
    )

    threshold = _select_threshold(splits, validation_probabilities[chosen.name])

    print("Refitting the selected configuration on train + validation")
    pipeline = chosen.build(args.seed)
    pipeline.fit(splits.X_train_and_validation, splits.y_train_and_validation)

    test_proba = positive_class_proba(pipeline, splits.X_test)
    test_metrics = classification_metrics(splits.y_test, test_proba, threshold=threshold)
    threshold_sweep = [
        classification_metrics(splits.y_test, test_proba, threshold=t)
        for t in sorted({*REPORTED_THRESHOLDS, round(threshold, 4)})
    ]
    _report_test_metrics(test_metrics)

    metadata = _build_metadata(
        chosen=chosen,
        splits=splits,
        validation_scores=validation_scores,
        threshold=threshold,
        test_proba=test_proba,
        test_metrics=test_metrics,
        threshold_sweep=threshold_sweep,
        pipeline=pipeline,
        args=args,
    )
    artifacts.save(pipeline, metadata)
    _save_demo_examples(splits, test_proba)
    print(f"\nSaved model to {artifacts.MODEL_PATH}")
    print(f"Saved metadata to {artifacts.METADATA_PATH}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_TRAINING_CSV,
        help="Path to the labeled cs-training.csv (default: data/raw/GiveMeSomeCredit/).",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED, help="Random seed for splits and forests."
    )
    return parser.parse_args()


def _evaluate_candidates(
    splits: Splits, seed: int
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Fit every candidate on the training split and score it on validation.

    Returns the comparison table and each candidate's validation probabilities.
    The latter are what the decision threshold is chosen from: these models
    never saw the validation rows, so the choice is made out-of-sample.
    """
    scores: list[dict[str, Any]] = []
    probabilities: dict[str, np.ndarray] = {}
    for candidate in CANDIDATES:
        print(f"\nFitting candidate '{candidate.name}' ({candidate.rationale})")
        pipeline = candidate.build(seed)
        pipeline.fit(splits.X_train, splits.y_train)
        proba = positive_class_proba(pipeline, splits.X_validation)
        metrics = classification_metrics(splits.y_validation, proba)
        print(
            f"  validation roc_auc={metrics['roc_auc']:.4f} "
            f"pr_auc={metrics['pr_auc']:.4f} recall={metrics['recall']:.4f} "
            f"precision={metrics['precision']:.4f}"
        )
        scores.append(
            {
                "candidate": candidate.name,
                "params": candidate.params,
                "add_indicator": candidate.add_indicator,
                **metrics,
            }
        )
        probabilities[candidate.name] = proba
    return scores, probabilities


def _select_threshold(splits: Splits, validation_proba: np.ndarray) -> float:
    """Choose the decision threshold on the validation split, by maximising F1.

    The conventional 0.50 is a poor cut-off under this much class imbalance:
    it makes the model so conservative that recall collapses. Maximising F1
    balances precision against recall instead of inheriting a default.

    This uses the candidate fitted on *train only*, scored on validation, so
    the threshold is picked on data that model never saw. The test split takes
    no part in the choice.
    """
    threshold = best_f1_threshold(splits.y_validation, validation_proba)
    chosen = classification_metrics(splits.y_validation, validation_proba, threshold)
    default = classification_metrics(
        splits.y_validation, validation_proba, DEFAULT_THRESHOLD
    )
    print(f"\nDecision threshold selected on validation (max F1): {threshold:.4f}")
    print(
        f"  at {threshold:.4f}  precision={chosen['precision']:.4f} "
        f"recall={chosen['recall']:.4f} f1={chosen['f1']:.4f}"
    )
    print(
        f"  at {DEFAULT_THRESHOLD:.4f}  precision={default['precision']:.4f} "
        f"recall={default['recall']:.4f} f1={default['f1']:.4f}  (conventional default)"
    )
    return threshold


def _build_metadata(
    *,
    chosen: Candidate,
    splits: Splits,
    validation_scores: list[dict[str, Any]],
    threshold: float,
    test_proba: np.ndarray,
    test_metrics: dict[str, Any],
    threshold_sweep: list[dict[str, Any]],
    pipeline: Pipeline,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Assemble everything the API and the docs need to describe this model."""
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "algorithm": "RandomForestClassifier (scikit-learn)",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_seed": args.seed,
        "dataset": {
            "name": DATASET_NAME,
            "url": DATASET_URL,
            "file": Path(args.data).name,
            "labeled_rows": int(len(splits.y_train_and_validation) + len(splits.y_test)),
        },
        "target": {"column": TARGET, "definition": TARGET_DEFINITION},
        "features": list(FEATURES),
        "class_order": [int(label) for label in pipeline.classes_],
        "selected_configuration": {
            "name": chosen.name,
            "rationale": chosen.rationale,
            "params": chosen.params,
            "add_indicator": chosen.add_indicator,
            "selection_metric": SELECTION_METRIC,
            "fitted_on": "train + validation (60% + 20%)",
        },
        "model_input_columns": model_input_names(pipeline),
        "decision_threshold": {
            "value": float(threshold),
            "selected_by": "maximum F1",
            "selected_on": (
                "validation split, scored by the candidate fitted on train only "
                "(never on the test split)"
            ),
            "conventional_default": DEFAULT_THRESHOLD,
        },
        "validation_comparison": validation_scores,
        "test_metrics": test_metrics,
        "test_threshold_sweep": threshold_sweep,
        "test_threshold_curve": threshold_curve(splits.y_test, test_proba),
        "test_score_distribution": score_distribution(splits.y_test, test_proba),
        "class_distribution": {
            "train": class_distribution(splits.y_train),
            "validation": class_distribution(splits.y_validation),
            "test": class_distribution(splits.y_test),
        },
        "feature_importances": _feature_importances(pipeline),
    }


def _feature_importances(pipeline: Pipeline) -> list[dict[str, float]]:
    """Mean impurity-decrease importances, descending.

    Names come from the fitted imputer, not the raw ``FEATURES`` list, because
    missing-indicator columns make the model's input wider than its input
    schema. These describe what the forest splits on; they are not causal
    effects.
    """
    importances = pipeline.named_steps["model"].feature_importances_
    names = model_input_names(pipeline)
    if len(names) != len(importances):  # pragma: no cover - guards a silent mismatch
        raise RuntimeError(
            f"{len(names)} feature names for {len(importances)} importances."
        )
    ranked = sorted(zip(names, importances), key=lambda pair: pair[1], reverse=True)
    return [{"feature": name, "importance": float(value)} for name, value in ranked]


def _save_demo_examples(splits: Splits, test_proba) -> None:
    """Save two real held-out records for the frontend's demo button.

    These are genuine dataset rows, not invented inputs, and the UI labels them
    as demonstration data.
    """
    frame = splits.X_test.assign(_proba=test_proba, _label=splits.y_test)
    complete = frame.dropna()
    examples = {
        "lower_risk": complete.nsmallest(1, "_proba"),
        "higher_risk": complete.nlargest(1, "_proba"),
    }
    payload = [
        {
            "label": name,
            "actual_outcome": int(rows["_label"].iloc[0]),
            "features": {
                feature: float(rows[feature].iloc[0]) for feature in FEATURES
            },
        }
        for name, rows in examples.items()
    ]
    artifacts.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (artifacts.ARTIFACTS_DIR / "demo_examples.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def _report_splits(splits: Splits) -> None:
    print("\nStratified splits")
    for name, labels in (
        ("train", splits.y_train),
        ("validation", splits.y_validation),
        ("test", splits.y_test),
    ):
        stats = class_distribution(labels)
        print(
            f"  {name:<11} {stats['total']:>7,} rows, "
            f"{stats['positive']:>6,} positive ({stats['positive_rate']:.2%})"
        )


def _report_test_metrics(metrics: dict[str, Any]) -> None:
    confusion = metrics["confusion_matrix"]
    print(f"\nHeld-out test metrics (threshold {metrics['threshold']:.4f})")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
        print(f"  {key:<10} {metrics[key]:.4f}")
    print(
        f"  confusion  TN={confusion['true_negatives']:,} "
        f"FP={confusion['false_positives']:,} "
        f"FN={confusion['false_negatives']:,} "
        f"TP={confusion['true_positives']:,}"
    )


if __name__ == "__main__":
    main()
