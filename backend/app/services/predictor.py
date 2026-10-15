"""Inference service: the only place that touches the trained pipeline.

Route handlers call this; they never load artifacts or build frames themselves.
The pipeline is loaded once at application startup and reused for every
request.
"""

from __future__ import annotations

import json
from functools import cached_property
from typing import Any

from app.core.config import (
    FALLBACK_CLASSIFICATION_THRESHOLD,
    RISK_BAND_DISCLAIMER,
    risk_band_definitions,
    risk_band_for,
)
from app.schemas.prediction import (
    DemoExample,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from ml import artifacts
from ml.metrics import positive_class_proba
from ml.preprocessing import as_feature_frame

PREDICTED_LABELS = {
    0: "No serious delinquency",
    1: "Serious delinquency",
}

_DEMO_EXAMPLES_PATH = artifacts.ARTIFACTS_DIR / "demo_examples.json"


class Predictor:
    """Wraps a loaded pipeline and its metadata."""

    def __init__(self, loaded: artifacts.LoadedModel) -> None:
        self._pipeline = loaded.pipeline
        self._metadata = loaded.metadata
        self._validate_classes()

    @classmethod
    def from_artifacts(cls) -> "Predictor":
        """Load the saved artifact. Raises ``ArtifactError`` if it is missing."""
        return cls(artifacts.load())

    @property
    def model_version(self) -> str:
        return self._metadata["model_version"]

    @property
    def features(self) -> list[str]:
        return list(self._metadata["features"])

    @cached_property
    def classification_threshold(self) -> float:
        """The cut-off chosen on validation during training.

        Read from the artifact rather than configured here, so the threshold
        the API applies is always the one the reported metrics were measured
        at.
        """
        threshold = self._metadata.get("decision_threshold", {}).get("value")
        return float(threshold) if threshold is not None else FALLBACK_CLASSIFICATION_THRESHOLD

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Score one borrower record."""
        frame = as_feature_frame([request.to_feature_record()])
        probability = float(positive_class_proba(self._pipeline, frame)[0])
        predicted_class = int(probability >= self.classification_threshold)
        return PredictionResponse(
            default_probability=probability,
            predicted_class=predicted_class,
            predicted_label=PREDICTED_LABELS[predicted_class],
            classification_threshold=self.classification_threshold,
            risk_band=risk_band_for(probability),
            risk_band_disclaimer=RISK_BAND_DISCLAIMER,
            model_version=self.model_version,
            target_definition=self._metadata["target"]["definition"],
        )

    def model_info(self) -> ModelInfoResponse:
        """Describe the model from the metadata saved at training time."""
        metadata = self._metadata
        return ModelInfoResponse(
            model_name=metadata["model_name"],
            model_version=metadata["model_version"],
            algorithm=metadata["algorithm"],
            trained_at=metadata["trained_at"],
            dataset_name=metadata["dataset"]["name"],
            dataset_url=metadata["dataset"]["url"],
            target_column=metadata["target"]["column"],
            target_definition=metadata["target"]["definition"],
            features=self.features,
            class_order=metadata["class_order"],
            classification_threshold=self.classification_threshold,
            risk_bands=risk_band_definitions(),
            risk_band_disclaimer=RISK_BAND_DISCLAIMER,
            test_metrics=metadata["test_metrics"],
            test_threshold_sweep=metadata["test_threshold_sweep"],
            test_threshold_curve=metadata["test_threshold_curve"],
            test_score_distribution=metadata["test_score_distribution"],
            class_distribution=metadata["class_distribution"],
            feature_importances=metadata["feature_importances"],
        )

    @cached_property
    def demo_examples(self) -> list[DemoExample]:
        """Real held-out dataset rows saved by the training run, if present."""
        if not _DEMO_EXAMPLES_PATH.exists():
            return []
        payload: list[dict[str, Any]] = json.loads(
            _DEMO_EXAMPLES_PATH.read_text(encoding="utf-8")
        )
        return [DemoExample(**entry) for entry in payload]

    def _validate_classes(self) -> None:
        """Fail at startup, not mid-request, if the artifact is not binary 0/1."""
        classes = list(self._pipeline.classes_)
        if sorted(int(label) for label in classes) != [0, 1]:
            raise artifacts.ArtifactError(
                f"Expected a binary 0/1 classifier; artifact reports classes {classes}."
            )
