"""Tests that the saved artifact loads and scores independently of training."""

from __future__ import annotations

import pandas as pd
import pytest

from app.core.config import risk_band_for
from app.services.predictor import Predictor
from ml import artifacts
from ml.dataset import FEATURES
from ml.metrics import positive_class_proba
from ml.preprocessing import as_feature_frame

pytestmark = pytest.mark.usefixtures("requires_artifact")


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    return Predictor.from_artifacts()


def test_artifact_loads_without_importing_train(predictor: Predictor) -> None:
    assert predictor.features == list(FEATURES)
    assert predictor.model_version


def test_metadata_records_real_metrics_and_split_sizes() -> None:
    metadata = artifacts.load().metadata

    test_metrics = metadata["test_metrics"]
    assert 0.5 < test_metrics["roc_auc"] <= 1.0
    assert 0.0 < test_metrics["pr_auc"] <= 1.0

    distribution = metadata["class_distribution"]
    assert distribution["train"]["total"] > distribution["test"]["total"]
    for part in distribution.values():
        # Stratification should keep every split near the 6.7% base rate.
        assert 0.05 < part["positive_rate"] < 0.09


def test_saved_pipeline_scores_a_raw_dataframe() -> None:
    pipeline = artifacts.load().pipeline
    frame = as_feature_frame([{feature: 1.0 for feature in FEATURES}])

    probabilities = positive_class_proba(pipeline, frame)

    assert probabilities.shape == (1,)
    assert 0.0 <= probabilities[0] <= 1.0


def test_pipeline_handles_placeholder_codes_at_inference_time() -> None:
    """A 98 in a counter must be imputed, not fed to the forest as a count."""
    pipeline = artifacts.load().pipeline
    record = {feature: 0.0 for feature in FEATURES}
    frame = as_feature_frame([{**record, "NumberOfTimes90DaysLate": 98}])

    transformed: pd.DataFrame = pipeline[:-1].transform(frame)

    assert transformed["NumberOfTimes90DaysLate"].iloc[0] != 98


@pytest.mark.parametrize(
    ("probability", "expected_band"),
    [(0.0, "Low"), (0.099, "Low"), (0.10, "Medium"), (0.29, "Medium"), (0.30, "High"), (1.0, "High")],
)
def test_risk_bands_cover_the_probability_range(probability: float, expected_band: str) -> None:
    assert risk_band_for(probability) == expected_band
