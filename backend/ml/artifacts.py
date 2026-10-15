"""Locations and I/O for the trained model artifact and its metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
FIGURES_DIR = ARTIFACTS_DIR / "figures"


class ArtifactError(RuntimeError):
    """Raised when the model artifact is missing or unreadable."""


@dataclass(frozen=True)
class LoadedModel:
    """A fitted pipeline together with the metadata saved alongside it."""

    pipeline: Pipeline
    metadata: dict[str, Any]


def save(pipeline: Pipeline, metadata: dict[str, Any]) -> None:
    """Write the fitted pipeline and its metadata to the artifacts directory."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def load() -> LoadedModel:
    """Load the fitted pipeline and metadata, raising ``ArtifactError`` if absent."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise ArtifactError(
            f"No model artifact in {ARTIFACTS_DIR}. "
            "Train one first: python -m ml.train (run from the backend/ directory)."
        )
    try:
        pipeline = joblib.load(MODEL_PATH)
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except Exception as error:  # pragma: no cover - corrupt artifact
        raise ArtifactError(f"Could not load the model artifact: {error}") from error
    return LoadedModel(pipeline=pipeline, metadata=metadata)
