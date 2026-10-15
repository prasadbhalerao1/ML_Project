"""HTTP routes. All model work is delegated to ``app.services.predictor``."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.model_registry import registry
from app.schemas.prediction import (
    DemoExample,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.services.predictor import Predictor

router = APIRouter()


def require_predictor() -> Predictor:
    """Dependency that 503s when no model artifact is loaded."""
    if not registry.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=registry.error or "Model is not loaded.",
        )
    assert registry.predictor is not None  # narrowed by is_loaded
    return registry.predictor


LoadedPredictor = Annotated[Predictor, Depends(require_predictor)]


@router.get("/health", response_model=HealthResponse, tags=["status"])
def health() -> HealthResponse:
    """Report API liveness and whether a model is available."""
    predictor = registry.predictor
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        model_version=predictor.model_version if predictor else None,
        detail=registry.error,
    )


@router.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
def model_info(predictor: LoadedPredictor) -> ModelInfoResponse:
    """Describe the trained model, its dataset and its measured test metrics."""
    return predictor.model_info()


@router.get("/demo-examples", response_model=list[DemoExample], tags=["model"])
def demo_examples(predictor: LoadedPredictor) -> list[DemoExample]:
    """Real held-out dataset rows, for pre-filling the prediction form.

    Demonstration data from the public dataset; not information about any
    current user.
    """
    return predictor.demo_examples


@router.post("/predict", response_model=PredictionResponse, tags=["model"])
def predict(request: PredictionRequest, predictor: LoadedPredictor) -> PredictionResponse:
    """Estimate the probability of serious delinquency for one borrower.

    Invalid or missing fields are rejected by Pydantic with a 422 listing each
    offending field.
    """
    try:
        return predictor.predict(request)
    except ValueError as error:  # e.g. an artifact without a positive class
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {error}",
        ) from error
