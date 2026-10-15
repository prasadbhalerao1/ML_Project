"""Shared test fixtures."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from ml import artifacts


@pytest.fixture(scope="session")
def client() -> TestClient:
    """A client whose lifespan has run, so the model artifact is loaded."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def requires_artifact() -> None:
    """Skip a test when no trained model is present on disk."""
    if not artifacts.MODEL_PATH.exists():
        pytest.skip("No model artifact; run `python -m ml.train` first.")


@pytest.fixture
def valid_payload() -> dict[str, float]:
    """A complete, plausible request body keyed by dataset column names."""
    return {
        "RevolvingUtilizationOfUnsecuredLines": 0.25,
        "age": 45,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.35,
        "MonthlyIncome": 5400,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1,
    }
