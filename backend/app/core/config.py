"""Application settings and the documented risk-band thresholds."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

API_TITLE = "Loan Delinquency Risk API"
API_DESCRIPTION = (
    "Educational prototype. Serves a Random Forest trained on the Kaggle "
    "'Give Me Some Credit' dataset to estimate the probability that a borrower "
    "experiences serious delinquency (90+ days past due) within two years. "
    "Not a credit-underwriting system."
)
API_VERSION = "1.0.0"

#: Origins allowed to call the API from a browser. Vite dev server by default;
#: override with a comma-separated CORS_ORIGINS environment variable.
DEFAULT_CORS_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


def cors_origins() -> list[str]:
    configured = os.getenv("CORS_ORIGINS", "").strip()
    if not configured:
        return list(DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


@dataclass(frozen=True)
class RiskBand:
    """An illustrative label for a range of predicted probabilities."""

    name: str
    upper_bound_exclusive: float


#: Presentation-only buckets for the predicted probability, ordered by bound.
#: These cut-offs are chosen for readability in the UI. They are NOT validated
#: lending thresholds and carry no regulatory or business meaning.
RISK_BANDS: tuple[RiskBand, ...] = (
    RiskBand(name="Low", upper_bound_exclusive=0.10),
    RiskBand(name="Medium", upper_bound_exclusive=0.30),
    RiskBand(name="High", upper_bound_exclusive=float("inf")),
)

RISK_BAND_DISCLAIMER = (
    "Risk bands are illustrative display buckets over the model's predicted "
    "probability, not empirically validated lending thresholds."
)

#: Fallback cut-off, used only if an artifact predates threshold selection.
#: The real threshold is chosen on the validation split during training and
#: stored in metadata.json, so the served model and the reported metrics can
#: never disagree about it.
FALLBACK_CLASSIFICATION_THRESHOLD = 0.5


def risk_band_for(probability: float) -> str:
    """Map a predicted probability to its illustrative band name."""
    for band in RISK_BANDS:
        if probability < band.upper_bound_exclusive:
            return band.name
    return RISK_BANDS[-1].name


def risk_band_definitions() -> list[dict[str, float | str | None]]:
    """Describe the bands for the /model-info response, so the UI need not guess.

    The open-ended top band reports ``None`` as its upper bound, because
    infinity has no JSON representation.
    """
    definitions: list[dict[str, float | str | None]] = []
    lower = 0.0
    for band in RISK_BANDS:
        upper = band.upper_bound_exclusive
        definitions.append(
            {
                "name": band.name,
                "lower_bound_inclusive": lower,
                "upper_bound_exclusive": None if math.isinf(upper) else upper,
            }
        )
        lower = upper
    return definitions
