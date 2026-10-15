"""Request and response schemas for the prediction endpoints.

JSON keys are the dataset's own column names, so a request body matches the
``features`` list returned by ``/model-info`` exactly. Python attributes use
snake_case because two of the column names contain hyphens.

Bounds come from the competition data dictionary and the observed ranges in
cs-training.csv (see docs/dataset-research.md); they reject impossible input
without rejecting the genuine extremes present in the data.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

#: Largest accepted value for the two ratio features. Both have heavy right
#: tails in the raw data (DebtRatio reaches ~330,000 where income is
#: unrecorded), so the bound only rules out values the dataset never contains.
MAX_RATIO = 100_000.0

#: Largest accepted value for the count features. The highest genuine count in
#: the training data is 58 open credit lines.
MAX_COUNT = 100

#: Upper bounds for the two optional fields.
MAX_MONTHLY_INCOME = 10_000_000.0
MAX_DEPENDENTS = 30


class PredictionRequest(BaseModel):
    """The ten predictors the trained pipeline expects."""

    #: Reject NaN/Infinity, accept either the alias or the Python name, and
    #: refuse unknown keys so a typo surfaces as a 422 rather than being
    #: silently ignored.
    model_config = ConfigDict(
        allow_inf_nan=False,
        populate_by_name=True,
        extra="forbid",
    )

    revolving_utilization_of_unsecured_lines: Annotated[
        float,
        Field(
            ge=0,
            le=MAX_RATIO,
            alias="RevolvingUtilizationOfUnsecuredLines",
            description=(
                "Balance on credit cards and personal lines of credit divided by "
                "total credit limits. 0.3 means 30% of available credit is used. "
                "Values above 1 occur in the data for over-limit accounts."
            ),
            examples=[0.25],
        ),
    ]
    age: Annotated[
        int,
        Field(ge=18, le=120, description="Borrower age in years.", examples=[45]),
    ]
    number_of_time_30_59_days_past_due_not_worse: Annotated[
        int,
        Field(
            ge=0,
            le=MAX_COUNT,
            alias="NumberOfTime30-59DaysPastDueNotWorse",
            description="Times 30-59 days past due, but no worse, in the last two years.",
            examples=[0],
        ),
    ]
    debt_ratio: Annotated[
        float,
        Field(
            ge=0,
            le=MAX_RATIO,
            alias="DebtRatio",
            description=(
                "Monthly debt payments, alimony and living costs divided by monthly "
                "gross income. Very large values appear in the data where income is "
                "unrecorded."
            ),
            examples=[0.35],
        ),
    ]
    monthly_income: Annotated[
        float | None,
        Field(
            ge=0,
            le=MAX_MONTHLY_INCOME,
            alias="MonthlyIncome",
            description=(
                "Monthly gross income. Optional: it is missing for about 20% of the "
                "training rows, and the pipeline's median imputer fills a null the "
                "same way it did during training."
            ),
            examples=[5400],
        ),
    ] = None
    number_of_open_credit_lines_and_loans: Annotated[
        int,
        Field(
            ge=0,
            le=MAX_COUNT,
            alias="NumberOfOpenCreditLinesAndLoans",
            description=(
                "Open installment loans plus lines of credit, e.g. car loans and "
                "credit cards."
            ),
            examples=[8],
        ),
    ]
    number_of_times_90_days_late: Annotated[
        int,
        Field(
            ge=0,
            le=MAX_COUNT,
            alias="NumberOfTimes90DaysLate",
            description="Times 90 or more days past due.",
            examples=[0],
        ),
    ]
    number_real_estate_loans_or_lines: Annotated[
        int,
        Field(
            ge=0,
            le=MAX_COUNT,
            alias="NumberRealEstateLoansOrLines",
            description=(
                "Mortgages and real estate loans, including home equity lines of credit."
            ),
            examples=[1],
        ),
    ]
    number_of_time_60_89_days_past_due_not_worse: Annotated[
        int,
        Field(
            ge=0,
            le=MAX_COUNT,
            alias="NumberOfTime60-89DaysPastDueNotWorse",
            description="Times 60-89 days past due, but no worse, in the last two years.",
            examples=[0],
        ),
    ]
    number_of_dependents: Annotated[
        int | None,
        Field(
            ge=0,
            le=MAX_DEPENDENTS,
            alias="NumberOfDependents",
            description=(
                "Dependents in the family, excluding the borrower. Optional, and "
                "median-imputed when omitted."
            ),
            examples=[1],
        ),
    ] = None

    def to_feature_record(self) -> dict[str, Any]:
        """Return the values keyed by their dataset column names."""
        return self.model_dump(by_alias=True)


class PredictionResponse(BaseModel):
    """One prediction, with the model metadata needed to interpret it."""

    default_probability: float = Field(
        description=(
            "Model-estimated probability of serious delinquency. A forest vote "
            "share, not a calibrated real-world probability."
        ),
        examples=[0.23],
    )
    predicted_class: int = Field(
        description="1 if default_probability >= classification_threshold, else 0.",
        examples=[0],
    )
    predicted_label: str = Field(
        description="The predicted class in words.",
        examples=["No serious delinquency"],
    )
    classification_threshold: float = Field(
        description="Probability cut-off used to derive predicted_class.", examples=[0.5]
    )
    risk_band: str = Field(
        description="Illustrative display bucket for the probability.", examples=["Medium"]
    )
    risk_band_disclaimer: str
    model_version: str = Field(examples=["1.0.0"])
    target_definition: str


class HealthResponse(BaseModel):
    """Liveness plus whether a usable model is loaded."""

    status: str = Field(examples=["ok"])
    model_loaded: bool
    model_version: str | None
    detail: str | None = Field(
        default=None, description="Why the model is unavailable, when it is not loaded."
    )


class MetricsSummary(BaseModel):
    """Test-set metrics produced by the training run."""

    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: dict[str, int]


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class ScoreDistribution(BaseModel):
    """Histogram of test-set predicted probabilities, split by actual class.

    Lets the UI show how far the two classes' scores overlap, which is what
    explains the recall ceiling. ``bin_edges`` has one more entry than either
    count list.
    """

    bin_edges: list[float]
    negative_counts: list[int]
    positive_counts: list[int]


class ThresholdCurvePoint(BaseModel):
    """One point on the precision/recall/F1-versus-threshold curve.

    Carries the raw confusion counts as well as the rates, so a reader can see
    what a given cut-off would cost in people rather than only in percentages.
    """

    threshold: float
    precision: float
    recall: float
    f1: float
    flagged: int
    true_positives: int
    false_positives: int
    false_negatives: int


class ModelInfoResponse(BaseModel):
    """Everything the frontend needs to describe the model honestly."""

    model_name: str
    model_version: str
    algorithm: str
    trained_at: str
    dataset_name: str
    dataset_url: str
    target_column: str
    target_definition: str
    features: list[str]
    class_order: list[int]
    classification_threshold: float
    risk_bands: list[dict[str, float | str | None]]
    risk_band_disclaimer: str
    test_metrics: MetricsSummary
    test_threshold_sweep: list[MetricsSummary]
    test_threshold_curve: list[ThresholdCurvePoint]
    test_score_distribution: ScoreDistribution
    class_distribution: dict[str, dict[str, float]]
    feature_importances: list[FeatureImportance]


class DemoExample(BaseModel):
    """A real held-out dataset row offered as demonstration input."""

    label: str
    actual_outcome: int = Field(
        description="The recorded label for this borrower in the dataset."
    )
    features: dict[str, float]
