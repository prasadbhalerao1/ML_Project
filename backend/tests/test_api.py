"""End-to-end API tests against the saved model artifact.

Tests needing a trained model use the ``requires_artifact`` fixture, so the
suite still runs (with skips) on a clean checkout before training.
"""

from __future__ import annotations

import pytest

from ml.dataset import FEATURES


def test_health_reports_status_and_model_availability(client) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["model_loaded"], bool)


def test_model_info_matches_the_trained_feature_schema(client, requires_artifact) -> None:
    response = client.get("/model-info")

    assert response.status_code == 200
    body = response.json()
    assert body["features"] == list(FEATURES)
    assert body["algorithm"].startswith("RandomForestClassifier")
    assert body["dataset_url"].startswith("https://www.kaggle.com/competitions/GiveMeSomeCredit")
    assert body["class_order"] == [0, 1]
    assert 0 < body["test_metrics"]["roc_auc"] <= 1


def test_predict_returns_a_probability_and_a_consistent_class(
    client, requires_artifact, valid_payload
) -> None:
    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    # The class must follow the threshold the response itself reports.
    assert body["predicted_class"] == int(
        body["default_probability"] >= body["classification_threshold"]
    )
    assert 0.0 < body["classification_threshold"] < 1.0
    assert body["risk_band"] in {"Low", "Medium", "High"}
    assert body["target_definition"]


def test_served_threshold_matches_the_artifact(client, requires_artifact, valid_payload) -> None:
    """The API must apply the threshold chosen at training time, not a default."""
    from ml import artifacts

    expected = artifacts.load().metadata["decision_threshold"]["value"]

    assert client.get("/model-info").json()["classification_threshold"] == pytest.approx(
        expected
    )
    assert client.post("/predict", json=valid_payload).json()[
        "classification_threshold"
    ] == pytest.approx(expected)


def test_predict_accepts_omitted_optional_fields(
    client, requires_artifact, valid_payload
) -> None:
    payload = {
        key: value
        for key, value in valid_payload.items()
        if key not in {"MonthlyIncome", "NumberOfDependents"}
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200


def test_a_clearly_delinquent_profile_scores_above_a_clean_one(
    client, requires_artifact, valid_payload
) -> None:
    risky = {
        **valid_payload,
        "NumberOfTimes90DaysLate": 6,
        "NumberOfTime60-89DaysPastDueNotWorse": 3,
        "RevolvingUtilizationOfUnsecuredLines": 0.98,
    }

    clean_probability = client.post("/predict", json=valid_payload).json()[
        "default_probability"
    ]
    risky_probability = client.post("/predict", json=risky).json()["default_probability"]

    assert risky_probability > clean_probability


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
            ("age", 5, "below the minimum plausible age"),
            ("age", 500, "above the maximum plausible age"),
            ("RevolvingUtilizationOfUnsecuredLines", -1, "negative ratio"),
            ("NumberOfTimes90DaysLate", -2, "negative count"),
            ("MonthlyIncome", -100, "negative income"),
            ("NumberOfDependents", 999, "implausible dependent count"),
    ],
)
def test_predict_rejects_out_of_range_values(
    client, requires_artifact, valid_payload, field, value, reason
) -> None:
    response = client.post("/predict", json={**valid_payload, field: value})

    assert response.status_code == 422, reason
    assert any(field in str(error["loc"]) for error in response.json()["detail"])


def test_predict_rejects_a_missing_required_field(
    client, requires_artifact, valid_payload
) -> None:
    payload = {k: v for k, v in valid_payload.items() if k != "age"}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert any("age" in str(error["loc"]) for error in response.json()["detail"])


@pytest.mark.parametrize("non_finite", ["NaN", "Infinity", "-Infinity"])
def test_predict_rejects_non_finite_numbers(
    client, requires_artifact, valid_payload, non_finite
) -> None:
    body = ", ".join(
        f'"{key}": {non_finite if key == "DebtRatio" else value}'
        for key, value in valid_payload.items()
    )

    response = client.post(
        "/predict",
        content="{" + body + "}",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422


def test_predict_rejects_unknown_fields(client, requires_artifact, valid_payload) -> None:
    response = client.post("/predict", json={**valid_payload, "LoanAmount": 10_000})

    assert response.status_code == 422


def test_predict_rejects_a_non_numeric_value(client, requires_artifact, valid_payload) -> None:
    response = client.post("/predict", json={**valid_payload, "age": "forty"})

    assert response.status_code == 422


def test_demo_examples_are_real_records_with_the_full_feature_set(
    client, requires_artifact
) -> None:
    response = client.get("/demo-examples")

    assert response.status_code == 200
    examples = response.json()
    assert examples, "training should have saved demonstration records"
    for example in examples:
        assert set(example["features"]) == set(FEATURES)
        assert example["actual_outcome"] in {0, 1}


def test_swagger_schema_is_served(client) -> None:
    assert client.get("/openapi.json").status_code == 200


def test_model_info_exposes_chart_data(client, requires_artifact) -> None:
    """The UI charts need distribution and curve data, not just scalars."""
    body = client.get("/model-info").json()

    dist = body["test_score_distribution"]
    assert len(dist["bin_edges"]) == len(dist["negative_counts"]) + 1
    assert len(dist["negative_counts"]) == len(dist["positive_counts"])
    assert sum(dist["positive_counts"]) > 0 and sum(dist["negative_counts"]) > 0

    curve = body["test_threshold_curve"]
    assert len(curve) > 10
    assert [p["threshold"] for p in curve] == sorted(p["threshold"] for p in curve)
    for point in curve:
        assert 0.0 <= point["precision"] <= 1.0
        assert 0.0 <= point["recall"] <= 1.0
