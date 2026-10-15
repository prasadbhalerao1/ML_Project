# API reference

FastAPI service that loads the trained pipeline once at startup and scores one
borrower per request.

| | |
|---|---|
| Local | `http://127.0.0.1:8000` |
| Hosted | `https://loan-default-api.vercel.app` |
| Through the website | `https://delinquency-risk.vercel.app/api/*` (same-origin proxy) |
| Interactive docs | `/docs` (Swagger UI), `/openapi.json` |

The bounds and required/optional flags below were read from the generated
OpenAPI schema, not written by hand.

## Endpoints

| Method | Path | Purpose | Needs a model |
|---|---|---|---|
| `GET` | `/health` | Liveness and whether a model is loaded | no |
| `GET` | `/model-info` | Model, dataset, metrics, chart data | yes |
| `GET` | `/demo-examples` | Two real held-out records for the form | yes |
| `POST` | `/predict` | Score one borrower | yes |

Endpoints marked *yes* return **503** with an explanatory `detail` when no
artifact is present, rather than failing at startup.

---

## `GET /health`

```json
{ "status": "ok", "model_loaded": true, "model_version": "1.0.0", "detail": null }
```

`detail` carries the reason when `model_loaded` is `false`. Use this as the
readiness check — it never 503s.

---

## `GET /model-info`

Everything the frontend needs to describe the model honestly. Notable fields:

| Field | Meaning |
|---|---|
| `features` | The 10 input column names, in training order |
| `class_order` | `predict_proba` column order, i.e. `[0, 1]` |
| `classification_threshold` | The cut-off actually applied, from the artifact |
| `risk_bands` | Display buckets with bounds; the top band's upper bound is `null` |
| `test_metrics` | Held-out metrics at that threshold |
| `test_threshold_sweep` | Four reference thresholds |
| `test_threshold_curve` | Dense curve with precision, recall, F1 **and** confusion counts |
| `test_score_distribution` | Histogram of test scores split by actual class |
| `class_distribution` | Row counts and positive rate per split |
| `feature_importances` | All 16 model input columns, descending |

`test_threshold_curve` and `test_score_distribution` exist to drive the charts
in the web app; see [Model evaluation](05-model-evaluation.md).

---

## `POST /predict`

### Request

JSON keys are the dataset's own column names, so a request body matches
`/model-info`'s `features` list exactly. Unknown keys are rejected.

| Field | Type | Required | Range |
|---|---|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | number | yes | 0 – 100000 |
| `age` | integer | yes | 18 – 120 |
| `NumberOfTime30-59DaysPastDueNotWorse` | integer | yes | 0 – 100 |
| `DebtRatio` | number | yes | 0 – 100000 |
| `MonthlyIncome` | number | **no** | 0 – 10000000 |
| `NumberOfOpenCreditLinesAndLoans` | integer | yes | 0 – 100 |
| `NumberOfTimes90DaysLate` | integer | yes | 0 – 100 |
| `NumberRealEstateLoansOrLines` | integer | yes | 0 – 100 |
| `NumberOfTime60-89DaysPastDueNotWorse` | integer | yes | 0 – 100 |
| `NumberOfDependents` | integer | **no** | 0 – 30 |

The two optional fields may be omitted or sent as `null`. The pipeline's median
imputer fills them exactly as it did during training, and the missing-value
indicator records that they were absent — so a null is handled, not guessed at.

The ratio bounds look generous because the raw data is: `DebtRatio` reaches
329,664 where income is unrecorded. The bounds reject impossible values without
rejecting the genuine extremes the model was trained on.

```bash
curl -X POST https://delinquency-risk.vercel.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "RevolvingUtilizationOfUnsecuredLines": 0.25,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.35,
    "MonthlyIncome": 5400,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 1
  }'
```

### Response

```json
{
  "default_probability": 0.00815693624730604,
  "predicted_class": 0,
  "predicted_label": "No serious delinquency",
  "classification_threshold": 0.2300842443936961,
  "risk_band": "Low",
  "risk_band_disclaimer": "Risk bands are illustrative display buckets over the model's predicted probability, not empirically validated lending thresholds.",
  "model_version": "1.0.0",
  "target_definition": "Borrower experienced serious delinquency (90 days past due or worse) within two years of the observation date"
}
```

The response separates three things on purpose:

- **`default_probability`** — the model's estimate. The mean of the per-tree
  leaf proportions. **Not calibrated**; treat it as a ranking score.
- **`predicted_class`** — purely whether the probability reaches
  `classification_threshold`. The threshold is returned so a caller can
  re-derive the class or apply its own.
- **`risk_band`** — a display bucket, with the disclaimer attached to every
  response.

---

## Errors

| Status | When | Body |
|---|---|---|
| `422` | A field is missing, out of range, non-numeric, non-finite, or unknown | `{"detail": [{"loc": [...], "msg": "...", "type": "...", "input": ...}]}` |
| `503` | No model artifact loaded | `{"detail": "No model artifact in ... Train one first: python -m ml.train"}` |

`NaN` and `Infinity` are rejected with a 422 like any other invalid value. That
needs a custom handler: FastAPI's default 422 body echoes the offending input,
and a non-finite float cannot be serialised to JSON, so the default response
would fail with a 500 instead. `app/api/errors.py` renders such inputs as
strings so the client gets a readable 422 naming the field.

---

## CORS

Configured through the `CORS_ORIGINS` environment variable (comma-separated).
It currently allows the two frontend production domains and the local dev
server; any other origin is rejected at preflight.

In normal use this never comes into play — the website calls the API through
its own origin. CORS matters only for direct calls to
`loan-default-api.vercel.app`.
