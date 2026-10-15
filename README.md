# Loan Delinquency Risk Prediction

An end-to-end machine learning project: a Random Forest classifier that
estimates the probability a borrower experiences **serious delinquency — 90
days past due or worse — within two years**, served through a FastAPI backend
and a React + TypeScript frontend.

> **Educational prototype.** This is a student project built on a public 2011
> dataset. It is not a credit-underwriting system and must not be used to
> approve, decline or price real credit. See
> [Limitations](documentation/09-limitations.md).

## Live demo

| | |
|---|---|
| Website | <https://delinquency-risk.vercel.app> |
| API (Swagger UI) | <https://loan-default-api.vercel.app/docs> |

The first request after a quiet period takes 8 to 9 seconds while the model
loads. How it is deployed, and why there are no CORS errors:
[Deployment](documentation/08-deployment.md).

## Contents

- [Objective and scope](#objective-and-scope)
- [Dataset](#dataset)
- [Results](#results)
- [Architecture](#architecture)
- [Setup](#setup) — full guide in [Setup](documentation/02-setup.md)
- [API](#api)
- [Tests](#tests)
- [Methodology summary](#methodology-summary)
- [Limitations](#limitations)
- [Future improvements](#future-improvements)

## Objective and scope

**In scope:** dataset research and documentation, data inspection, leakage
checking, preprocessing without train/serve skew, Random Forest training and
honest evaluation, a REST API, and a four-page web interface.

**Out of scope, deliberately:** other algorithms, extensive hyperparameter
search, databases, authentication, containers, or any deployment machinery. The
goal is to implement and understand one model properly.

**What the model predicts:** one specific historical outcome, `SeriousDlqin2yrs`
— whether a borrower experienced 90+ days past due delinquency within two years
of the observation date. It does not predict loan default in general and it
does not say whether a borrower will repay.

## Dataset

**Give Me Some Credit** — Kaggle competition
**<https://www.kaggle.com/competitions/GiveMeSomeCredit/data>**

| | |
|---|---|
| Rows used | 150,000 labeled (`cs-training.csv`) |
| Features | 10 numeric predictors |
| Target | `SeriousDlqin2yrs`, 6.68% positive |
| Data credited to | Credit Fusion and Kaggle |
| Access | Kaggle account + acceptance of the competition rules |

Full research, the comparison against two alternative datasets, and the
inspected schema: [Dataset](documentation/03-dataset.md).

### Download the data

The raw CSVs are **not** committed — download them yourself.

**Browser:** sign in to Kaggle, open
<https://www.kaggle.com/competitions/GiveMeSomeCredit/data>, accept the rules
on the **Rules** tab (downloads are blocked until you do), and save
`cs-training.csv` to `data/raw/GiveMeSomeCredit/`.

**Kaggle API:**

```bash
pip install kaggle
# Kaggle → Settings → API → "Create New Token", save kaggle.json to
#   Windows: C:\Users\<you>\.kaggle\kaggle.json
#   macOS/Linux: ~/.kaggle/kaggle.json  (chmod 600)
kaggle competitions download -c GiveMeSomeCredit -p data/raw/GiveMeSomeCredit
cd data/raw/GiveMeSomeCredit && unzip GiveMeSomeCredit.zip
```

The API returns 403 until the rules have been accepted in a browser once.

Expected result:

```
data/raw/GiveMeSomeCredit/
├── cs-training.csv        # the only file used for modelling
├── cs-test.csv            # unlabeled; not used
├── Data Dictionary.xls
└── sampleEntry.csv
```

### Licence and rules

The data is distributed under Kaggle's **competition rules** for Give Me Some
Credit, not an open-data licence. Read them on the competition's Rules tab
before using the data beyond private study. This project does not redistribute
the data: everyone downloads it from Kaggle themselves. The exact 2011 rules
text was not independently verified here — see
[Dataset](documentation/03-dataset.md#licence-attribution-and-competition-rules).

## Results

Reproducible with seed 42. Decision threshold **0.2302**, selected on the
validation split (not assumed to be 0.50 — see below). The final model was
refitted on train + validation, so those are in-sample; **only the test column
is a held-out estimate.**

| Metric | Train (in-sample) | Validation (in-sample) | **Test (held out)** |
|---|---:|---:|---:|
| ROC-AUC | 0.9329 | 0.9333 | **0.8654** |
| PR-AUC | 0.5232 | 0.5105 | **0.4014** |
| Accuracy | 0.9264 | 0.9265 | **0.9199** |
| Precision | 0.4580 | 0.4565 | **0.4153** |
| Recall | 0.5472 | 0.5267 | **0.4868** |
| F1 | 0.4986 | 0.4891 | **0.4482** |

Test confusion matrix: **TN 26,621 · FP 1,374 · FN 1,029 · TP 976**. Of 2,005
borrowers who really did become seriously delinquent, the model caught 976.

### Classification report (test split)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 — No serious delinquency | 0.9628 | 0.9509 | 0.9568 | 27,995 |
| 1 — Serious delinquency | 0.4153 | 0.4868 | 0.4482 | 2,005 |
| **Accuracy** | | | **0.9199** | 30,000 |
| Macro avg | 0.6891 | 0.7189 | 0.7025 | 30,000 |
| Weighted avg | 0.9262 | 0.9199 | 0.9228 | 30,000 |

Use **macro avg F1 (0.7025)** or the **positive-class F1 (0.4482)** as a
headline — never the weighted average (0.9228), which is inflated by the 93%
majority class.

### Threshold selection

0.50 is a bad cut-off at a 6.68% base rate. The threshold is chosen by
maximising F1 on the **validation** split, using the candidate fitted on train
only, so the test split plays no part. It is stored in `metadata.json` and read
by the API at startup.

| | Precision | Recall | F1 |
|---|---:|---:|---:|
| Validation @ 0.2302 (selected) | 0.4127 | 0.4703 | 0.4396 |
| **Test @ 0.2302** | **0.4153** | **0.4868** | **0.4482** |
| Test @ 0.50 (conventional) | 0.6267 | 0.1566 | 0.2506 |

Validation F1 0.4396 → test F1 0.4482: the threshold transfers to unseen data.

| Threshold | Accuracy | Precision | Recall | F1 | Flagged |
|---:|---:|---:|---:|---:|---:|
| 0.1000 | 0.8469 | 0.2610 | 0.7047 | 0.3810 | 5,413 |
| 0.2000 | 0.9131 | 0.3884 | 0.5222 | 0.4454 | 2,696 |
| **0.2302 (in use)** | **0.9199** | **0.4153** | **0.4868** | **0.4482** | **2,350** |
| 0.3000 | 0.9292 | 0.4650 | 0.3970 | 0.4283 | 1,712 |
| 0.5000 | 0.9374 | 0.6267 | 0.1566 | 0.2506 | 501 |

### Generalisation and stability

**Gap (train − test):** ROC-AUC +0.0675, PR-AUC +0.1218. The forest still
overfits despite `min_samples_leaf=20`; the honest figure is the test column.

**5-fold cross-validation** on train + validation (test excluded, same
threshold) confirms the test score is stable, not a lucky split:

| Metric | CV mean ± std | Test |
|---|---:|---:|
| ROC-AUC | 0.8632 ± 0.0053 | 0.8654 |
| PR-AUC | 0.3978 ± 0.0148 | 0.4014 |
| F1 | 0.4415 ± 0.0130 | 0.4482 |
| Recall | 0.4792 ± 0.0120 | 0.4868 |

Every test metric falls within one standard deviation of its CV mean.

**Read accuracy carefully.** Always predicting "no delinquency" scores 0.9332
on this test set. Tuning the threshold deliberately *lowered* accuracy from
0.9374 to 0.9199 while tripling recall — the right trade here, and the reason
accuracy must not drive the decision. ROC-AUC and PR-AUC (no-skill baseline
0.0668) are the figures that carry information.

Full discussion, class weighting, feature importance and the false
positive/negative trade-off:
[Model evaluation](documentation/05-model-evaluation.md).

## Architecture

```
┌──────────────────┐      HTTP/JSON       ┌──────────────────┐
│  React + Vite    │ ───────────────────► │  FastAPI         │
│  TypeScript      │ ◄─────────────────── │  Pydantic        │
│  Tailwind CSS    │                      │  predictor svc   │
└──────────────────┘                      └────────┬─────────┘
                                                   │ loads once at startup
                                                   ▼
                                       ┌───────────────────────┐
                                       │ model.joblib          │
                                       │  cleaner → imputer    │
                                       │  → RandomForest       │
                                       └───────────────────────┘
```

| Layer | Technology |
|---|---|
| Model | scikit-learn `RandomForestClassifier`, joblib |
| Data | pandas, NumPy; matplotlib + seaborn for figures |
| API | FastAPI, Pydantic v2, Uvicorn |
| Frontend | React 19, Vite, TypeScript, Tailwind CSS v4, React Router |
| Tests | pytest |

```
LoanDefault/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app, CORS, lifespan model load
│   │   ├── api/
│   │   │   ├── routes.py           # endpoints only; no model logic
│   │   │   └── errors.py           # JSON-safe 422 for non-finite input
│   │   ├── core/
│   │   │   ├── config.py           # settings, risk bands, threshold
│   │   │   └── model_registry.py   # holds the single Predictor instance
│   │   ├── schemas/prediction.py   # Pydantic request/response models
│   │   └── services/predictor.py   # the only code that touches the pipeline
│   ├── ml/
│   │   ├── dataset.py              # schema contract + loading
│   │   ├── preprocessing.py        # cleaner + pipeline builder
│   │   ├── splits.py               # deterministic 60/20/20
│   │   ├── metrics.py              # metrics + safe predict_proba lookup
│   │   ├── artifacts.py            # artifact save/load
│   │   ├── explore.py              # inspection + EDA figures
│   │   ├── train.py                # training entry point
│   │   ├── evaluate.py             # per-split report, gap, cross-validation
│   │   └── artifacts/              # model.joblib, metadata.json, figures/
│   ├── tests/
│   ├── pytest.ini
│   ├── requirements.txt         # runtime only: what the hosted API installs
│   └── requirements-dev.txt     # + training, plotting and test tools
├── frontend/
│   ├── src/
│   │   ├── components/             # Layout, ui, NumberField, ProbabilityBar
│   │   │   └── charts/             # ScoreSeparationChart, ThresholdCurveChart (hand-rolled SVG)
│   │   ├── hooks/                  # useAsync, usePrediction, usePredictionForm
│   │   ├── pages/                  # Dashboard, Predict, Result, ModelInfo
│   │   ├── services/               # api.ts, featureFields.ts
│   │   ├── types/api.ts
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── .env.example
│   └── package.json
├── data/raw/GiveMeSomeCredit/      # git-ignored; download from Kaggle
├── documentation/                  # reference docs + SVG diagrams
│   ├── 01-overview.md … 09-limitations.md
│   └── diagrams/
├── dev/                            # working material, not part of the system
│   ├── task_claude.md              # the original brief
│   ├── deck/                       # seminar slides + print.pdf handout
│   ├── report/                     # the written report (docx + pdf)
│   └── build-scripts/              # generators for deck, sheet and report
└── README.md
```

**Design note.** `backend/ml/dataset.py` is the single source of truth for
feature names; the fitted pipeline carries its own preprocessing, so the API
cannot apply a different transformation than training did.

## Setup

**Full step-by-step instructions, including troubleshooting:
[Setup](documentation/02-setup.md).** Quickstart below.

**Prerequisites:** Python 3.11+ (built on 3.14), Node.js 18+ (built on 22), and
the Kaggle data downloaded as described above.

```bash
# 1. Install (from the repository root)
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate          # macOS/Linux
pip install -r backend/requirements-dev.txt

# 2. Train the model (about a minute)
cd backend
python -m ml.train

# 3. Start the API  — leave running
uvicorn app.main:app --reload --port 8000

# 4. Start the frontend — in a second terminal
cd frontend
npm install
npm run dev
```

Then open <http://localhost:5173>. Swagger UI is at
<http://127.0.0.1:8000/docs>.

Every backend command runs **from the `backend/` directory**, so that `ml.*`
and `app.*` resolve as packages. The frontend needs no configuration: it
defaults to `http://127.0.0.1:8000`, and `frontend/.env.example` shows how to
override that.

### Other commands

| Task | Directory | Command |
|---|---|---|
| Inspect the data, write EDA charts | `backend` | `python -m ml.explore` |
| Per-split metrics + classification report | `backend` | `python -m ml.evaluate` |
| The above plus 5-fold cross-validation | `backend` | `python -m ml.evaluate --cross-validate` |
| Run the test suite | `backend` | `pytest` |
| Type-check and build the frontend | `frontend` | `npm run typecheck && npm run build` |

`ml.train` accepts `--data <path>` and `--seed <int>`. The API starts even with
no trained model: `/health` then reports `model_loaded: false` with the reason,
and the prediction endpoints return 503.

## API

Base URL `http://127.0.0.1:8000`. CORS allows `http://localhost:5173` by
default; override with the `CORS_ORIGINS` environment variable
(comma-separated).

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | API status, model availability, model version |
| GET | `/model-info` | Model, dataset, target, features, metrics, risk bands |
| GET | `/demo-examples` | Two real held-out rows for the form's demo button |
| POST | `/predict` | Score one borrower |
| GET | `/docs` | Swagger UI |

### `POST /predict`

Request keys are the dataset's own column names, matching `/model-info`'s
`features` exactly. `MonthlyIncome` and `NumberOfDependents` are optional and
may be `null` — the pipeline's median imputer fills them as it did in training.

```bash
curl -X POST http://127.0.0.1:8000/predict \
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

```json
{
  "default_probability": 0.00815693624730604,
  "predicted_class": 0,
  "predicted_label": "No serious delinquency",
  "classification_threshold": 0.5,
  "risk_band": "Low",
  "risk_band_disclaimer": "Risk bands are illustrative display buckets over the model's predicted probability, not empirically validated lending thresholds.",
  "model_version": "1.0.0",
  "target_definition": "Borrower experienced serious delinquency (90 days past due or worse) within two years of the observation date"
}
```

*(`default_probability` above is from an actual call; it will match on any
machine that trains with seed 42 on the same CSV.)*

**Status codes:** `200` success · `422` validation failure, listing each bad
field (missing, out of range, non-numeric, non-finite, or unknown) · `503` no
model artifact loaded.

**Risk bands** are display buckets — Low `< 0.10`, Medium `< 0.30`, High
otherwise — configured in `backend/app/core/config.py`. They are **not**
validated lending thresholds, and the API says so in every response. The
response separates the three things deliberately: the estimated probability,
the binary class derived from the 0.50 threshold, and the illustrative band.

## Tests

```bash
cd backend
pytest
```

**61 tests, all passing.** They cover:

- the dataset contract — the target can never enter the feature list
- placeholder-code and impossible-age cleaning
- imputation leaving row counts intact
- split determinism, disjointness and preserved class ratios
- the test split never leaking into the refitting set
- metric arithmetic against hand-checked examples
- `predict_proba` column lookup under a reversed `classes_` order
- per-split evaluation and the in-sample ≥ held-out invariant
- the chart data: score histograms bin every record once, and the threshold curve's
  counts agree with its rates and stop before precision degenerates
- artifact loading independent of the training code
- every endpoint, valid and invalid requests, non-finite numbers, unknown
  fields, omitted optional fields, and the probability/class mapping

Tests needing a trained model skip cleanly if `model.joblib` is absent, so the
suite runs on a fresh checkout.

Frontend type checking and production build:

```bash
cd frontend
npm run typecheck
npm run build
```

## Methodology summary

Full detail in [Methodology](documentation/04-methodology.md).

1. **Drop** `Unnamed: 0` only — it is the CSV's row number.
2. **Mark as missing** the 96/98 placeholder codes in the three past-due
   counters (269 rows) and `age < 18` (1 row).
3. **Median-impute** all numeric predictors, fitted on the training split only,
   adding a binary indicator column recording what was missing.
4. **Keep** every record: no rows deleted, no outliers capped.
5. **Split** 60/20/20 stratified, seed 42.
6. **Compare** five Random Forest configurations on validation by PR-AUC.
7. **Select the decision threshold** on validation by maximising F1.
8. **Refit** the winner on train + validation; score the test split once.
9. **Save** the whole pipeline with joblib, plus metadata and metrics.

Preprocessing lives *inside* the saved pipeline, so inference cannot diverge
from training.

**Leakage check:** no predictor correlates with the target above |0.13|, none
is derived from the label, and there are no post-outcome fields. The
delinquency counters describe the two years *before* the observation date; the
target describes the two years *after*.

**Selected model:** 300 trees, `min_samples_leaf=20`, no class weighting, plus
missing-value indicator columns. Five configurations were compared on
validation PR-AUC. Class weighting was tested and rejected — it moved the
model along the precision/recall curve (recall 0.16 → 0.72, precision 0.60 →
0.24) without improving PR-AUC, so an adjustable threshold was preferred to a
baked-in one.

## Limitations

Summarised; full discussion in [Limitations](documentation/09-limitations.md).

- **2011 data** of undocumented provenance; no out-of-time validation is
  possible because there is no date column.
- **Fairness untestable** — the dataset has no protected attributes, so the
  absence of measured bias is not evidence of fairness. `age` *is* used as a
  feature.
- **19.82% of incomes are imputed.** The 96/98 placeholder rows (54.65%
  positive) are imputed too, but a missing-value indicator column preserves
  the fact that the value was absent.
- **Probabilities are not calibrated** and calibration was not evaluated. Treat
  the output as a ranking score, not a real-world frequency.
- **Even at the tuned threshold the model misses about half of all
  delinquencies** (recall 0.4868) and raises 1,374 false alarms. Maximising F1
  weights both errors equally, which lending economics does not — and the
  dataset lacks the cost information needed to do better.
- **No authentication, rate limiting, logging, monitoring or governance.**

## Future improvements

Not implemented — listed to separate plans from completed work.

- Evaluate calibration (`CalibratedClassifierCV`, reliability diagram, Brier
  score) and either calibrate or state the miscalibration quantitatively.
- Compute permutation importance on the test split, which is less biased than
  impurity-based importance.
- Cross-validated threshold selection against an explicit cost matrix.
- A missingness-indicator variant for `MonthlyIncome`.
- Per-prediction explanations (e.g. SHAP) in the result page.
- Frontend component tests (Vitest + Testing Library).

## Documentation

Full reference documentation, with diagrams, lives in
**[`documentation/`](documentation/README.md)**.

| # | Document | Contents |
|---|---|---|
| 01 | [Overview](documentation/01-overview.md) | What the system does, headline results, how the pieces fit |
| 02 | [Setup](documentation/02-setup.md) | Install, get the data, train, run both servers |
| 03 | [Dataset](documentation/03-dataset.md) | Source, licence, schema, data quality, leakage check |
| 04 | [Methodology](documentation/04-methodology.md) | Preprocessing, splits, model selection, threshold |
| 05 | [Model evaluation](documentation/05-model-evaluation.md) | Metrics, curves, cross-validation, feature importance |
| 06 | [Architecture](documentation/06-architecture.md) | Code layout, request lifecycle, design decisions |
| 07 | [API reference](documentation/07-api-reference.md) | Every endpoint, schema and status code |
| 08 | [Deployment](documentation/08-deployment.md) | Vercel hosting, the same-origin proxy, CORS |
| 09 | [Limitations](documentation/09-limitations.md) | Bias, imbalance, calibration, why this is not a credit system |

Working material — the seminar deck, the written report, the original brief and
the scripts that build them — is in **[`dev/`](dev/README.md)**.
