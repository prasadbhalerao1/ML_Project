# Dataset research

## Selected dataset

**Give Me Some Credit** — Kaggle competition
<https://www.kaggle.com/competitions/GiveMeSomeCredit/data>

| | |
|---|---|
| Title | Give Me Some Credit |
| Direct URL | <https://www.kaggle.com/competitions/GiveMeSomeCredit/data> |
| Platform | Kaggle (competition, not a Kaggle *dataset* page) |
| Data credited to | Credit Fusion and Kaggle, per the competition's acknowledgements |
| Period | Competition ran September–December 2011; the data is US consumer credit data from that era |
| Files used | `cs-training.csv` (labeled), `Data Dictionary.xls` |
| Files not used | `cs-test.csv` (unlabeled), `sampleEntry.csv` |
| Version | The competition data has no version identifier; the files below were downloaded in October 2026 and verified by checksum of their shape and contents |

Everything in the "Actual inspection" section was produced by running
`python -m ml.explore` against the downloaded `cs-training.csv`, not copied from
the competition page.

## Actual inspection

`cs-training.csv` holds **150,000 rows × 12 columns**: one unnamed row-index
column, the target, and ten predictors. After dropping the row index:

| Column | Type | Role | Notes from the data dictionary and the file |
|---|---|---|---|
| `SeriousDlqin2yrs` | int64 | **target** | Person experienced 90 days past due delinquency or worse |
| `RevolvingUtilizationOfUnsecuredLines` | float64 | feature | Card and personal line balances ÷ total credit limits. Observed range 0 – 50,708 |
| `age` | int64 | feature | Age in years. Observed range 0 – 109 |
| `NumberOfTime30-59DaysPastDueNotWorse` | int64 | feature | Count in the last 2 years. Observed range 0 – 98 |
| `DebtRatio` | float64 | feature | Debt payments, alimony and living costs ÷ gross monthly income. Observed range 0 – 329,664 |
| `MonthlyIncome` | float64 | feature | Gross monthly income. **19.82% missing** |
| `NumberOfOpenCreditLinesAndLoans` | int64 | feature | Open loans and lines of credit. Observed range 0 – 58 |
| `NumberOfTimes90DaysLate` | int64 | feature | Count. Observed range 0 – 98 |
| `NumberRealEstateLoansOrLines` | int64 | feature | Mortgages and home equity lines. Observed range 0 – 54 |
| `NumberOfTime60-89DaysPastDueNotWorse` | int64 | feature | Count in the last 2 years. Observed range 0 – 98 |
| `NumberOfDependents` | float64 | feature | Dependents excluding the borrower. **2.62% missing** |

### Target definition and class meanings

`SeriousDlqin2yrs` is binary:

- **1** — the borrower experienced serious delinquency, 90 days past due or
  worse, within two years of the observation date. 10,026 rows, **6.68%**.
- **0** — no such delinquency was recorded in that window. 139,974 rows, 93.32%.

The target is a *specific historical outcome in a two-year window*. It is not
"loan default" in general, and it does not state whether a borrower eventually
repaid. The application's wording follows this definition everywhere.

### Data-quality findings

| Finding | Count | Decision |
|---|---|---|
| `MonthlyIncome` missing | 29,731 (19.82%) | Median-impute inside the pipeline |
| `NumberOfDependents` missing | 3,924 (2.62%) | Median-impute inside the pipeline |
| The three late-payment counters equal 96 or 98 | 269 rows, in all three columns at once | Treated as placeholder codes → missing → imputed |
| `age` = 0 | 1 row | Impossible for a credit applicant → missing → imputed |
| Rows identical on all predictors | 646 | **Kept** (see methodology) |
| Rows identical including the label | 609 | **Kept** |
| `RevolvingUtilizationOfUnsecuredLines` > 1 | 3,321 | **Kept** — over-limit balances are genuine |
| `DebtRatio` > 10 | 28,877 | **Kept** — concentrated in rows with no recorded income |

### Leakage check

Pearson correlation of each predictor with the target, strongest first:

```
NumberOfTime30-59DaysPastDueNotWorse     +0.1256
NumberOfTimes90DaysLate                  +0.1172
age                                      -0.1154
NumberOfTime60-89DaysPastDueNotWorse     +0.1023
NumberOfDependents                       +0.0460
NumberOfOpenCreditLinesAndLoans          -0.0297
MonthlyIncome                            -0.0197
DebtRatio                                -0.0076
NumberRealEstateLoansOrLines             -0.0070
RevolvingUtilizationOfUnsecuredLines     -0.0018
```

No predictor is near |1.0| and none is computed from the label. The delinquency
counters describe behaviour *in the two years before* the observation date,
while the target describes the two years *after* it, so they are legitimate
history rather than leakage. There is no post-outcome field such as a recovery
amount or a charge-off flag.

## Alternatives evaluated

| Dataset | Rows / shape | Target | Leakage risk | Verdict |
|---|---|---|---|---|
| **Give Me Some Credit** <br> <https://www.kaggle.com/competitions/GiveMeSomeCredit/data> | 150,000 × 11, one flat file | `SeriousDlqin2yrs`, clearly documented | **Low** — verified above | **Selected** |
| **Loan Default Prediction** <br> <https://www.kaggle.com/competitions/loan-default-pred/data> | 27,003 × 47, one flat file | `loan_status` (0/1), 14.7% positive | **High** — see below | Rejected |
| **Home Credit Default Risk** <br> <https://www.kaggle.com/competitions/home-credit-default-risk/data> | ~308k main rows plus 7 related tables, ~2.7 GB | `TARGET` | Low, but requires careful aggregation | Rejected as out of scope |

### Why Loan Default Prediction was rejected

This file was downloaded and inspected (`loan_train.csv`, LendingClub-style
columns). It carries several fields that are only known *after* the loan's
outcome is decided:

`recoveries`, `collection_recovery_fee`, `out_prncp`, `out_prncp_inv`,
`total_pymnt`, `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee`,
`last_pymnt_d`, `last_pymnt_amnt`

A model trained on these would score superbly and learn nothing: a non-zero
`recoveries` value essentially announces that the loan defaulted. Using it
responsibly means dropping roughly a quarter of the columns first, which makes
it a worse teaching dataset for a first Random Forest. It also has heavy
missingness in `mths_since_last_record` (92.8%) and `mths_since_last_delinq`
(64.4%), plus free-text columns (`desc`, `emp_title`, `title`, `url`) that need
NLP-ish handling outside this project's scope.

### Why Home Credit Default Risk was rejected

Not downloaded. It is the richest of the three, but it spreads the signal
across eight tables (`application_train`, `bureau`, `bureau_balance`,
`previous_application`, `POS_CASH_balance`, `installments_payments`,
`credit_card_balance`) that must be joined and aggregated before modelling.
That feature-engineering work would dominate the project and obscure the
Random Forest itself, which is the thing being learned here.

## Why Give Me Some Credit was selected

1. **One flat labeled file.** No joins, no aggregation — the preprocessing
   stays short enough to read and verify.
2. **An unambiguous, documented target** with an official data dictionary
   shipped alongside the data.
3. **No post-outcome columns**, verified by inspecting all eleven fields.
4. **Realistic but tractable difficulty**: 6.68% positives makes class
   imbalance a real lesson rather than an afterthought, and 19.8% missing
   income makes imputation necessary rather than decorative.
5. **Size suits a laptop.** 150,000 × 11 trains in under a minute.
6. **Ten numeric features** map cleanly onto a web form a non-specialist can
   fill in.

## Download and setup

The competition data requires a Kaggle account and acceptance of the
competition rules. Either route works:

### Option A — browser

1. Sign in at <https://www.kaggle.com>.
2. Open <https://www.kaggle.com/competitions/GiveMeSomeCredit/data>.
3. Accept the competition rules if prompted (the **Rules** tab, "I Understand
   and Accept"). Downloads are blocked until this is done.
4. Download `cs-training.csv` and, optionally, `Data Dictionary.xls`.
5. Place them at `data/raw/GiveMeSomeCredit/`.

### Option B — Kaggle API

```bash
pip install kaggle
# Kaggle → your profile → Settings → API → "Create New Token"
# Save the downloaded kaggle.json to:
#   Windows: C:\Users\<you>\.kaggle\kaggle.json
#   macOS/Linux: ~/.kaggle/kaggle.json   (then: chmod 600 ~/.kaggle/kaggle.json)

kaggle competitions download -c GiveMeSomeCredit -p data/raw/GiveMeSomeCredit
cd data/raw/GiveMeSomeCredit && unzip GiveMeSomeCredit.zip
```

The API call fails with a 403 until the rules have been accepted in the browser
at least once.

Expected layout:

```
data/raw/GiveMeSomeCredit/
├── cs-training.csv        # 150,000 labeled rows — the only file used for modelling
├── cs-test.csv            # 101,503 rows, target column entirely blank — not used
├── Data Dictionary.xls
└── sampleEntry.csv
```

## Licence, attribution and competition rules

- The data is distributed under Kaggle's **competition rules** for Give Me Some
  Credit, which are accepted on the competition's Rules tab. They are not an
  open-data licence such as CC-BY, and they are not reproduced here: read them
  on the competition page before using the data for anything beyond private
  study.
- Kaggle's competition rules typically restrict data use to the purposes of the
  competition. This project treats the data as **coursework material only** and
  does not redistribute it: the raw CSVs are git-ignored and must be downloaded
  by each person from Kaggle directly.
- Attribution used throughout this project: *Give Me Some Credit*, Kaggle
  competition, data credited to Credit Fusion and Kaggle,
  <https://www.kaggle.com/competitions/GiveMeSomeCredit>.
- **Not independently verified by this project:** the exact text of the 2011
  competition rules and the precise legal status of the data. The Kaggle pages
  are rendered client-side and could not be retrieved programmatically here.
  Read the Rules tab yourself before relying on any statement above.

## Known limitations, biases and data-quality issues

- **Age of the data.** It describes US consumer credit around 2011, in the
  aftermath of the financial crisis. Borrower behaviour, lending standards and
  the macroeconomy have all moved since. See [limitations.md](09-limitations.md).
- **No geography, product or lender information.** There is no way to check
  whether the sample represents any particular population, or to test for
  disparate impact across groups — the protected attributes simply are not
  present, which also means their absence cannot be verified as harmless.
- **Placeholder codes are undocumented, and they matter.** The 96/98 values in
  the three delinquency counters are not explained in the data dictionary. The
  269 rows carrying them are **54.65% positive**, versus 6.60% elsewhere.
  Treating them as missing is an inference, and a costly one; see
  [methodology.md](04-methodology.md) for the reasoning and the trade-off.
- **`DebtRatio` is unreliable where income is missing.** Its enormous upper tail
  (max 329,664) is concentrated in rows with no recorded income, which suggests
  the ratio was computed with a placeholder or zero denominator.
- **Class imbalance.** 6.68% positives means accuracy is nearly useless as a
  headline metric; see [model-evaluation.md](05-model-evaluation.md).
- **Unlabeled competition test file.** `cs-test.csv` has an empty target column.
  No metric in this project is computed on it; all reported numbers come from a
  split of the labeled training file.
