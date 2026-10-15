# Limitations

This project is coursework. It demonstrates an end-to-end machine learning
workflow; it is not a credit-underwriting system and must not be used as one.
The sections below set out why.

## Dataset representativeness and possible bias

**The data is old.** Give Me Some Credit was released for a competition that
ran in late 2011, describing US consumer credit behaviour from around the
financial crisis. Lending standards, credit products, household debt levels and
the macroeconomic backdrop have all changed substantially since. A model fitted
on it describes a population that no longer exists in that form.

**The sampling frame is unknown.** The competition does not document how these
150,000 borrowers were selected, from which lenders, products or regions. There
is no way to establish what population the sample represents, so there is no
basis for generalising from it — to the present day, to another country, or to
any particular lender's applicants.

**Fairness cannot be assessed.** The dataset contains no race, sex, ethnicity,
geography or similar attributes. That absence is not protection:

- `age` *is* present and is used by the model. Age is a protected
  characteristic in credit decisions in many jurisdictions, and this project
  does not evaluate whether the model's use of it would be lawful or
  defensible.
- Features such as income, debt ratio and credit utilisation can act as
  **proxies** for protected characteristics through historical patterns in
  lending and employment. A model can produce disparate impact without ever
  seeing a protected attribute.
- Because the attributes needed for a disparate-impact test are absent, no such
  test was run. **The absence of measured bias here is not evidence of
  fairness** — it is the absence of measurement.

**Historical outcomes encode historical decisions.** The label records who did
become seriously delinquent, which reflects who was granted credit in the first
place and on what terms. Patterns learned from that are patterns of a past
lending regime, including its mistakes.

## Missing and anomalous values

- **`MonthlyIncome` is missing for 19.82% of rows** (29,731). These are filled
  with the training median, with an indicator column marking that they were
  missing. For roughly one row in five the model still has no actual income
  figure — flattening a real signal and adding error that none of the reported
  metrics separates out.
- **`NumberOfDependents` is missing for 2.62%** (3,924), handled the same way.
- **The 96/98 placeholder codes are handled, not solved.** 269 rows carry 96 or
  98 in all three past-due counters, and those borrowers are **54.65%
  positive** against 6.60% for everyone else. The project imputes the value and
  adds a missing-value indicator, so the model knows the figure was absent
  without anyone guessing what the codes mean. That recovers some of the
  signal, but the underlying meaning is still unknown — and 269 rows is too few
  for the model to learn much from the flag alone.
- **`DebtRatio` is not comparable across rows.** Its maximum is 329,664 and
  large values concentrate where income is unrecorded, which suggests the ratio
  was computed against a placeholder denominator. The feature is used as given,
  with its most extreme values effectively meaningless.
- **Extreme values are kept, not capped.** `RevolvingUtilizationOfUnsecuredLines`
  reaches 50,708, which cannot be a genuine utilisation ratio. A Random Forest
  is robust to such values, so they were left alone rather than treated on an
  unverifiable assumption — but they are still wrong values in the input.

## Class imbalance

Only 6.68% of borrowers are positive. Consequences that persist regardless of
modelling choices:

- **Accuracy is close to meaningless.** 0.9199 against 0.9332 for a model that
  always answers "no" — the trained model scores *lower*. Any accuracy figure
  quoted for this task should be read next to that baseline.
- **Even at a tuned threshold the model misses about half of all
  delinquencies**: recall 0.4868, so 1,029 of 2,005 actual cases in the test
  set go unflagged — and it wrongly flags 1,374 borrowers who were fine. The
  conventional 0.50 cut-off is far worse (recall 0.1566), which is why the
  threshold is selected on validation rather than assumed.
- **Precision and recall cannot both be high.** The threshold table in
  [model-evaluation.md](05-model-evaluation.md) shows the whole trade-off:
  reaching 0.70 recall means accepting 0.26 precision — three false alarms for
  every genuine catch.
- **The minority class is small in absolute terms too.** 2,005 positives in the
  test split means metrics on that class carry meaningful sampling noise; small
  differences between configurations should not be over-interpreted.

## Residual overfitting

The model scores **ROC-AUC 0.9329 on data it was fitted on and 0.8654 on the
held-out test split** — a gap of 0.0675. On PR-AUC the gap is larger in
relative terms: 0.5232 in-sample against 0.4014 held out, so a quarter of the
apparent performance does not survive contact with unseen borrowers.

`min_samples_leaf=20` reduced this materially (the depth-unlimited baseline was
worse on validation across the board) and the missing-value indicators narrowed
it a little further, but neither removed it. 300 trees grown on 120,000 rows
retain plenty of capacity to memorise.

Practical consequences:

- **Never quote the in-sample numbers.** They describe memorisation.
- The gap is a reason to expect the test figure itself to be optimistic for
  genuinely *new* borrowers — a test split drawn randomly from the same 2011
  snapshot is still the same population. See distribution shift below.
- Further regularisation (larger leaves, limited `max_depth`, fewer features
  per split) would narrow the gap, and was not pursued — the project caps
  hyperparameter work deliberately.

Five-fold cross-validation (ROC-AUC 0.8632 ± 0.0053) confirms the *held-out*
estimate is stable across partitions, so the 0.8654 is reliable as a measure of
this model on this data. Stability is not the same as absence of overfitting:
the model consistently overfits by about the same amount.

## Distribution shift

The model was fitted and evaluated on random splits of a single 2011 snapshot.
Every metric reported therefore measures performance on data drawn from the
*same* distribution as the training data. That is the easiest possible test and
it says nothing about:

- **Temporal shift.** There is no date column, so no out-of-time validation is
  possible — which is the standard way credit models are validated, precisely
  because random splits flatter them. Performance on current borrowers is
  unknown and should be assumed worse.
- **Population shift.** Applying the model to a different country, lender or
  product class would be extrapolation beyond anything tested here.
- **Form inputs are unconstrained.** The web form accepts combinations of
  values that may be rare or absent in the training data. The model will still
  return a confident-looking probability for them. There is no out-of-
  distribution detection, and no warning when an input is unusual.

## Limitations of the probability estimates

- **The probabilities are not calibrated and calibration was not evaluated.**
  The output is the mean of the per-tree class probabilities, where each
  tree contributes the class proportion of its leaf. Random Forest
  probabilities are typically poorly calibrated, compressed away from 0 and 1.
  A displayed 0.30 does **not** mean 30 of 100 such borrowers become
  delinquent; it means the model ranks this borrower above others. Use it for
  comparison, not as a literal frequency.
- **No uncertainty is reported.** Every prediction is a point estimate. A
  borrower whose attributes resemble thousands of training rows and one whose
  attributes are near-unique are presented identically.
- **The risk bands are presentational.** Low / Medium / High are display
  buckets at 0.10 and 0.30, chosen for legibility in the interface. They are
  configured in `backend/app/core/config.py`, they are not validated lending
  thresholds, and they carry no business or regulatory meaning. The API returns
  this disclaimer with every prediction.
- **The decision threshold optimises F1, which is itself an assumption.** 0.2302
  is selected on the validation split by maximising F1 — a defensible,
  documented criterion, but one that weights a false positive and a false
  negative equally. Real lending does not: a missed delinquency usually costs
  far more than a declined good borrower. Choosing the threshold properly needs
  loss-given-default and margin figures the dataset does not contain.
- **The target is narrow.** A prediction concerns one defined event: serious
  delinquency, 90+ days past due, within two years of the observation date. It
  is not a statement about default generally, about eventual repayment, or
  about creditworthiness.

## Why this is an educational prototype, not a credit system

A system used for real lending decisions would need, at minimum:

| Requirement | Status here |
|---|---|
| Current, representative, documented data | 2011 competition data of unknown provenance |
| Out-of-time validation | Not possible — no date column |
| Fairness and disparate-impact testing | Not possible — no protected attributes |
| Calibrated, monitored probabilities | Not calibrated, not monitored |
| Adverse-action reasons for declines | Not implemented |
| Regulatory compliance (e.g. ECOA, FCRA, GDPR) | Not addressed |
| Model governance, versioning, audit trail | A version string in `metadata.json` |
| Drift monitoring and scheduled retraining | None |
| Human review of automated decisions | None |
| Security, authentication, rate limiting, logging | None — the API is open and unauthenticated |

Beyond the technical gaps, credit decisions materially affect people's lives,
and automated ones are regulated for that reason. Using this project — or
anything built from it without the work above — to approve, decline, price or
otherwise act on a real credit application would be irresponsible and in many
jurisdictions unlawful.

**What it is good for:** learning how a dataset is inspected and documented,
how leakage is checked for, how preprocessing is kept consistent between
training and serving, why accuracy fails under class imbalance, how a
threshold trades precision against recall, and how a trained model is served
through an API to a web frontend.
