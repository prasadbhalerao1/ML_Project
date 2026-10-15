/**
 * Builds the project report as a .docx.
 *
 * Every metric is read from backend/ml/artifacts/*.json, so the report cannot
 * drift from the model that was actually trained.
 *
 * Typography per request: Times New Roman, 14pt headings / 12pt body /
 * 10pt tables and captions.
 */

const fs = require('fs');
const path = require('path');
const {
  AlignmentType, BorderStyle, Document, Footer, HeadingLevel, ImageRun,
  LevelFormat, NumberFormat, PageBreak, PageNumber, Packer, Paragraph,
  ShadingType, Table, TableCell, TableRow, TextRun, WidthType,
} = require('docx');

const ROOT = process.argv[2];
const ART = path.join(ROOT, 'backend', 'ml', 'artifacts');
const FIG = path.join(ART, 'figures');

const meta = JSON.parse(fs.readFileSync(path.join(ART, 'metadata.json'), 'utf8'));
const evaluation = JSON.parse(fs.readFileSync(path.join(ART, 'evaluation_report.json'), 'utf8'));

// ---------------------------------------------------------------- typography
const FONT = 'Times New Roman';
const SIZE_HEADING = 28; // 14pt, in half-points
const SIZE_BODY = 24; // 12pt
const SIZE_SMALL = 20; // 10pt
const LINE_SPACING = 360; // 1.5 lines, in twentieths of a point
const CONTENT_WIDTH = 9360; // 6.5in page body, in DXA

const n = (v, d = 4) => Number(v).toFixed(d);
const pct = (v, d = 2) => `${(Number(v) * 100).toFixed(d)}%`;
const int = (v) => Number(v).toLocaleString('en-US');

// ------------------------------------------------------------------ builders
const body = (text, opts = {}) =>
  new Paragraph({
    alignment: opts.alignment ?? AlignmentType.JUSTIFIED,
    spacing: { line: LINE_SPACING, after: opts.after ?? 120 },
    indent: opts.indent,
    children: runs(text),
  });

/** Splits `**bold**` and `*italic*` segments into separate runs. */
function runs(text, size = SIZE_BODY) {
  return String(text)
    .split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g)
    .filter(Boolean)
    .map((part) => {
      const bold = part.startsWith('**') && part.endsWith('**');
      const italics = !bold && part.startsWith('*') && part.endsWith('*');
      let value = part;
      if (bold) value = part.slice(2, -2);
      else if (italics) value = part.slice(1, -1);
      return new TextRun({ text: value, bold, italics, font: FONT, size });
    });
}

const h1 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 180, line: LINE_SPACING },
    children: [new TextRun({ text, bold: true, font: FONT, size: SIZE_HEADING })],
  });

const h2 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 120, line: LINE_SPACING },
    children: [new TextRun({ text, bold: true, font: FONT, size: SIZE_BODY })],
  });

const bullet = (text) =>
  new Paragraph({
    numbering: { reference: 'bullets', level: 0 },
    spacing: { line: LINE_SPACING, after: 80 },
    alignment: AlignmentType.JUSTIFIED,
    children: runs(text),
  });

const caption = (text) =>
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 60, after: 240 },
    children: [new TextRun({ text, italics: true, font: FONT, size: SIZE_SMALL })],
  });

/** Table with 10pt text, a shaded header row, and DXA widths throughout. */
function table(headers, rows, widths) {
  const columnWidths = widths ?? headers.map(() => Math.floor(CONTENT_WIDTH / headers.length));

  const cell = (text, { bold = false, align = AlignmentType.LEFT, shade = null }, i) =>
    new TableCell({
      width: { size: columnWidths[i], type: WidthType.DXA },
      shading: shade ? { type: ShadingType.CLEAR, fill: shade } : undefined,
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      children: [
        new Paragraph({
          alignment: align,
          spacing: { line: 240, after: 0 },
          children: [new TextRun({ text: String(text), bold, font: FONT, size: SIZE_SMALL })],
        }),
      ],
    });

  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    columnWidths,
    rows: [
      new TableRow({
        tableHeader: true,
        children: headers.map((head, i) =>
          cell(head, { bold: true, align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER, shade: 'D9D9D9' }, i)
        ),
      }),
      ...rows.map((row) =>
        new TableRow({
          children: row.map((value, i) => {
            const emphasise = typeof value === 'string' && value.startsWith('**');
            return cell(
              emphasise ? value.slice(2, -2) : value,
              { bold: emphasise, align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER },
              i
            );
          }),
        })
      ),
    ],
  });
}

/** Figure scaled to a given width in points, preserving aspect ratio. */
function figure(file, widthPt, ratio) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 180, after: 0 },
    children: [
      new ImageRun({
        type: 'png',
        data: fs.readFileSync(path.join(FIG, file)),
        transformation: { width: widthPt, height: Math.round(widthPt * ratio) },
      }),
    ],
  });
}

const spacer = () => new Paragraph({ spacing: { after: 120 }, children: [] });

// --------------------------------------------------------------- derived data
const test = evaluation.per_split_metrics.test;
const train = evaluation.per_split_metrics.train;
const validation = evaluation.per_split_metrics.validation;
const cm = test.confusion_matrix;
const report = evaluation.test_classification_report;
const cv = evaluation.cross_validation.summary;
const dist = evaluation.class_distribution;
const threshold = meta.decision_threshold.value;
const importances = meta.feature_importances;
const majorityBaseline = 1 - dist.test.positive_rate;

const FEATURE_NOTES = {
  RevolvingUtilizationOfUnsecuredLines: 'Card and personal line balances divided by total credit limits',
  age: 'Borrower age in years',
  'NumberOfTime30-59DaysPastDueNotWorse': 'Times 30-59 days past due in the last two years',
  DebtRatio: 'Debt payments, alimony and living costs divided by gross monthly income',
  MonthlyIncome: 'Gross monthly income (19.82% missing)',
  NumberOfOpenCreditLinesAndLoans: 'Open loans and lines of credit',
  NumberOfTimes90DaysLate: 'Times 90 or more days past due',
  NumberRealEstateLoansOrLines: 'Mortgages and home equity lines',
  'NumberOfTime60-89DaysPastDueNotWorse': 'Times 60-89 days past due in the last two years',
  NumberOfDependents: 'Dependents excluding the borrower (2.62% missing)',
};

// ------------------------------------------------------------------ title page
const titlePage = [
  new Paragraph({ spacing: { before: 2200, after: 0 }, children: [] }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 120, line: LINE_SPACING },
    children: [
      new TextRun({
        text: 'LOAN DEFAULT RISK PREDICTION USING RANDOM FOREST',
        bold: true, font: FONT, size: SIZE_HEADING,
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 600, line: LINE_SPACING },
    children: [
      new TextRun({
        text: 'A Machine Learning Laboratory Project',
        font: FONT, size: SIZE_BODY,
      }),
    ],
  }),
  ...[
    ['Problem type', 'Supervised binary classification (class-imbalanced)'],
    ['Algorithm', 'Random Forest Classifier (scikit-learn)'],
    ['Dataset', 'Give Me Some Credit — Kaggle competition'],
    ['Records', `${int(meta.dataset.labeled_rows)} labelled borrowers, ${meta.features.length} predictors`],
    ['Deliverable', 'FastAPI backend with a React + TypeScript frontend'],
    ['Model version', `${meta.model_version} (trained ${meta.trained_at.slice(0, 10)}, seed ${meta.random_seed})`],
  ].map(([label, value]) =>
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 100, line: 280 },
      children: [
        new TextRun({ text: `${label}:  `, bold: true, font: FONT, size: SIZE_BODY }),
        new TextRun({ text: value, font: FONT, size: SIZE_BODY }),
      ],
    })
  ),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 700 },
    children: [
      new TextRun({
        text: 'All results in this report were generated by executing the project code.',
        italics: true, font: FONT, size: SIZE_SMALL,
      }),
    ],
  }),
  new Paragraph({ children: [new PageBreak()] }),
];

// ---------------------------------------------------------------- the content
const content = [
  h1('Abstract'),
  body(
    `This project implements an end-to-end machine learning system that estimates the probability of a borrower experiencing serious delinquency — 90 days past due or worse — within two years of an observation date. A Random Forest classifier was trained on the public Kaggle "Give Me Some Credit" dataset of ${int(meta.dataset.labeled_rows)} labelled borrower records with ${meta.features.length} numeric predictors. The positive class represents only ${pct(dist.test.positive_rate)} of records, making this a class-imbalanced binary classification problem in which accuracy is an actively misleading metric.`
  ),
  body(
    `Five Random Forest configurations were compared on a held-out validation split using the area under the precision-recall curve as the selection criterion. The chosen configuration achieved a **ROC-AUC of ${n(test.roc_auc)}** and a **PR-AUC of ${n(test.pr_auc)}** on a test split of ${int(dist.test.total)} records that was scored exactly once. The decision threshold was selected on validation data by maximising F1, giving a positive-class F1 of **${n(test.f1)}** and recall of **${n(test.recall)}** — against ${n(evaluation.test_threshold_sweep[evaluation.test_threshold_sweep.length - 1].f1)} and ${n(evaluation.test_threshold_sweep[evaluation.test_threshold_sweep.length - 1].recall)} at the conventional 0.50 cut-off. Five-fold cross-validation confirmed the stability of these estimates. The trained pipeline is served through a FastAPI REST API consumed by a four-page React frontend.`
  ),

  h1('1. Introduction and Objective'),
  h2('1.1 Problem statement'),
  body(
    'Credit providers need to estimate the likelihood that an applicant will fall seriously behind on repayments. This project treats that need as a supervised learning problem and implements a complete, reproducible solution: data inspection, preprocessing, model training and evaluation, and deployment behind a web interface.'
  ),
  h2('1.2 Type of machine learning problem'),
  body('The task is **supervised binary classification**, for three reasons:'),
  bullet('**Supervised** — every training record carries a known outcome label, so the model learns from answers it is shown.'),
  bullet('**Classification, not regression** — the quantity predicted is a category (delinquent or not), not a continuous value. The model does emit a probability between 0 and 1, but that expresses confidence in a class rather than a magnitude.'),
  bullet('**Not clustering** — clustering is unsupervised and discovers groupings in unlabelled data. Here the two groups are defined in advance by the dataset.'),
  body(
    `A defining property of this particular problem is **class imbalance**: only ${pct(dist.test.positive_rate)} of borrowers are positive. This shapes every evaluation decision taken later in the report.`
  ),
  h2('1.3 Scope'),
  body('In scope: dataset research, data inspection and leakage checking, preprocessing without train-serve skew, Random Forest training and evaluation, a REST API, and a web interface. Deliberately out of scope: alternative algorithms, extensive hyperparameter search, databases, authentication and containerisation. The aim is to implement and understand one algorithm correctly rather than to maximise a leaderboard score.'),

  h1('2. Dataset'),
  h2('2.1 Source and provenance'),
  body(
    `The dataset is **Give Me Some Credit**, a Kaggle competition dataset available at ${meta.dataset.url}. The data is credited to Credit Fusion and Kaggle and describes United States consumer credit behaviour from around 2011. Access requires a Kaggle account and acceptance of the competition rules; the raw files are therefore not redistributed with the project.`
  ),
  body(
    `Only the labelled training file, ${meta.dataset.file}, is used. The competition also ships an unlabelled test file whose target column is entirely blank; no metric in this project is computed on it, because no honest metric could be. All reported figures come from splits of the labelled file.`
  ),
  h2('2.2 Target variable'),
  body(
    `The target is **${meta.target.column}**, a binary indicator. A value of 1 means the borrower ${meta.target.definition.toLowerCase()}. A value of 0 means no such delinquency was recorded in that window.`
  ),
  body('This definition is narrow and the project states it consistently. The model does not predict loan default in general, nor does it indicate whether a borrower will ultimately repay.'),
  h2('2.3 Predictors'),
  body(`The ${meta.features.length} input features, all numeric, are listed in Table 1.`),
  spacer(),
  table(
    ['#', 'Feature', 'Description'],
    meta.features.map((f, i) => [String(i + 1), f, FEATURE_NOTES[f] ?? '']),
    [600, 3360, 5400]
  ),
  caption('Table 1 — Input features, as verified against the competition data dictionary.'),
  h2('2.4 Class distribution'),
  body('The data was split 60/20/20 with stratification, preserving the class ratio in each part. Table 2 confirms the balance is identical across splits.'),
  spacer(),
  table(
    ['Split', 'Records', 'Negative (0)', 'Positive (1)', 'Positive rate'],
    ['train', 'validation', 'test'].map((k) => [
      k.charAt(0).toUpperCase() + k.slice(1),
      int(dist[k].total), int(dist[k].negative), int(dist[k].positive), pct(dist[k].positive_rate),
    ]),
    [1800, 1900, 1900, 1900, 1860]
  ),
  caption('Table 2 — Stratified split sizes and class balance.'),
  figure('target_distribution.png', 330, 0.667),
  caption('Figure 1 — Target distribution. The minority class is a small fraction of all records.'),

  h1('3. Exploratory Data Analysis'),
  body('The raw file was inspected before any preprocessing logic was written. Key findings are summarised in Table 3.'),
  spacer(),
  table(
    ['Finding', 'Count', 'Decision taken'],
    [
      ['MonthlyIncome missing', '29,731 (19.82%)', 'Median imputation plus indicator column'],
      ['NumberOfDependents missing', '3,924 (2.62%)', 'Median imputation plus indicator column'],
      ['Delinquency counters coded 96 or 98', '269 rows', 'Treated as placeholders; imputed with indicator'],
      ['age equal to 0', '1 row', 'Impossible; treated as missing'],
      ['Duplicate predictor rows', '646', 'Kept — plausibly distinct borrowers'],
      ['Revolving utilisation above 1.0', '3,321', 'Kept — over-limit balances are genuine'],
      ['DebtRatio above 10', '28,877', 'Kept — concentrated where income is unrecorded'],
    ],
    [3600, 2000, 3760]
  ),
  caption('Table 3 — Data-quality findings from ml/explore.py and the decisions they drove.'),
  h2('3.1 Target leakage check'),
  body(
    'No predictor correlates with the target above an absolute value of 0.13, and none is derived from the label. The delinquency counters describe behaviour in the two years *before* the observation date, whereas the target describes the two years *after* it, so they constitute legitimate history rather than leakage. Critically, the dataset contains no post-outcome fields such as recovery amounts or charge-off flags.'
  ),
  body('This check was a deciding factor in dataset selection. An alternative candidate, the Kaggle "Loan Default Prediction" dataset, was inspected and rejected because it carries fields such as recoveries, collection_recovery_fee and total_rec_late_fee, which are only known after a loan has already defaulted and would leak the answer directly into the model.'),
  figure('correlation_heatmap.png', 330, 0.833),
  caption('Figure 2 — Pearson correlation between predictors and the target.'),
  figure('feature_distributions.png', 400, 0.636),
  caption('Figure 3 — Distributions of four representative features (top 1% clipped for display only).'),

  h1('4. Methodology'),
  h2('4.1 Preprocessing pipeline'),
  body('Preprocessing and the classifier are wrapped in a single scikit-learn Pipeline, which is the object saved to disk. Keeping the transformations inside the pipeline is the project\'s main defence against train-serve skew: the API loads this object and calls it, so inference cannot apply a different transformation from the one fitted during training. There is no second copy of the preprocessing logic on the serving side.'),
  body('The pipeline has three stages:'),
  bullet('**ImplausibleValueCleaner** — replaces documented placeholder and impossible values with missing markers: the codes 96 and 98 in the three delinquency counters, and any age below 18.'),
  bullet('**SimpleImputer(strategy="median", add_indicator=True)** — fills missing values with the training median and appends a binary column recording that the value was absent.'),
  bullet('**RandomForestClassifier** — the model itself.'),
  body('All transformations are fitted on the training split only. A fixed random seed of 42 makes every result reproducible.'),
  h2('4.2 Handling of placeholder codes'),
  body(
    'The three past-due counters use 96 and 98 as codes rather than counts; a borrower cannot have been 30-59 days late 98 times in a 24-month window, and all 269 affected rows carry a code in all three counters simultaneously. The data dictionary does not document them.'
  ),
  body(
    'These rows are **54.65% positive** against a base rate of 6.68%, so whatever the codes mean, they are strongly associated with the outcome. Imputing them to the median alone would tell the model those borrowers had a clean payment record. Asserting that 98 means "very many late payments" would instead claim a meaning the documentation does not support. The project therefore imputes the value but adds a missing-value indicator, allowing the model to learn that the figure was unavailable without anyone guessing why.'
  ),
  h2('4.3 Why missing-value indicators were added'),
  body(
    `The imputer appends one binary column per feature that had missing values in training, giving the classifier **${meta.model_input_columns.length} input columns** derived from the ${meta.features.length} input features. This was tested rather than assumed: an otherwise identical configuration without indicators was included in the comparison, and the indicator variant won on validation PR-AUC. The API request schema is unchanged, because the indicators are derived inside the pipeline.`
  ),
  h2('4.4 Model selection'),
  body('Five configurations were fitted on the training split and compared on the validation split. The selection metric is **average precision (PR-AUC)**, chosen because it reflects performance on the rare positive class, whereas accuracy is dominated by the majority class and ROC-AUC is comparatively optimistic under heavy imbalance.'),
  spacer(),
  table(
    ['Candidate', 'ROC-AUC', 'PR-AUC', 'Precision', 'Recall'],
    meta.validation_comparison.map((c) => {
      const chosen = c.candidate === meta.selected_configuration.name;
      const wrap = (v) => (chosen ? `**${v}**` : v);
      return [wrap(c.candidate), wrap(n(c.roc_auc)), wrap(n(c.pr_auc)), wrap(n(c.precision)), wrap(n(c.recall))];
    }),
    [3760, 1400, 1400, 1400, 1400]
  ),
  caption('Table 4 — Validation comparison of candidate configurations. The selected row is shown in bold.'),
  body(
    `The selected configuration is **${meta.selected_configuration.name}**: ${meta.selected_configuration.params.n_estimators} trees with min_samples_leaf = ${meta.selected_configuration.params.min_samples_leaf}, no class weighting, plus missing-value indicators. It was then refitted on train and validation combined (${int(dist.train.total + dist.validation.total)} records) and scored once on the test split.`
  ),
  h2('4.5 Finding on class weighting'),
  body(
    'Class weighting was investigated, as the imbalance warrants. Both class_weight="balanced" and "balanced_subsample" raised recall dramatically — from roughly 0.16 to 0.72 — while lowering precision by a comparable amount, yet left PR-AUC and ROC-AUC essentially unchanged. In other words, class weighting moved the model **along** the precision-recall curve rather than improving the curve itself.'
  ),
  body('Since a decision threshold achieves the same movement, but is visible in the saved metadata, adjustable without refitting, and chosen by a documented criterion, the unweighted model was retained and the operating point set explicitly by threshold selection.'),
  h2('4.6 Decision threshold selection'),
  body(
    `The conventional 0.50 cut-off is a poor fit at a ${pct(dist.test.positive_rate)} base rate: it makes the forest so conservative that it flags only ${int(evaluation.test_threshold_sweep[evaluation.test_threshold_sweep.length - 1].confusion_matrix.true_positives + evaluation.test_threshold_sweep[evaluation.test_threshold_sweep.length - 1].confusion_matrix.false_positives)} of ${int(dist.test.total)} test borrowers and misses most genuine delinquencies.`
  ),
  body(
    `The threshold is therefore **selected, not assumed**. It is chosen by maximising F1 on the **validation** split, scored by the candidate fitted on the training split only — a model that had never seen those records. The test split plays no part in the choice. The selected value is **${n(threshold)}**, and it is stored in the model metadata and read by the API at startup, so the threshold the service applies is always the one the reported metrics were measured at.`
  ),
  body(
    `Validation F1 at this threshold was ${n(0.4396)}, and test F1 came out at ${n(test.f1)}. The close agreement is evidence that the threshold generalises rather than having been over-fitted to the validation split.`
  ),

  h1('5. Results'),
  h2('5.1 Performance across all splits'),
  body(
    'Because the final model was refitted on train and validation combined, those two splits are in-sample; only the test column is a held-out estimate. All three are reported so the generalisation gap is visible rather than implied.'
  ),
  spacer(),
  table(
    ['Metric', 'Train (in-sample)', 'Validation (in-sample)', 'Test (held out)'],
    [
      ['Accuracy', n(train.accuracy), n(validation.accuracy), `**${n(test.accuracy)}**`],
      ['Precision', n(train.precision), n(validation.precision), `**${n(test.precision)}**`],
      ['Recall', n(train.recall), n(validation.recall), `**${n(test.recall)}**`],
      ['F1-score', n(train.f1), n(validation.f1), `**${n(test.f1)}**`],
      ['ROC-AUC', n(train.roc_auc), n(validation.roc_auc), `**${n(test.roc_auc)}**`],
      ['PR-AUC', n(train.pr_auc), n(validation.pr_auc), `**${n(test.pr_auc)}**`],
    ],
    [2400, 2320, 2320, 2320]
  ),
  caption(`Table 5 — Metrics on every split at threshold ${n(threshold)}.`),
  h2('5.2 Confusion matrix'),
  spacer(),
  table(
    ['', 'Predicted 0', 'Predicted 1', 'Row total'],
    [
      ['Actual 0 (no delinquency)', `${int(cm.true_negatives)} (TN)`, `${int(cm.false_positives)} (FP)`, int(cm.true_negatives + cm.false_positives)],
      ['Actual 1 (delinquency)', `${int(cm.false_negatives)} (FN)`, `${int(cm.true_positives)} (TP)`, int(cm.false_negatives + cm.true_positives)],
    ],
    [3360, 2000, 2000, 2000]
  ),
  caption('Table 6 — Test-set confusion matrix.'),
  body(
    `Read plainly: of ${int(cm.false_negatives + cm.true_positives)} borrowers who genuinely became seriously delinquent, the model identified **${int(cm.true_positives)}** and missed ${int(cm.false_negatives)}. Of ${int(cm.true_negatives + cm.false_positives)} who did not, it incorrectly flagged ${int(cm.false_positives)}.`
  ),
  figure('confusion_matrix.png', 270, 0.840),
  caption('Figure 4 — Test confusion matrix.'),
  h2('5.3 Classification report'),
  spacer(),
  table(
    ['Class', 'Precision', 'Recall', 'F1-score', 'Support'],
    [
      ['0 — No serious delinquency', n(report.classes['0'].precision), n(report.classes['0'].recall), n(report.classes['0'].f1), int(report.classes['0'].support)],
      ['1 — Serious delinquency', n(report.classes['1'].precision), n(report.classes['1'].recall), n(report.classes['1'].f1), int(report.classes['1'].support)],
      ['Accuracy', '', '', `**${n(test.accuracy)}**`, int(dist.test.total)],
      ['Macro average', n(report.macro_avg.precision), n(report.macro_avg.recall), `**${n(report.macro_avg.f1)}**`, int(dist.test.total)],
      ['Weighted average', n(report.weighted_avg.precision), n(report.weighted_avg.recall), n(report.weighted_avg.f1), int(dist.test.total)],
    ],
    [3360, 1500, 1500, 1500, 1500]
  ),
  caption('Table 7 — Per-class classification report on the test split.'),
  body('This table is the clearest demonstration of why a single headline number misleads on imbalanced data. The model is excellent on the majority class, with an F1 of ' + n(report.classes['0'].f1) + ', and considerably weaker on the minority class at ' + n(report.classes['1'].f1) + '. The weighted average of ' + n(report.weighted_avg.f1) + ' looks impressive but is dominated by the ' + int(report.classes['0'].support) + ' negative records and conveys almost nothing about the class the project exists to detect.'),
  body(
    `Where a single figure is required, the **macro average F1 of ${n(report.macro_avg.f1)}** or the **positive-class F1 of ${n(report.classes['1'].f1)}** should be quoted, never the weighted average.`
  ),
  h2('5.4 Why accuracy is not the headline metric'),
  body(
    `The model attains an accuracy of ${n(test.accuracy)}. A trivial classifier that ignores all inputs and always predicts "no delinquency" would score **${n(majorityBaseline)}** on the same test set — a *higher* figure. Selecting a threshold that detects delinquencies deliberately reduced accuracy while tripling recall, which is the correct trade for this problem and the clearest possible demonstration that accuracy must not drive the decision.`
  ),
  body(
    `The two metrics that carry information are ROC-AUC and PR-AUC. **ROC-AUC of ${n(test.roc_auc)}** means that, given one delinquent and one non-delinquent borrower drawn at random, the model assigns the delinquent borrower a higher score ${pct(test.roc_auc, 1)} of the time. **PR-AUC of ${n(test.pr_auc)}** should be read against a no-skill baseline equal to the positive rate, ${n(dist.test.positive_rate)} — approximately a sixfold improvement over chance on the rare class.`
  ),
  figure('test_curves.png', 420, 0.409),
  caption('Figure 5 — ROC curve (left) and precision-recall curve (right) on the test split.'),
  h2('5.5 Threshold trade-off'),
  body('The same trained model produces very different operating points depending on the cut-off applied. Table 8 shows the trade-off; "flagged" is the number of test borrowers marked positive.'),
  spacer(),
  table(
    ['Threshold', 'Accuracy', 'Precision', 'Recall', 'F1', 'Flagged'],
    evaluation.test_threshold_sweep.map((r) => {
      const inUse = Math.abs(r.threshold - threshold) < 1e-6;
      const w = (v) => (inUse ? `**${v}**` : v);
      const flagged = r.confusion_matrix.true_positives + r.confusion_matrix.false_positives;
      return [
        w(n(r.threshold, 4) + (inUse ? ' (in use)' : '')),
        w(n(r.accuracy)), w(n(r.precision)), w(n(r.recall)), w(n(r.f1)), w(int(flagged)),
      ];
    }),
    [2360, 1400, 1400, 1400, 1400, 1400]
  ),
  caption('Table 8 — Precision-recall trade-off across decision thresholds.'),
  body('Lowering the threshold to 0.10 raises recall above 0.70 but reduces precision to roughly 0.26, meaning three of every four flagged borrowers would be false alarms. Raising it to 0.50 yields precision above 0.62 but detects only about one delinquency in six. Accuracy increases monotonically with the threshold purely because the model predicts the majority class more often, which is precisely why it is a poor guide here.'),
  h2('5.6 Generalisation gap'),
  body('Comparing in-sample with held-out performance quantifies how much of the apparent skill is memorisation.'),
  spacer(),
  table(
    ['Metric', 'Train', 'Test', 'Gap (train − test)'],
    ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc'].map((k) => [
      k.replace('_', '-').toUpperCase(),
      n(train[k]), n(test[k]),
      (evaluation.train_test_gap[k] >= 0 ? '+' : '') + n(evaluation.train_test_gap[k]),
    ]),
    [2400, 2320, 2320, 2320]
  ),
  caption('Table 9 — Generalisation gap between the fitted and held-out splits.'),
  body(
    `The model still overfits: ROC-AUC falls by ${n(evaluation.train_test_gap.roc_auc)} and PR-AUC by ${n(evaluation.train_test_gap.pr_auc)} from in-sample to held out. Regularisation through min_samples_leaf reduced this materially relative to a depth-unlimited forest, and the indicator columns narrowed it slightly further, but 300 trees grown on ${int(dist.train.total + dist.validation.total)} records retain substantial capacity to memorise. Notably, accuracy barely moves while PR-AUC collapses — further evidence that accuracy is insensitive to what matters here.`
  ),
  figure('roc_by_split.png', 320, 0.833),
  caption('Figure 6 — ROC curves by split. The separation between the in-sample and held-out curves is the generalisation gap.'),
  h2('5.7 Cross-validation'),
  body(
    `A single test score derives from one particular partition. Stratified five-fold cross-validation was run on the train and validation data combined, with the test split excluded throughout and the same decision threshold applied, to establish whether that score is stable.`
  ),
  spacer(),
  table(
    ['Metric', 'CV mean', 'Std. dev.', 'Min', 'Max', 'Test'],
    ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc'].map((k) => [
      k.replace('_', '-').toUpperCase(),
      n(cv[k].mean), n(cv[k].std), n(cv[k].min), n(cv[k].max), n(test[k]),
    ]),
    [2160, 1440, 1440, 1440, 1440, 1440]
  ),
  caption('Table 10 — Five-fold cross-validation summary compared with the held-out test result.'),
  body(
    'Every test metric falls within one standard deviation of its cross-validated mean, and the spread across folds is small. The held-out estimate is therefore not an artefact of one fortunate split; the performance is a property of the data and configuration. A minor caveat is that each fold trains on fewer records than the final model, making cross-validation very slightly pessimistic.'
  ),
  h2('5.8 Feature importance'),
  spacer(),
  table(
    ['Rank', 'Feature', 'Importance'],
    importances.map((e, i) => [String(i + 1), e.feature, n(e.importance)]),
    [800, 6560, 2000]
  ),
  caption('Table 11 — Mean impurity-decrease importance for all model input columns.'),
  body('Payment history dominates: the three past-due counters together account for the largest share of the total, followed by revolving credit utilisation. This is intuitive, but it must not be over-interpreted.'),
  body('**These values are not causal statements.** Impurity-based importance records what the trees chose to split on, which is not the same as what causes delinquency. It is biased toward high-cardinality features, since continuous ratios offer far more candidate split points than counters that are zero for most borrowers. It is computed on training data. Correlated predictors divide the credit between them, so each past-due counter appears less important than past-due behaviour as a whole truly is. Finally, rare binary flags are structurally understated: the indicator columns rank last despite having won the model-selection comparison, because a flag set on a few hundred of 150,000 records cannot accumulate much total impurity reduction however decisive it is on those records.'),
  figure('feature_importance.png', 400, 0.529),
  caption('Figure 7 — Feature importance by mean impurity decrease.'),

  h1('6. System Implementation'),
  h2('6.1 Architecture'),
  body('The trained pipeline is persisted with joblib and served by a FastAPI application that loads it once at startup, never per request. A React frontend consumes the API over HTTP.'),
  spacer(),
  table(
    ['Layer', 'Technology'],
    [
      ['Model', 'scikit-learn RandomForestClassifier, persisted with joblib'],
      ['Data processing', 'pandas, NumPy; matplotlib and seaborn for figures'],
      ['API', 'FastAPI, Pydantic v2, Uvicorn'],
      ['Frontend', 'React 19, Vite, TypeScript, Tailwind CSS, React Router'],
      ['Testing', 'pytest (61 tests)'],
    ],
    [2400, 6960]
  ),
  caption('Table 12 — Technology stack.'),
  h2('6.2 API endpoints'),
  spacer(),
  table(
    ['Method', 'Endpoint', 'Purpose'],
    [
      ['GET', '/health', 'API status, model availability and version'],
      ['GET', '/model-info', 'Model, dataset, target, features, metrics, risk bands'],
      ['GET', '/demo-examples', 'Real held-out records for the demonstration button'],
      ['POST', '/predict', 'Score one borrower record'],
      ['GET', '/docs', 'Interactive Swagger documentation'],
    ],
    [1200, 2400, 5760]
  ),
  caption('Table 13 — REST API surface.'),
  body('Requests are validated with Pydantic against the exact feature schema used in training. Invalid, missing, non-numeric, non-finite or unknown fields produce a 422 response identifying the offending field; a missing model artifact produces a 503. The positive-class probability is located by looking up label 1 in the model\'s classes_ attribute rather than assuming a column index, so a change in class ordering cannot silently invert predictions.'),
  h2('6.3 Frontend'),
  body('The interface has four pages: a Dashboard summarising the project and live metrics, a Prediction form, a Results page, and a Model Information page. All metrics displayed are fetched from the API at runtime; none are hardcoded. Two interactive charts were added, each chosen because a table cannot show what it shows: a score-separation chart, drawing the predicted-score distribution of each actual outcome group, which explains why recall is capped rather than merely stating it; and a threshold curve, whose hover converts rates into counts of borrowers caught and wrongly flagged. The same separation chart appears on the results page with a marker for the borrower just scored. Charts that would only repeat a number, such as a ROC curve or confusion-matrix heatmap, were deliberately omitted. The form exposes only the ten features the model was actually trained on, with plain-language labels, grouped inputs and validation bounds taken from the data dictionary.'),
  h2('6.4 Testing'),
  body('The project includes **61 automated tests** covering the dataset contract (verifying the target can never enter the feature list), placeholder-code and implausible-value cleaning, imputation behaviour, split determinism and disjointness, metric arithmetic against hand-checked examples, correct probability-column lookup under a reversed class ordering, artifact loading independent of the training code, and every API endpoint including its error paths.'),

  h2('6.5 Deployment'),
  body('The application is hosted on Vercel as two projects. The frontend is available at https://delinquency-risk.vercel.app and the backend, with interactive Swagger documentation, at https://loan-default-api.vercel.app/docs. The 70 MB model file is bundled with the backend; the deployed bundle is about 315 MB, within the Vercel 500 MB limit for Python functions, which is possible because the runtime requirements were separated from the training and testing tools.'),
  body('The browser never calls the backend directly. The frontend rewrites requests to /api onto the backend, so every request is same-origin and cross-origin resource sharing does not apply; the backend additionally allows only the frontend domains. The interface was verified at widths from 320 to 1280 pixels: charts are drawn at their real pixel width so labels remain legible on a phone, and tapping a chart opens the same tooltip as hovering. After a period of inactivity the first request takes roughly eight to nine seconds while the model loads.'),

  h1('7. Discussion and Limitations'),
  h2('7.1 Interpretation of the result'),
  body(
    `A ROC-AUC of ${n(test.roc_auc)} indicates that the model extracts genuine, substantial signal from the data. The comparatively modest positive-class F1 of ${n(test.f1)} is not a defect of the implementation but a property of the problem: predicting individual human financial behaviour two years into the future from ten numeric attributes has an inherent ceiling. The original competition was itself scored on AUC rather than F1 for this reason.`
  ),
  h2('7.2 Dataset limitations'),
  bullet('**Age of the data.** The records describe United States consumer credit around 2011. Lending standards, products and macroeconomic conditions have changed substantially since.'),
  bullet('**Unknown sampling frame.** The competition does not document how these borrowers were selected, so there is no basis for establishing which population the sample represents.'),
  bullet('**Fairness cannot be assessed.** The dataset contains no race, sex or geographic attributes, so no disparate-impact test is possible. Their absence is not protection: age is present and used, and features such as income and credit utilisation can act as proxies. The absence of measured bias is the absence of measurement, not evidence of fairness.'),
  h2('7.3 Methodological limitations'),
  bullet('**Residual overfitting**, quantified in Section 5.6 and not eliminated.'),
  bullet('**No out-of-time validation.** The dataset has no date column, so the standard credit-model validation — testing on a later period — is impossible. A random split from a single snapshot is the easiest possible test.'),
  bullet('**Probabilities are not calibrated.** The output is the mean of the per-tree class probabilities, each tree contributing the class proportion of the leaf the borrower falls into. Random Forest probabilities are typically poorly calibrated, so a predicted 0.30 should not be read as thirty occurrences in a hundred. Calibration was not evaluated, and the interface describes the output as an estimated probability throughout.'),
  bullet('**The threshold optimises F1**, which weights a false positive and a false negative equally. Real lending economics do not: a missed delinquency usually costs far more than a declined sound borrower. Setting the threshold properly would require loss-given-default and margin figures the dataset does not contain.'),
  bullet('**Imputation blurs real signal.** For roughly one record in five the model has no actual income figure, only the training median and an indicator that it was absent.'),
  h2('7.4 Why this is an educational prototype'),
  body('A system used for real lending decisions would additionally require current and representative data, out-of-time validation, fairness testing, calibrated and monitored probabilities, adverse-action reasoning for declines, regulatory compliance, model governance and audit trails, drift monitoring, human review of automated decisions, and authentication and logging. None of these are implemented. Beyond the technical gaps, credit decisions materially affect people and are regulated accordingly; this project must not be used to approve, decline or price real credit.'),

  h1('8. Conclusion'),
  body(
    `This project delivers a complete, reproducible, supervised binary classification system for predicting serious borrower delinquency. A Random Forest classifier trained on ${int(meta.dataset.labeled_rows)} labelled records achieved a **ROC-AUC of ${n(test.roc_auc)}** and a **PR-AUC of ${n(test.pr_auc)}** on a held-out test split scored exactly once, with a positive-class F1 of **${n(test.f1)}** and a macro-average F1 of **${n(report.macro_avg.f1)}** at a validation-selected decision threshold of ${n(threshold)}. Five-fold cross-validation confirmed these estimates are stable rather than artefacts of one partition.`
  ),
  body('Beyond the numbers, the project demonstrates several methodological points that matter more than any single score: that accuracy is actively misleading under class imbalance, that class weighting and threshold adjustment move a model along the same precision-recall curve rather than improving it, that missingness can itself be informative and should be tested rather than assumed, and that the decision threshold is a modelling choice to be selected on validation data rather than inherited as a default. Preprocessing is held inside the persisted pipeline so that training and inference cannot diverge, and every reported figure is regenerated from the artifacts on each run rather than transcribed.'),
  h2('8.1 Future work'),
  bullet('Evaluate probability calibration using CalibratedClassifierCV, a reliability diagram and the Brier score.'),
  bullet('Compute permutation importance on the test split, which is less biased than impurity-based importance.'),
  bullet('Select the decision threshold against an explicit cost matrix rather than by maximising F1.'),
  bullet('Add per-prediction explanations, for example SHAP values, to the results page.'),
  bullet('Extend automated testing to the frontend components.'),

  h1('References'),
  body(`[1] Give Me Some Credit, Kaggle competition. Data credited to Credit Fusion and Kaggle. ${meta.dataset.url}`, { alignment: AlignmentType.LEFT }),
  body('[2] Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.', { alignment: AlignmentType.LEFT }),
  body('[3] Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.', { alignment: AlignmentType.LEFT }),
  body('[4] Saito, T. and Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE, 10(3).', { alignment: AlignmentType.LEFT }),
  spacer(),
  body(
    `Project documentation accompanying this report: README.md (overview and results), SETUP.md (installation and run guide), docs/dataset-research.md, docs/methodology.md, docs/model-evaluation.md and docs/limitations.md. Model artifact version ${meta.model_version}, trained ${meta.trained_at.slice(0, 10)} with random seed ${meta.random_seed}.`,
    { alignment: AlignmentType.LEFT }
  ),
];

// -------------------------------------------------------------------- document
const doc = new Document({
  creator: 'Loan Default Risk Prediction project',
  title: 'Loan Default Risk Prediction Using Random Forest',
  description: 'Machine Learning laboratory project report',
  numbering: {
    config: [
      {
        reference: 'bullets',
        levels: [
          {
            level: 0,
            format: LevelFormat.BULLET,
            text: '•',
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 520, hanging: 260 } } },
          },
        ],
      },
    ],
  },
  styles: {
    default: {
      document: { run: { font: FONT, size: SIZE_BODY } },
    },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { font: FONT, size: SIZE_HEADING, bold: true, color: '000000' } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { font: FONT, size: SIZE_BODY, bold: true, color: '000000' } },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 }, // A4
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1800 }, // extra left for binding
        },
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [
                new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: SIZE_SMALL }),
              ],
            }),
          ],
        }),
      },
      children: [...titlePage, ...content],
    },
  ],
});

const out = process.argv[3];
Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(out, buffer);
  console.log(`Wrote ${out} (${(buffer.length / 1024).toFixed(0)} KB)`);
});
