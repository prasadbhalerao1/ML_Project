/** Page 4 — full model, dataset and evaluation details from the API. */

import {
  Alert,
  Card,
  DetailRow,
  KaggleLink,
  PageHeader,
  Spinner,
  formatMetric,
  formatPercent,
} from '../components/ui';
import { ScoreSeparationChart } from '../components/charts/ScoreSeparationChart';
import { ThresholdCurveChart } from '../components/charts/ThresholdCurveChart';
import { useAsync } from '../hooks/useAsync';
import { getModelInfo } from '../services/api';
import { FEATURE_FIELDS } from '../services/featureFields';
import type { ModelInfo } from '../types/api';

export function ModelInfoPage() {
  const { data, loading, error, reload } = useAsync<ModelInfo>(getModelInfo);

  return (
    <div className="space-y-6">
      <PageHeader title="Model information">
        Everything below is read from the metadata saved by the training run —
        no figures are hardcoded in this page.
      </PageHeader>

      {loading && (
        <Card>
          <Spinner label="Loading model information…" />
        </Card>
      )}
      {error && (
        <Alert title="Could not load model information" onRetry={reload}>
          {error}
        </Alert>
      )}
      {data && (
        <>
          <Overview info={data} />
          <Metrics info={data} />
          <ScoreSeparation info={data} />
          <ThresholdTable info={data} />
          <Features info={data} />
        </>
      )}
    </div>
  );
}

function Overview({ info }: { info: ModelInfo }) {
  return (
    <Card title="Overview">
      <dl>
        <DetailRow label="Algorithm">{info.algorithm}</DetailRow>
        <DetailRow label="Model">
          {info.model_name} (version {info.model_version})
        </DetailRow>
        <DetailRow label="Trained at">{info.trained_at}</DetailRow>
        <DetailRow label="Dataset">{info.dataset_name}</DetailRow>
        <DetailRow label="Dataset source">
          <KaggleLink href={info.dataset_url} />
        </DetailRow>
        <DetailRow label={`Target (${info.target_column})`}>
          {info.target_definition}
        </DetailRow>
        <DetailRow label="Class order">
          predict_proba columns map to classes [{info.class_order.join(', ')}]
        </DetailRow>
        <DetailRow label="Decision threshold">
          {info.classification_threshold.toFixed(4)}
          <p className="mt-1 text-xs text-slate-500">
            Chosen by maximising F1 on the validation split, not assumed to be
            0.50. The test split played no part in the choice.
          </p>
        </DetailRow>
        <DetailRow label="Risk bands">
          <ul className="space-y-0.5">
            {info.risk_bands.map((band) => (
              <li key={band.name}>
                <span className="font-medium">{band.name}</span>:{' '}
                {formatPercent(band.lower_bound_inclusive, 0)} –{' '}
                {band.upper_bound_exclusive === null
                  ? '100%'
                  : formatPercent(band.upper_bound_exclusive, 0)}
              </li>
            ))}
          </ul>
          <p className="mt-1.5 text-xs text-slate-500">
            {info.risk_band_disclaimer}
          </p>
        </DetailRow>
      </dl>
    </Card>
  );
}

function Metrics({ info }: { info: ModelInfo }) {
  const { test_metrics: metrics, class_distribution: distribution } = info;
  const matrix = metrics.confusion_matrix;

  return (
    <Card
      title="Held-out test metrics"
      description={`At threshold ${metrics.threshold.toFixed(2)}. The test split was never used for model selection.`}
    >
      <table className="w-full text-sm">
        <caption className="sr-only">Test-set evaluation metrics</caption>
        <tbody>
          {(
            [
              ['Accuracy', metrics.accuracy],
              ['Precision', metrics.precision],
              ['Recall', metrics.recall],
              ['F1-score', metrics.f1],
              ['ROC-AUC', metrics.roc_auc],
              ['PR-AUC (average precision)', metrics.pr_auc],
            ] as const
          ).map(([label, value]) => (
            <tr key={label} className="border-b border-slate-100 last:border-0">
              <th scope="row" className="py-2 text-left font-medium text-slate-600">
                {label}
              </th>
              <td className="py-2 text-right tabular-nums text-slate-900">
                {formatMetric(value)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3 className="mt-6 text-sm font-semibold text-slate-900">
        Confusion matrix
      </h3>
      <div className="mt-2 overflow-x-auto">
        <table className="min-w-full border border-slate-200 text-sm">
          <thead>
            <tr className="bg-slate-50 text-slate-600">
              <th className="p-2" />
              <th scope="col" className="p-2 font-medium">
                Predicted 0
              </th>
              <th scope="col" className="p-2 font-medium">
                Predicted 1
              </th>
            </tr>
          </thead>
          <tbody className="tabular-nums">
            <tr className="border-t border-slate-200">
              <th scope="row" className="bg-slate-50 p-2 text-left font-medium text-slate-600">
                Actual 0
              </th>
              <td className="p-2 text-center">
                {matrix.true_negatives.toLocaleString()}
                <span className="block text-xs text-slate-500">
                  true negatives
                </span>
              </td>
              <td className="p-2 text-center">
                {matrix.false_positives.toLocaleString()}
                <span className="block text-xs text-slate-500">
                  false positives
                </span>
              </td>
            </tr>
            <tr className="border-t border-slate-200">
              <th scope="row" className="bg-slate-50 p-2 text-left font-medium text-slate-600">
                Actual 1
              </th>
              <td className="p-2 text-center">
                {matrix.false_negatives.toLocaleString()}
                <span className="block text-xs text-slate-500">
                  false negatives
                </span>
              </td>
              <td className="p-2 text-center">
                {matrix.true_positives.toLocaleString()}
                <span className="block text-xs text-slate-500">
                  true positives
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <h3 className="mt-6 text-sm font-semibold text-slate-900">
        Class distribution
      </h3>
      <ul className="mt-2 space-y-1 text-sm text-slate-700">
        {Object.entries(distribution).map(([split, counts]) => (
          <li key={split}>
            <span className="font-medium capitalize">{split}</span>:{' '}
            {counts.total.toLocaleString()} rows,{' '}
            {counts.positive.toLocaleString()} positive (
            {formatPercent(counts.positive_rate, 2)})
          </li>
        ))}
      </ul>
    </Card>
  );
}

function ThresholdTable({ info }: { info: ModelInfo }) {
  return (
    <Card
      title="Threshold trade-off"
      description="The same trained model at different probability cut-offs. Lowering the threshold catches more delinquencies (higher recall) at the cost of more false alarms (lower precision)."
    >
      <ThresholdCurveChart
        curve={info.test_threshold_curve}
        threshold={info.classification_threshold}
      />
      <p className="mt-3 mb-5 text-sm text-slate-600">
        Hover any point to see what that cut-off would cost in people rather
        than percentages. F1 has a broad, flat peak rather than one sharp
        optimum, so the exact value matters less than being in the right
        region. The curve stops where the model flags fewer than 20 borrowers —
        past that, precision is computed from a handful of cases and swings
        between 0 and 1, which would read as signal when it is noise. The table
        below gives the same trade-off at four reference points.
      </p>
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead className="text-slate-600">
            <tr className="border-b border-slate-200">
              <th scope="col" className="py-2 text-left font-medium">
                Threshold
              </th>
              <th scope="col" className="py-2 text-right font-medium">
                Precision
              </th>
              <th scope="col" className="py-2 text-right font-medium">
                Recall
              </th>
              <th scope="col" className="py-2 text-right font-medium">
                F1
              </th>
              <th scope="col" className="py-2 text-right font-medium">
                Flagged
              </th>
            </tr>
          </thead>
          <tbody className="tabular-nums">
            {info.test_threshold_sweep.map((row) => (
              <tr key={row.threshold} className="border-b border-slate-100 last:border-0">
                <td className="py-2">{row.threshold.toFixed(2)}</td>
                <td className="py-2 text-right">{formatMetric(row.precision)}</td>
                <td className="py-2 text-right">{formatMetric(row.recall)}</td>
                <td className="py-2 text-right">{formatMetric(row.f1)}</td>
                <td className="py-2 text-right">
                  {(
                    row.confusion_matrix.true_positives +
                    row.confusion_matrix.false_positives
                  ).toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function ScoreSeparation({ info }: { info: ModelInfo }) {
  return (
    <Card
      title="Why the errors happen"
      description="Each curve is one actual outcome group, drawn as a share of its own total because negatives outnumber positives roughly 14 to 1. Where the two overlap, no threshold can tell the groups apart."
    >
      <ScoreSeparationChart
        distribution={info.test_score_distribution}
        threshold={info.classification_threshold}
      />
      <p className="mt-3 text-sm text-slate-600">
        Most borrowers who stayed current score near zero, which is why the
        model is accurate overall. Borrowers who did become delinquent are
        spread across the whole range rather than clustered at the top — so a
        large share of them sit inside the same region as the healthy
        borrowers. That overlap, not a flaw in the implementation, is what caps
        recall. Moving the dashed line left catches more of the orange area but
        sweeps up more blue with it.
      </p>
    </Card>
  );
}

function Features({ info }: { info: ModelInfo }) {
  const labels = new Map(FEATURE_FIELDS.map((field) => [field.key as string, field.label]));
  const INDICATOR_PREFIX = 'missingindicator_';

  /** Readable name for a model input column, including the missing-value flags. */
  const describe = (feature: string): string | undefined => {
    if (feature.startsWith(INDICATOR_PREFIX)) {
      const source = feature.slice(INDICATOR_PREFIX.length);
      return `Was missing: ${labels.get(source) ?? source}`;
    }
    return labels.get(feature);
  };
  const maxImportance = Math.max(
    ...info.feature_importances.map((entry) => entry.importance),
    0.0001,
  );

  return (
    <Card
      title="Input features and importance"
      description="Importance is the forest's mean impurity decrease: it shows what the trees split on, not what causes delinquency."
    >
      <ol className="space-y-3">
        {info.feature_importances.map((entry) => (
          <li key={entry.feature}>
            <div className="flex items-baseline justify-between gap-4 text-sm">
              {/* min-w-0 + break-words let a long column name wrap instead of
                  pushing the value off a narrow screen. */}
              <span className="min-w-0 break-words text-slate-800">
                {describe(entry.feature) ?? entry.feature}
              </span>
              <span className="shrink-0 tabular-nums text-slate-500">
                {entry.importance.toFixed(4)}
              </span>
            </div>
            <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
              <div
                className="h-full rounded-full bg-indigo-500"
                style={{ width: `${(entry.importance / maxImportance) * 100}%` }}
              />
            </div>
            {/* Only worth showing when the readable label differs from the
                raw column name; for the indicator columns they are identical. */}
            {describe(entry.feature) !== undefined && (
              <p className="mt-0.5 break-all font-mono text-xs text-slate-400">
                {entry.feature}
              </p>
            )}
          </li>
        ))}
      </ol>
      <p className="mt-5 text-xs text-slate-500">
        Request schema: the API expects exactly these {info.features.length}{' '}
        keys, by their dataset column names.
      </p>
    </Card>
  );
}
