/** Page 1 — project overview and the model's measured test metrics. */

import { Link } from 'react-router-dom';
import { useAsync } from '../hooks/useAsync';
import { getModelInfo } from '../services/api';
import {
  Alert,
  Card,
  KaggleLink,
  MetricTile,
  PageHeader,
  Spinner,
  formatMetric,
  formatPercent,
} from '../components/ui';
import type { ModelInfo } from '../types/api';

export function DashboardPage() {
  const { data, loading, error, reload } = useAsync<ModelInfo>(getModelInfo);

  return (
    <div className="space-y-6">
      <PageHeader title="Loan delinquency risk prediction">
        A Random Forest classifier trained on the public Kaggle “Give Me Some
        Credit” dataset. It estimates the probability that a borrower
        experiences serious delinquency — 90 days past due or worse — within two
        years of the observation date.
      </PageHeader>

      <Card title="What the model does and does not do">
        <ul className="list-disc space-y-1.5 pl-5 text-sm text-slate-700">
          <li>
            It outputs a probability for one specific historical outcome:
            serious delinquency within two years.
          </li>
          <li>
            It does not predict every kind of loan default, and it does not say
            whether a borrower will repay.
          </li>
          <li>
            It is an educational prototype, not a credit-underwriting system. No
            lending decision should rest on it.
          </li>
        </ul>
        <Link
          to="/predict"
          className="mt-5 inline-flex rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700"
        >
          Try a prediction
        </Link>
      </Card>

      {loading && (
        <Card>
          <Spinner label="Loading model metrics…" />
        </Card>
      )}

      {error && (
        <Alert title="Model details unavailable" onRetry={reload}>
          {error}
        </Alert>
      )}

      {data && <MetricsSection info={data} />}
    </div>
  );
}

function MetricsSection({ info }: { info: ModelInfo }) {
  const { test_metrics: metrics, class_distribution: distribution } = info;
  const testSplit = distribution.test;

  return (
    <>
      <Card
        title="Held-out test performance"
        description={`Measured on ${testSplit?.total.toLocaleString() ?? '—'} borrowers the model never saw during training, at the ${metrics.threshold.toFixed(2)} probability threshold selected on the validation split.`}
      >
        <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          <MetricTile
            label="ROC-AUC"
            value={formatMetric(metrics.roc_auc)}
            hint="Ranking quality, threshold-free"
          />
          <MetricTile
            label="PR-AUC"
            value={formatMetric(metrics.pr_auc)}
            hint="Average precision on the rare class"
          />
          <MetricTile
            label="Accuracy"
            value={formatMetric(metrics.accuracy)}
            hint="Misleading on its own here"
          />
          <MetricTile label="Precision" value={formatMetric(metrics.precision)} />
          <MetricTile label="Recall" value={formatMetric(metrics.recall)} />
          <MetricTile label="F1" value={formatMetric(metrics.f1)} />
        </dl>

        <Alert tone="warning" title="Why accuracy alone is not enough">
          Only{' '}
          {testSplit ? formatPercent(testSplit.positive_rate, 2) : 'about 6.7%'}{' '}
          of borrowers in this dataset experienced serious delinquency. Always
          predicting “no delinquency” would already score about{' '}
          {testSplit ? formatPercent(1 - testSplit.positive_rate, 1) : '93.3%'}{' '}
          accuracy while catching nobody. Recall and PR-AUC are the metrics that
          matter.
        </Alert>
      </Card>

      <Card title="Model and data">
        <dl className="space-y-3 text-sm">
          <div>
            <dt className="font-medium text-slate-500">Model</dt>
            <dd className="text-slate-900">
              {info.model_name} v{info.model_version} — {info.algorithm}
            </dd>
          </div>
          <div>
            <dt className="font-medium text-slate-500">Dataset</dt>
            <dd className="text-slate-900">
              {info.dataset_name}
              <br />
              <KaggleLink href={info.dataset_url} />
            </dd>
          </div>
          <div>
            <dt className="font-medium text-slate-500">
              Target ({info.target_column})
            </dt>
            <dd className="text-slate-900">{info.target_definition}</dd>
          </div>
        </dl>
        <Link
          to="/model"
          className="mt-4 inline-flex text-sm font-medium text-indigo-700 hover:text-indigo-900"
        >
          Full model information →
        </Link>
      </Card>
    </>
  );
}
