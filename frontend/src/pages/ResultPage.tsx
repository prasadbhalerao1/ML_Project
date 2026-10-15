/** Page 3 — the prediction result for the most recent submission. */

import { Link } from 'react-router-dom';
import { ProbabilityBar } from '../components/ProbabilityBar';
import { ScoreSeparationChart } from '../components/charts/ScoreSeparationChart';
import {
  Alert,
  Card,
  PageHeader,
  formatPercent,
} from '../components/ui';
import { useAsync } from '../hooks/useAsync';
import { usePrediction, type PredictionRecord } from '../hooks/usePrediction';
import { getModelInfo } from '../services/api';
import { FEATURE_FIELDS } from '../services/featureFields';
import type { ModelInfo } from '../types/api';

export function ResultPage() {
  const { record } = usePrediction();
  // Loaded only for the risk-band boundaries drawn on the probability bar.
  const { data: info } = useAsync<ModelInfo>(getModelInfo);

  if (!record) {
    return (
      <div className="space-y-6">
        <PageHeader title="Prediction result" />
        <Alert tone="info" title="No prediction yet">
          Submit the borrower form to see a result here.{' '}
          <Link to="/predict" className="font-medium underline">
            Go to the prediction form
          </Link>
          .
        </Alert>
      </div>
    );
  }

  return <ResultView record={record} info={info} />;
}

function ResultView({
  record,
  info,
}: {
  record: PredictionRecord;
  info: ModelInfo | null;
}) {
  const { request, response } = record;
  const isPositive = response.predicted_class === 1;

  return (
    <div className="space-y-6">
      <PageHeader title="Prediction result">
        A model estimate for the inputs you submitted, under the dataset's
        target definition.
      </PageHeader>

      <Card>
        <div className="flex flex-wrap items-center gap-3">
          <span
            className={`rounded-full px-3 py-1 text-sm font-semibold ${
              isPositive
                ? 'bg-red-100 text-red-800'
                : 'bg-emerald-100 text-emerald-800'
            }`}
          >
            Predicted: {response.predicted_label}
          </span>
          <span className="rounded-full bg-slate-100 px-3 py-1 text-sm font-medium text-slate-700">
            Risk band: {response.risk_band}
          </span>
          <span className="text-xs text-slate-500">
            model v{response.model_version}
          </span>
        </div>

        <div className="mt-6">
          <ProbabilityBar
            probability={response.default_probability}
            band={response.risk_band}
            bands={info?.risk_bands}
          />
        </div>

        <dl className="mt-6 grid gap-4 text-sm sm:grid-cols-3">
          <div>
            <dt className="font-medium text-slate-500">Estimated probability</dt>
            <dd className="mt-0.5 tabular-nums text-slate-900">
              {formatPercent(response.default_probability, 2)}
            </dd>
          </div>
          <div>
            <dt className="font-medium text-slate-500">Predicted class</dt>
            <dd className="mt-0.5 text-slate-900">
              {response.predicted_class} — {response.predicted_label}
            </dd>
          </div>
          <div>
            <dt className="font-medium text-slate-500">Decision threshold</dt>
            <dd className="mt-0.5 tabular-nums text-slate-900">
              {response.classification_threshold.toFixed(4)}
            </dd>
          </div>
        </dl>
      </Card>

      {info && (
        <Card
          title="Where this borrower sits"
          description="The same score plotted against 30,000 held-out borrowers whose actual outcome is known, so the number above has something to be compared with."
        >
          <ScoreSeparationChart
            distribution={info.test_score_distribution}
            threshold={response.classification_threshold}
            marker={{
              probability: response.default_probability,
              label: 'This borrower',
            }}
          />
          <p className="mt-3 text-sm text-slate-600">
            A probability only means something relative to the population it
            came from. The teal marker shows where this borrower falls. Note
            that both outcome groups are present across most of the range: a
            score in the overlap is genuinely ambiguous, not a near-certain
            verdict either way.
          </p>
        </Card>
      )}

      <Card title="How to read this">
        <ul className="list-disc space-y-2 pl-5 text-sm text-slate-700">
          <li>
            <strong>Target.</strong> {response.target_definition}. The result
            says nothing about any other kind of default.
          </li>
          <li>
            <strong>Probability.</strong> The average across the forest's 300
            trees of the proportion of delinquent borrowers in the leaf this
            borrower lands in. It has not been calibrated against
            observed default frequencies, so treat it as a relative score rather
            than a literal real-world chance.
          </li>
          <li>
            <strong>Predicted class.</strong> Simply whether the probability
            reaches the {response.classification_threshold.toFixed(4)}{' '}
            threshold, which was chosen by maximising F1 on the validation
            split. A different threshold would trade recall against precision —
            see the model information page.
          </li>
          <li>
            <strong>Risk band.</strong> {response.risk_band_disclaimer}
          </li>
        </ul>

        <Alert tone="warning" title="Not a lending decision">
          This is a student project trained on a 2011 US credit dataset. It
          cannot tell you what a borrower will do, and must not be used to
          approve, decline or price real credit.
        </Alert>
      </Card>

      <Card title="Inputs you submitted">
        <dl className="grid gap-x-6 gap-y-3 sm:grid-cols-2">
          {FEATURE_FIELDS.map((field) => {
            const value = request[field.key];
            return (
              <div
                key={field.key}
                className="flex items-baseline justify-between gap-4 border-b border-slate-100 pb-2"
              >
                <dt className="text-sm text-slate-600">{field.label}</dt>
                <dd className="text-sm font-medium tabular-nums text-slate-900">
                  {value === null ? (
                    <span className="font-normal text-slate-400">
                      not provided (imputed)
                    </span>
                  ) : (
                    value.toLocaleString()
                  )}
                </dd>
              </div>
            );
          })}
        </dl>
        <Link
          to="/predict"
          className="mt-5 inline-flex rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
        >
          Edit inputs and predict again
        </Link>
      </Card>
    </div>
  );
}
