/** Page 2 — the borrower input form. */

import { useCallback, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { NumberField } from '../components/NumberField';
import { Alert, Card, PageHeader, Spinner } from '../components/ui';
import { usePrediction } from '../hooks/usePrediction';
import { usePredictionForm } from '../hooks/usePredictionForm';
import { useAsync } from '../hooks/useAsync';
import { getDemoExamples, postPrediction } from '../services/api';
import { FEATURE_GROUPS } from '../services/featureFields';
import type { DemoExample, PredictionRequest } from '../types/api';

export function PredictPage() {
  const navigate = useNavigate();
  const { setRecord } = usePrediction();

  const handleSubmit = useCallback(
    async (request: PredictionRequest) => {
      const response = await postPrediction(request);
      setRecord({ request, response });
      navigate('/result');
    },
    [navigate, setRecord],
  );

  const form = usePredictionForm(handleSubmit);

  return (
    <div className="space-y-6">
      <PageHeader title="Estimate delinquency risk">
        Enter the ten borrower attributes the trained model uses. These are the
        only inputs it was fitted on — there is no loan amount, interest rate or
        employment field, because the dataset does not contain them.
      </PageHeader>

      <DemoExampleLoader onLoad={form.setAllValues} />

      <form
        noValidate
        onSubmit={(event) => {
          event.preventDefault();
          void form.submit();
        }}
        className="space-y-6"
      >
        {FEATURE_GROUPS.map((group) => (
          <Card
            key={group.title}
            title={group.title}
            description={group.description}
          >
            <div className="grid gap-5 sm:grid-cols-2">
              {group.fields.map((field) => (
                <NumberField
                  key={field.key}
                  field={field}
                  value={form.values[field.key]}
                  error={form.fieldErrors[field.key]}
                  onChange={(value) => form.setValue(field.key, value)}
                />
              ))}
            </div>
          </Card>
        ))}

        {form.submitError && <Alert title="Could not predict">{form.submitError}</Alert>}

        <div className="flex flex-wrap items-center gap-3">
          <button
            type="submit"
            disabled={form.submitting}
            className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-indigo-400"
          >
            {form.submitting ? 'Predicting…' : 'Predict risk'}
          </button>
          <button
            type="button"
            onClick={form.reset}
            disabled={form.submitting}
            className="rounded-lg border border-slate-300 bg-white px-4 py-2.5 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:opacity-60"
          >
            Reset
          </button>
          {form.submitting && <Spinner label="Scoring with the model…" />}
        </div>
      </form>
    </div>
  );
}

/** Button row that fills the form from a real held-out dataset record. */
function DemoExampleLoader({
  onLoad,
}: {
  onLoad: (features: Record<string, number>) => void;
}) {
  const { data, loading, error } = useAsync<DemoExample[]>(getDemoExamples);
  const [notice, setNotice] = useState<string | null>(null);

  if (loading) {
    return (
      <Card>
        <Spinner label="Loading demonstration records…" />
      </Card>
    );
  }

  if (error) {
    return (
      <Alert tone="info" title="Demonstration records unavailable">
        {error} You can still fill the form by hand.
      </Alert>
    );
  }

  if (!data || data.length === 0) return null;

  return (
    <Card
      title="Load demonstration data"
      description="Real rows from the public Kaggle dataset's held-out test split. This is demonstration data about past, anonymous borrowers — not information about you."
    >
      <div className="flex flex-wrap gap-3">
        {data.map((example) => (
          <button
            key={example.label}
            type="button"
            onClick={() => {
              onLoad(example.features);
              setNotice(
                `Loaded the ${describeDemoLabel(example.label)} example. Its recorded outcome in the dataset was “${example.actual_outcome === 1 ? 'serious delinquency' : 'no serious delinquency'}”.`,
              );
            }}
            className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-indigo-300 hover:bg-indigo-50"
          >
            {describeDemoLabel(example.label)} example
          </button>
        ))}
      </div>
      {notice && (
        <p role="status" className="mt-3 text-xs text-slate-600">
          {notice}
        </p>
      )}
    </Card>
  );
}

const describeDemoLabel = (label: string) => label.replace(/_/g, '-');
