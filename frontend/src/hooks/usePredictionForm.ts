/**
 * Form state, client-side validation and submission for the prediction form.
 *
 * Kept out of the page component so the page stays presentational. Values are
 * held as strings because that is what an `<input>` gives us, and are only
 * converted to numbers once they pass validation.
 */

import { useCallback, useState } from 'react';
import { FEATURE_FIELDS, type FeatureField } from '../services/featureFields';
import { describeError } from './useAsync';
import type { FeatureKey, PredictionRequest } from '../types/api';

export type FormValues = Record<FeatureKey, string>;
export type FieldErrors = Partial<Record<FeatureKey, string>>;

/** A plausible mid-range starting point, so the form is never blank. */
const INITIAL_VALUES: FormValues = {
  RevolvingUtilizationOfUnsecuredLines: '0.3',
  age: '45',
  'NumberOfTime30-59DaysPastDueNotWorse': '0',
  DebtRatio: '0.35',
  MonthlyIncome: '5400',
  NumberOfOpenCreditLinesAndLoans: '8',
  NumberOfTimes90DaysLate: '0',
  NumberRealEstateLoansOrLines: '1',
  'NumberOfTime60-89DaysPastDueNotWorse': '0',
  NumberOfDependents: '1',
};

export function usePredictionForm(
  onSuccess: (request: PredictionRequest) => Promise<void> | void,
) {
  const [values, setValues] = useState<FormValues>(INITIAL_VALUES);
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({});
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const setValue = useCallback((key: FeatureKey, value: string) => {
    setValues((current) => ({ ...current, [key]: value }));
    setFieldErrors((current) => {
      if (!current[key]) return current;
      const { [key]: _removed, ...rest } = current;
      return rest;
    });
  }, []);

  /** Replace every value at once, e.g. when loading a demonstration record. */
  const setAllValues = useCallback((record: Record<string, number>) => {
    setValues(
      Object.fromEntries(
        FEATURE_FIELDS.map((field) => [
          field.key,
          record[field.key] === undefined ? '' : String(record[field.key]),
        ]),
      ) as FormValues,
    );
    setFieldErrors({});
    setSubmitError(null);
  }, []);

  const reset = useCallback(() => {
    setValues(INITIAL_VALUES);
    setFieldErrors({});
    setSubmitError(null);
  }, []);

  const submit = useCallback(async () => {
    const { errors, request } = validate(values);
    setFieldErrors(errors);
    if (!request) {
      setSubmitError('Please correct the highlighted fields.');
      return;
    }

    setSubmitError(null);
    setSubmitting(true);
    try {
      await onSuccess(request);
    } catch (caught) {
      setSubmitError(describeError(caught));
    } finally {
      setSubmitting(false);
    }
  }, [onSuccess, values]);

  return {
    values,
    fieldErrors,
    submitError,
    submitting,
    setValue,
    setAllValues,
    reset,
    submit,
  };
}

function validate(values: FormValues): {
  errors: FieldErrors;
  request: PredictionRequest | null;
} {
  const errors: FieldErrors = {};
  const request = {} as PredictionRequest;

  for (const field of FEATURE_FIELDS) {
    const raw = values[field.key].trim();

    if (raw === '') {
      if (field.optional) {
        // Null tells the API to fall back to the pipeline's median imputer.
        request[field.key] = null;
        continue;
      }
      errors[field.key] = 'This field is required.';
      continue;
    }

    const parsed = Number(raw);
    const problem = describeValueProblem(parsed, field);
    if (problem) {
      errors[field.key] = problem;
      continue;
    }
    request[field.key] = parsed;
  }

  return Object.keys(errors).length > 0
    ? { errors, request: null }
    : { errors, request };
}

function describeValueProblem(value: number, field: FeatureField): string | null {
  if (!Number.isFinite(value)) return 'Enter a finite number.';
  if (value < field.min) return `Must be ${field.min} or more.`;
  if (value > field.max) return `Must be ${field.max.toLocaleString()} or less.`;
  if (field.step === 1 && !Number.isInteger(value)) {
    return 'Enter a whole number.';
  }
  return null;
}
