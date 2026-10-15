/**
 * Form metadata for the ten model inputs: one entry per dataset column.
 *
 * `key` is the dataset column name, which is also the JSON key the API
 * expects, so the form cannot drift from the trained feature schema. Bounds
 * mirror the Pydantic constraints in backend/app/schemas/prediction.py.
 */

import type { FeatureKey } from '../types/api';

export interface FeatureField {
  key: FeatureKey;
  label: string;
  /** Short plain-language note for the less obvious financial variables. */
  help?: string;
  min: number;
  max: number;
  step: number;
  /** Optional fields are median-imputed by the pipeline when left blank. */
  optional?: boolean;
  unit?: string;
}

export interface FeatureGroup {
  title: string;
  description: string;
  fields: FeatureField[];
}

export const FEATURE_GROUPS: FeatureGroup[] = [
  {
    title: 'Borrower profile',
    description: 'Basic details about the applicant.',
    fields: [
      {
        key: 'age',
        label: 'Age',
        min: 18,
        max: 120,
        step: 1,
        unit: 'years',
      },
      {
        key: 'MonthlyIncome',
        label: 'Monthly income',
        help: 'Gross monthly income. Leave blank if unknown — the model imputes the training median, exactly as it did for the 20% of training rows with no recorded income.',
        min: 0,
        max: 10_000_000,
        step: 100,
        optional: true,
      },
      {
        key: 'NumberOfDependents',
        label: 'Number of dependents',
        help: 'People in the family excluding the borrower. Leave blank if unknown.',
        min: 0,
        max: 30,
        step: 1,
        optional: true,
      },
    ],
  },
  {
    title: 'Credit usage',
    description: 'How much credit is held and how much of it is drawn.',
    fields: [
      {
        key: 'RevolvingUtilizationOfUnsecuredLines',
        label: 'Revolving credit utilisation',
        help: 'Credit-card and personal line-of-credit balances divided by total credit limits. 0.3 means 30% of available credit is used; values above 1 mean over-limit balances.',
        min: 0,
        max: 100_000,
        step: 0.01,
        unit: 'ratio',
      },
      {
        key: 'DebtRatio',
        label: 'Debt ratio',
        help: 'Monthly debt payments, alimony and living costs divided by monthly gross income.',
        min: 0,
        max: 100_000,
        step: 0.01,
        unit: 'ratio',
      },
      {
        key: 'NumberOfOpenCreditLinesAndLoans',
        label: 'Open credit lines and loans',
        help: 'Open installment loans plus lines of credit, e.g. car loans and credit cards.',
        min: 0,
        max: 100,
        step: 1,
      },
      {
        key: 'NumberRealEstateLoansOrLines',
        label: 'Real estate loans or lines',
        help: 'Mortgages and real estate loans, including home equity lines of credit.',
        min: 0,
        max: 100,
        step: 1,
      },
    ],
  },
  {
    title: 'Payment history (last two years)',
    description:
      'Counts of past-due episodes. These are the strongest signals in the trained model.',
    fields: [
      {
        key: 'NumberOfTime30-59DaysPastDueNotWorse',
        label: 'Times 30–59 days past due',
        help: 'Late, but never worse than 59 days, on that occasion.',
        min: 0,
        max: 100,
        step: 1,
      },
      {
        key: 'NumberOfTime60-89DaysPastDueNotWorse',
        label: 'Times 60–89 days past due',
        help: 'Late, but never worse than 89 days, on that occasion.',
        min: 0,
        max: 100,
        step: 1,
      },
      {
        key: 'NumberOfTimes90DaysLate',
        label: 'Times 90+ days past due',
        min: 0,
        max: 100,
        step: 1,
      },
    ],
  },
];

/** Every field, flattened, for lookups and iteration. */
export const FEATURE_FIELDS: FeatureField[] = FEATURE_GROUPS.flatMap(
  (group) => group.fields,
);
