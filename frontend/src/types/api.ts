/**
 * TypeScript mirrors of the FastAPI response models in
 * backend/app/schemas/prediction.py. Request keys are the dataset's own
 * column names, so they match `ModelInfo.features` exactly.
 */

/** Dataset column names the trained model expects, in training order. */
export const FEATURE_KEYS = [
  'RevolvingUtilizationOfUnsecuredLines',
  'age',
  'NumberOfTime30-59DaysPastDueNotWorse',
  'DebtRatio',
  'MonthlyIncome',
  'NumberOfOpenCreditLinesAndLoans',
  'NumberOfTimes90DaysLate',
  'NumberRealEstateLoansOrLines',
  'NumberOfTime60-89DaysPastDueNotWorse',
  'NumberOfDependents',
] as const;

export type FeatureKey = (typeof FEATURE_KEYS)[number];

/** Request body for POST /predict. Optional fields may be omitted or null. */
export type PredictionRequest = Record<FeatureKey, number | null>;

export interface PredictionResponse {
  default_probability: number;
  predicted_class: 0 | 1;
  predicted_label: string;
  classification_threshold: number;
  risk_band: string;
  risk_band_disclaimer: string;
  model_version: string;
  target_definition: string;
}

export interface ConfusionMatrix {
  true_negatives: number;
  false_positives: number;
  false_negatives: number;
  true_positives: number;
}

export interface MetricsSummary {
  threshold: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  pr_auc: number;
  confusion_matrix: ConfusionMatrix;
}

export interface RiskBandDefinition {
  name: string;
  lower_bound_inclusive: number;
  /** Null for the open-ended top band. */
  upper_bound_exclusive: number | null;
}

export interface ClassDistribution {
  total: number;
  negative: number;
  positive: number;
  positive_rate: number;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

/**
 * Histogram of test-set predicted scores, split by actual class.
 * `bin_edges` has one more entry than either count array.
 */
export interface ScoreDistribution {
  bin_edges: number[];
  negative_counts: number[];
  positive_counts: number[];
}

export interface ThresholdCurvePoint {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  /** Borrowers predicted positive at this cut-off. */
  flagged: number;
  true_positives: number;
  false_positives: number;
  false_negatives: number;
}

export interface ModelInfo {
  model_name: string;
  model_version: string;
  algorithm: string;
  trained_at: string;
  dataset_name: string;
  dataset_url: string;
  target_column: string;
  target_definition: string;
  features: string[];
  class_order: number[];
  classification_threshold: number;
  risk_bands: RiskBandDefinition[];
  risk_band_disclaimer: string;
  test_metrics: MetricsSummary;
  test_threshold_sweep: MetricsSummary[];
  test_threshold_curve: ThresholdCurvePoint[];
  test_score_distribution: ScoreDistribution;
  class_distribution: Record<string, ClassDistribution>;
  feature_importances: FeatureImportance[];
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  model_version: string | null;
  detail: string | null;
}

/** A real held-out dataset row offered as demonstration input. */
export interface DemoExample {
  label: string;
  actual_outcome: 0 | 1;
  features: Record<string, number>;
}
