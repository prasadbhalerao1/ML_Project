/**
 * The only module that talks to the FastAPI backend. Components call these
 * functions and never build URLs or parse responses themselves.
 */

import type {
  DemoExample,
  HealthStatus,
  ModelInfo,
  PredictionRequest,
  PredictionResponse,
} from '../types/api';

// trim() also strips a stray byte-order mark, which Windows tooling can
// prepend to a value piped into an environment variable and which would
// otherwise end up inside every request URL.
const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000')
  .trim()
  .replace(/\/$/, '');

/** An API failure with a message fit to show a user. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export const getHealth = () => request<HealthStatus>('/health');
export const getModelInfo = () => request<ModelInfo>('/model-info');
export const getDemoExamples = () => request<DemoExample[]>('/demo-examples');

export const postPrediction = (body: PredictionRequest) =>
  request<PredictionResponse>('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, init);
  } catch {
    throw new ApiError(
      `Could not reach the API at ${BASE_URL}. Is the FastAPI server running?`,
    );
  }

  if (!response.ok) {
    throw new ApiError(await readErrorMessage(response), response.status);
  }
  return (await response.json()) as T;
}

/**
 * Turn an error body into one readable sentence. FastAPI sends a 422 as a list
 * of per-field entries and other errors as a single `detail` string.
 */
async function readErrorMessage(response: Response): Promise<string> {
  const fallback = `Request failed with status ${response.status}.`;
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    return fallback;
  }

  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail === 'string') return detail;

  if (Array.isArray(detail)) {
    const issues = detail.map((entry) => {
      const { loc, msg } = entry as { loc?: unknown[]; msg?: string };
      const field = Array.isArray(loc) ? loc[loc.length - 1] : undefined;
      return field ? `${String(field)}: ${msg ?? 'invalid value'}` : String(msg);
    });
    return `Validation failed. ${issues.join('; ')}`;
  }
  return fallback;
}
