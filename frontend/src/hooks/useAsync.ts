/** Runs an async loader once on mount and tracks loading/error/data state. */

import { useCallback, useEffect, useState } from 'react';
import { ApiError } from '../services/api';

export interface AsyncState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  reload: () => void;
}

export function useAsync<T>(loader: () => Promise<T>): AsyncState<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);

    loader()
      .then((result) => {
        if (active) setData(result);
      })
      .catch((caught: unknown) => {
        if (active) setError(describeError(caught));
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
    // `loader` is recreated on every render by callers, so the attempt counter
    // is what deliberately re-runs the effect.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [attempt]);

  const reload = useCallback(() => setAttempt((value) => value + 1), []);

  return { data, loading, error, reload };
}

/** Turn an unknown thrown value into a message safe to render. */
export function describeError(caught: unknown): string {
  if (caught instanceof ApiError) return caught.message;
  if (caught instanceof Error) return caught.message;
  return 'Something went wrong.';
}
