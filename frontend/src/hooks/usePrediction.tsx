/**
 * Holds the most recent prediction so the form page and the result page can
 * share it without either one owning the other's state.
 */

import {
  createContext,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from 'react';
import type { PredictionRequest, PredictionResponse } from '../types/api';

export interface PredictionRecord {
  request: PredictionRequest;
  response: PredictionResponse;
}

interface PredictionContextValue {
  record: PredictionRecord | null;
  setRecord: (record: PredictionRecord) => void;
}

const PredictionContext = createContext<PredictionContextValue | null>(null);

export function PredictionProvider({ children }: { children: ReactNode }) {
  const [record, setRecord] = useState<PredictionRecord | null>(null);
  const value = useMemo(() => ({ record, setRecord }), [record]);
  return (
    <PredictionContext.Provider value={value}>
      {children}
    </PredictionContext.Provider>
  );
}

export function usePrediction(): PredictionContextValue {
  const context = useContext(PredictionContext);
  if (!context) {
    throw new Error('usePrediction must be used inside a PredictionProvider.');
  }
  return context;
}
