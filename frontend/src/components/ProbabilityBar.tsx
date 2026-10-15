/**
 * Accessible bar for a predicted probability, with the configured risk-band
 * boundaries marked so the reader can see where the band label comes from.
 */

import type { RiskBandDefinition } from '../types/api';
import { formatPercent } from './ui';

const BAND_COLORS: Record<string, string> = {
  Low: 'bg-emerald-500',
  Medium: 'bg-amber-500',
  High: 'bg-red-500',
};

export function ProbabilityBar({
  probability,
  band,
  bands = [],
}: {
  probability: number;
  band: string;
  bands?: RiskBandDefinition[];
}) {
  const percent = Math.min(100, Math.max(0, probability * 100));
  const boundaries = bands
    .map((entry) => entry.lower_bound_inclusive)
    .filter((value) => value > 0);

  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="text-sm font-medium text-slate-600">
          Estimated probability of serious delinquency
        </span>
        <span className="text-2xl font-semibold tabular-nums text-slate-900">
          {formatPercent(probability)}
        </span>
      </div>

      <div
        role="progressbar"
        aria-valuenow={Number(percent.toFixed(1))}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`Estimated probability of serious delinquency: ${formatPercent(probability)}`}
        className="relative mt-2 h-4 w-full overflow-hidden rounded-full bg-slate-200"
      >
        <div
          className={`h-full rounded-full transition-[width] duration-500 ${BAND_COLORS[band] ?? 'bg-indigo-500'}`}
          style={{ width: `${percent}%` }}
        />
        {boundaries.map((boundary) => (
          <span
            key={boundary}
            aria-hidden="true"
            title={`Band boundary at ${formatPercent(boundary, 0)}`}
            className="absolute top-0 h-full w-px bg-slate-500/60"
            style={{ left: `${boundary * 100}%` }}
          />
        ))}
      </div>

      <div className="mt-1 flex justify-between text-xs text-slate-500">
        <span>0%</span>
        <span>100%</span>
      </div>
    </div>
  );
}
