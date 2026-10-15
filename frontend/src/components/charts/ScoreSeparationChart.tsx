/**
 * How the model's predicted scores are distributed for each actual class.
 *
 * Why this chart exists: the confusion matrix reports errors at one cut-off,
 * but it cannot show *why* those errors are unavoidable. This can. The two
 * distributions overlap heavily in the middle, so no threshold separates them
 * cleanly — which is the honest explanation for the recall ceiling, and the
 * reason the decision threshold is a trade-off rather than a solved value.
 *
 * Each class is drawn as a share of its own total, not as a raw count.
 * Negatives outnumber positives roughly 14:1, so raw counts would flatten the
 * positive series into the axis and show nothing.
 */

import { useState } from 'react';
import type { ScoreDistribution } from '../../types/api';
import {
  INK, SERIES, asPercent, plotArea, scaleLinear, tickCountFor, ticks,
} from './chartTheme';
import { ChartFrame, useChartBox, type TooltipState } from './ChartFrame';

interface Props {
  distribution: ScoreDistribution;
  /** Decision threshold, drawn as a rule so the split point is visible. */
  threshold: number;
  /** Optional: one borrower's score, for the result page. */
  marker?: { probability: number; label: string };
}

export function ScoreSeparationChart({ distribution, threshold, marker }: Props) {
  const [hover, setHover] = useState<number | null>(null);

  const { bin_edges: edges, negative_counts: negatives, positive_counts: positives } = distribution;
  const negativeTotal = negatives.reduce((a, b) => a + b, 0) || 1;
  const positiveTotal = positives.reduce((a, b) => a + b, 0) || 1;

  // Share of each class per bin, so the two shapes are comparable.
  const negativeShare = negatives.map((c) => c / negativeTotal);
  const positiveShare = positives.map((c) => c / positiveTotal);
  const peak = Math.max(...negativeShare, ...positiveShare);

  const { ref, box } = useChartBox();
  const area = plotArea(box);
  const x = scaleLinear([0, 1], [area.left, area.right]);
  const y = scaleLinear([0, peak], [area.bottom, area.top]);

  /** Step outline across the bins, closed to the baseline so it can be filled. */
  const stepPath = (shares: number[]) => {
    const parts: string[] = [`M ${x(edges[0] ?? 0)} ${area.bottom}`];
    shares.forEach((share, i) => {
      const left = x(edges[i] ?? 0);
      const right = x(edges[i + 1] ?? 1);
      parts.push(`L ${left} ${y(share)}`, `L ${right} ${y(share)}`);
    });
    parts.push(`L ${x(edges[edges.length - 1] ?? 1)} ${area.bottom} Z`);
    return parts.join(' ');
  };

  const binFromX = (pixelX: number) => {
    const value = (pixelX - area.left) / area.width;
    if (value < 0 || value > 1) return null;
    const index = Math.min(negativeShare.length - 1, Math.floor(value * negativeShare.length));
    return index >= 0 ? index : null;
  };

  let tooltip: TooltipState | null = null;
  if (hover !== null) {
    const from = edges[hover] ?? 0;
    const to = edges[hover + 1] ?? 1;
    tooltip = {
      x: x((from + to) / 2),
      y: area.top + area.height * 0.3,
      content: (
        <div className="space-y-1">
          <p className="font-medium text-slate-900">
            Score {asPercent(from)}–{asPercent(to)}
          </p>
          <p className="flex items-center justify-between gap-3 text-slate-600">
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2 w-2 rounded-sm" style={{ background: SERIES.indigo }} />
              No delinquency
            </span>
            <span className="tabular-nums">{(negatives[hover] ?? 0).toLocaleString()}</span>
          </p>
          <p className="flex items-center justify-between gap-3 text-slate-600">
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2 w-2 rounded-sm" style={{ background: SERIES.amber }} />
              Delinquency
            </span>
            <span className="tabular-nums">{(positives[hover] ?? 0).toLocaleString()}</span>
          </p>
        </div>
      ),
    };
  }

  const thresholdX = x(threshold);
  const markerX = marker ? x(Math.min(1, Math.max(0, marker.probability))) : null;

  return (
    <ChartFrame
      title={
        `Distribution of predicted scores by actual outcome, with the decision threshold at ` +
        `${asPercent(threshold, 1)}. The two classes overlap substantially.`
      }
      xLabel="Predicted probability of serious delinquency"
      yLabel="Share of that class"
      box={box}
      containerRef={ref}
      xTicks={ticks(0, 1, tickCountFor(box, 5, 4)).map((t) => ({ value: x(t), label: asPercent(t) }))}
      yTicks={ticks(0, peak, 4).map((t) => ({ value: t, label: asPercent(t, 0) }))}
      legend={[
        { label: 'Actually no serious delinquency', colour: SERIES.indigo },
        { label: 'Actually serious delinquency', colour: SERIES.amber },
        { label: `Decision threshold (${asPercent(threshold, 1)})`, colour: INK.marker, dashed: true },
        ...(marker ? [{ label: marker.label, colour: SERIES.teal, dashed: true }] : []),
      ]}
      tooltip={tooltip}
      onPointerMove={(pixelX) => setHover(binFromX(pixelX))}
      onPointerLeave={() => setHover(null)}
    >
      <path d={stepPath(negativeShare)} fill={SERIES.indigo} fillOpacity={0.45} stroke={SERIES.indigo} strokeWidth={2} strokeLinejoin="round" />
      <path d={stepPath(positiveShare)} fill={SERIES.amber} fillOpacity={0.45} stroke={SERIES.amber} strokeWidth={2} strokeLinejoin="round" />

      {/* Decision threshold. Everything right of this line is predicted positive. */}
      <line
        x1={thresholdX} x2={thresholdX} y1={area.top} y2={area.bottom}
        stroke={INK.marker} strokeWidth={2} strokeDasharray="5 4"
      />
      <text x={thresholdX + 6} y={area.top + 12} fontSize={11} fill={INK.strong}>
        predicted positive →
      </text>

      {markerX !== null && marker && (
        <>
          <line
            x1={markerX} x2={markerX} y1={area.top} y2={area.bottom}
            stroke={SERIES.teal} strokeWidth={2.5} strokeDasharray="2 3"
          />
          <circle cx={markerX} cy={area.top + 4} r={4.5} fill={SERIES.teal} stroke="#ffffff" strokeWidth={2} />
        </>
      )}

      {hover !== null && (
        <rect
          x={x(edges[hover] ?? 0)}
          y={area.top}
          width={Math.max(1, x(edges[hover + 1] ?? 1) - x(edges[hover] ?? 0))}
          height={area.height}
          fill={INK.strong}
          fillOpacity={0.07}
        />
      )}
    </ChartFrame>
  );
}
