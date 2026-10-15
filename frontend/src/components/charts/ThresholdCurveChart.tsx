/**
 * Precision, recall and F1 across every decision threshold.
 *
 * Why this chart exists: the summary table lists four thresholds, which is
 * enough to state the trade-off but not to show its *shape* — where precision
 * and recall cross, how sharply recall falls away, and that F1 has a broad
 * flat peak rather than one magic value. Seeing the peak is what makes the
 * chosen threshold look like a decision instead of an arbitrary constant.
 */

import { useState } from 'react';
import type { ThresholdCurvePoint } from '../../types/api';
import {
  INK, SERIES, asPercent, plotArea, scaleLinear, tickCountFor, ticks,
} from './chartTheme';
import { ChartFrame, useChartBox, type TooltipState } from './ChartFrame';

const LINES = [
  { key: 'precision', label: 'Precision', colour: SERIES.amber },
  { key: 'recall', label: 'Recall', colour: SERIES.indigo },
  { key: 'f1', label: 'F1', colour: SERIES.teal },
] as const;

export function ThresholdCurveChart({
  curve,
  threshold,
}: {
  curve: ThresholdCurvePoint[];
  threshold: number;
}) {
  const [hover, setHover] = useState<number | null>(null);

  const { ref, box } = useChartBox();
  const area = plotArea(box);
  const from = curve[0]?.threshold ?? 0;
  const to = curve[curve.length - 1]?.threshold ?? 1;
  const x = scaleLinear([from, to], [area.left, area.right]);
  const y = scaleLinear([0, 1], [area.bottom, area.top]);

  type LineKey = (typeof LINES)[number]['key'];

  const linePath = (key: LineKey) =>
    curve
      .map((point, i) => `${i === 0 ? 'M' : 'L'} ${x(point.threshold)} ${y(point[key])}`)
      .join(' ');

  const nearestIndex = (pixelX: number) => {
    if (pixelX < area.left - 4 || pixelX > area.right + 4) return null;
    let best = 0;
    let bestDistance = Infinity;
    curve.forEach((point, i) => {
      const distance = Math.abs(x(point.threshold) - pixelX);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = i;
      }
    });
    return best;
  };

  const active = hover !== null ? curve[hover] : undefined;
  // Total positives is constant across the curve: caught plus missed.
  const totalPositives = curve[0]
    ? curve[0].true_positives + curve[0].false_negatives
    : 0;

  const tooltip: TooltipState | null = active
    ? {
        x: x(active.threshold),
        y: area.top + area.height * 0.25,
        content: (
          <div className="space-y-1.5">
            <p className="font-medium text-slate-900">
              If the cut-off were {active.threshold.toFixed(2)}
            </p>
            {/* The point of hovering: turn the rates into people. */}
            <p className="text-slate-700">
              Catches{' '}
              <span className="font-semibold tabular-nums">
                {active.true_positives.toLocaleString()}
              </span>{' '}
              of {totalPositives.toLocaleString()} delinquencies
            </p>
            <p className="text-slate-700">
              Wrongly flags{' '}
              <span className="font-semibold tabular-nums">
                {active.false_positives.toLocaleString()}
              </span>{' '}
              sound borrowers
            </p>
            <div className="mt-1 border-t border-slate-100 pt-1.5">
              {LINES.map((line) => (
                <p key={line.key} className="flex items-center justify-between gap-3 text-slate-600">
                  <span className="flex items-center gap-1.5">
                    <span className="inline-block h-2 w-2 rounded-sm" style={{ background: line.colour }} />
                    {line.label}
                  </span>
                  <span className="tabular-nums">{active[line.key].toFixed(3)}</span>
                </p>
              ))}
            </div>
          </div>
        ),
      }
    : null;

  const thresholdX = x(threshold);

  /*
   * Direct labels sit at the left edge, where the three series are furthest
   * apart. They converge toward the right, and they cross in the middle, so
   * neither of those is a safe anchor. F1 is a harmonic mean of the other two,
   * so it always lies between them — the lowest of the three therefore gets
   * its label below the line and the rest above, which keeps all three clear
   * of each other whatever the data does.
   */
  const labelPoint = curve[0];
  const lowestKey = labelPoint
    ? LINES.reduce((low, line) =>
        labelPoint[line.key] < labelPoint[low.key] ? line : low,
      ).key
    : null;

  return (
    <ChartFrame
      title={
        'Precision, recall and F1 plotted against the decision threshold. ' +
        `The threshold in use is ${threshold.toFixed(4)}, near the F1 peak.`
      }
      xLabel="Decision threshold"
      yLabel="Score"
      box={box}
      containerRef={ref}
      xTicks={ticks(from, to, tickCountFor(box, 6, 3)).map((t) => ({ value: x(t), label: t.toFixed(2) }))}
      yTicks={ticks(0, 1, 4).map((t) => ({ value: t, label: asPercent(t, 0) }))}
      legend={[
        ...LINES.map((line) => ({ label: line.label, colour: line.colour })),
        { label: `Threshold in use (${threshold.toFixed(2)})`, colour: INK.marker, dashed: true },
      ]}
      tooltip={tooltip}
      onPointerMove={(pixelX) => setHover(nearestIndex(pixelX))}
      onPointerLeave={() => setHover(null)}
    >
      <line
        x1={thresholdX} x2={thresholdX} y1={area.top} y2={area.bottom}
        stroke={INK.marker} strokeWidth={2} strokeDasharray="5 4"
      />

      {LINES.map((line) => (
        <path
          key={line.key}
          d={linePath(line.key)}
          fill="none"
          stroke={line.colour}
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      ))}

      {/* Direct labels so identity is never colour alone. Anchored near the
          left, where the three series are well separated — at the right edge
          they all converge toward zero and the labels would overlap. */}
      {labelPoint &&
        LINES.map((line) => (
          <text
            key={`${line.key}-label`}
            x={x(labelPoint.threshold) + 7}
            y={y(labelPoint[line.key]) + (line.key === lowestKey ? 15 : -8)}
            fontSize={11}
            fontWeight={600}
            fill={line.colour}
            stroke="#ffffff"
            strokeWidth={3}
            paintOrder="stroke"
          >
            {line.label}
          </text>
        ))}

      {active && (
        <>
          <line
            x1={x(active.threshold)} x2={x(active.threshold)} y1={area.top} y2={area.bottom}
            stroke={INK.strong} strokeWidth={1} strokeOpacity={0.35}
          />
          {LINES.map((line) => (
            <circle
              key={`${line.key}-dot`}
              cx={x(active.threshold)}
              cy={y(active[line.key])}
              r={4.5}
              fill={line.colour}
              stroke="#ffffff"
              strokeWidth={2}
            />
          ))}
        </>
      )}
    </ChartFrame>
  );
}
