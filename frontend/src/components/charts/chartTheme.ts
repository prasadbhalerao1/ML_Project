/**
 * Shared constants and geometry helpers for the hand-rolled SVG charts.
 *
 * The project deliberately has no charting library: these charts are two
 * specific forms, and a dependency would cost more than it saves. Everything
 * here is plain arithmetic over a fixed viewBox, scaled responsively by CSS.
 *
 * The categorical colours were validated for colour-vision deficiency and for
 * contrast against a white surface. Indigo + amber alone: worst adjacent pair
 * ΔE 34.0 protan / 39.8 normal. Adding teal (the three-line threshold chart)
 * lowers that to ΔE 12.5 protan / 24.3 normal, still above the target of 8, and
 * those lines are also labelled directly. All steps are ≥ 3:1 contrast. They
 * are Tailwind palette steps already used elsewhere in the app, so the charts
 * match the surrounding UI.
 */

/** Categorical series colours, assigned in fixed order — never cycled. */
export const SERIES = {
  /** indigo-600 — the app's primary accent. */
  indigo: '#4f46e5',
  /** amber-600. */
  amber: '#d97706',
  /** teal-600. */
  teal: '#0d9488',
} as const;

/** Recessive ink for grid, axes and labels, so the data stays dominant. */
export const INK = {
  grid: '#e2e8f0', // slate-200
  axis: '#cbd5e1', // slate-300
  label: '#64748b', // slate-500
  strong: '#334155', // slate-700
  marker: '#0f172a', // slate-900
} as const;

/** Internal coordinate system. CSS scales the rendered size from this. */
export interface PlotBox {
  width: number;
  height: number;
  margin: { top: number; right: number; bottom: number; left: number };
}

/** Below this container width a chart switches to its compact layout. */
export const NARROW_BREAKPOINT = 520;

/** Narrowest width a chart is laid out for; anything smaller scrolls the page, not the chart. */
const MIN_WIDTH = 220;

/**
 * Plot geometry for a container of the given pixel width.
 *
 * Charts are drawn at their real size (viewBox width equals the rendered
 * width) so a 12px label stays 12px. Scaling one fixed 720-unit drawing down to
 * a phone would shrink the text to about 5px and make the axes unreadable.
 */
export function boxFor(containerWidth: number): PlotBox {
  const width = Math.max(MIN_WIDTH, Math.round(containerWidth));
  const narrow = width < NARROW_BREAKPOINT;
  return {
    width,
    height: narrow ? 270 : 300,
    // The left margin holds the tick labels plus the rotated axis title, so
    // it is the same on both layouts; only the right and top edges give way.
    margin: narrow
      ? { top: 12, right: 12, bottom: 46, left: 56 }
      : { top: 14, right: 18, bottom: 46, left: 56 },
  };
}

/** Fewer ticks on a narrow chart, so neighbouring labels never collide. */
export const tickCountFor = (box: PlotBox, wide: number, narrow: number) =>
  box.width < NARROW_BREAKPOINT ? narrow : wide;

/** Used only for the first paint, before the container has been measured. */
export const DEFAULT_BOX: PlotBox = boxFor(720);

export const plotArea = (box: PlotBox) => ({
  left: box.margin.left,
  top: box.margin.top,
  right: box.width - box.margin.right,
  bottom: box.height - box.margin.bottom,
  width: box.width - box.margin.left - box.margin.right,
  height: box.height - box.margin.top - box.margin.bottom,
});

/** Maps a data value onto a pixel position. */
export const scaleLinear = (domain: [number, number], range: [number, number]) => {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  return (value: number) => r0 + ((value - d0) / span) * (r1 - r0);
};

/** Evenly spaced tick values across a domain, inclusive of both ends. */
export const ticks = (from: number, to: number, count: number): number[] =>
  Array.from({ length: count + 1 }, (_, i) => from + ((to - from) * i) / count);

export const asPercent = (value: number, digits = 0) => `${(value * 100).toFixed(digits)}%`;
