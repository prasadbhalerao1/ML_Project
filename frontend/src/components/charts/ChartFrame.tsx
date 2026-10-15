/**
 * The furniture every chart here shares: a responsive SVG, axes, grid, legend
 * and a hover tooltip positioned over the plot.
 *
 * Keeping it in one place means the two chart components contain only the
 * marks that make them different.
 */

import {
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
  type Ref,
} from 'react';
import { DEFAULT_BOX, INK, boxFor, plotArea, type PlotBox } from './chartTheme';

export interface LegendEntry {
  label: string;
  colour: string;
  /** Rendered as a dashed rule rather than a filled swatch. */
  dashed?: boolean;
}

export interface AxisTick {
  value: number;
  label: string;
}

export interface TooltipState {
  /** Position within the SVG coordinate system. */
  x: number;
  y: number;
  content: ReactNode;
}

interface ChartFrameProps {
  /** Describes the chart for screen readers; the surrounding text carries detail. */
  title: string;
  xLabel: string;
  yLabel: string;
  xTicks: AxisTick[];
  yTicks: AxisTick[];
  legend: LegendEntry[];
  box?: PlotBox;
  /** Attach to the element whose width the chart should fill (see `useChartBox`). */
  containerRef?: Ref<HTMLDivElement>;
  tooltip: TooltipState | null;
  /** Marks drawn inside the plot area, in SVG coordinates. */
  children: ReactNode;
  onPointerMove?: (xInPlot: number) => void;
  onPointerLeave?: () => void;
}

export function ChartFrame({
  title, xLabel, yLabel, xTicks, yTicks, legend,
  box = DEFAULT_BOX, containerRef, tooltip, children, onPointerMove, onPointerLeave,
}: ChartFrameProps) {
  const area = plotArea(box);

  /** Convert a pointer event into SVG coordinates and hand it to the chart. */
  const track = (event: React.PointerEvent<SVGSVGElement>) => {
    if (!onPointerMove) return;
    const rect = event.currentTarget.getBoundingClientRect();
    onPointerMove((event.clientX - rect.left) * (box.width / rect.width));
  };

  return (
    <figure className="m-0">
      <Legend entries={legend} />
      <div className="relative" ref={containerRef}>
        <svg
          viewBox={`0 0 ${box.width} ${box.height}`}
          role="img"
          aria-label={title}
          className="block w-full"
          // A finger drag across the chart inspects it; a vertical drag still
          // scrolls the page, so the chart never traps scrolling on a phone.
          style={{ height: 'auto', touchAction: 'pan-y' }}
          onPointerMove={track}
          onPointerDown={track}
          onPointerLeave={onPointerLeave}
        >
          <title>{title}</title>

          {/* Horizontal gridlines only: vertical ones compete with the marks. */}
          {yTicks.map((tick, i) => {
            const y = area.bottom - (area.height * i) / (yTicks.length - 1);
            return (
              <line
                key={tick.label}
                x1={area.left} x2={area.right} y1={y} y2={y}
                stroke={INK.grid} strokeWidth={1}
              />
            );
          })}

          {children}

          {/* Axes drawn after the marks so they sit cleanly on top. */}
          <line x1={area.left} x2={area.right} y1={area.bottom} y2={area.bottom} stroke={INK.axis} strokeWidth={1} />
          <line x1={area.left} x2={area.left} y1={area.top} y2={area.bottom} stroke={INK.axis} strokeWidth={1} />

          {yTicks.map((tick, i) => {
            const y = area.bottom - (area.height * i) / (yTicks.length - 1);
            return (
              <text
                key={tick.label} x={area.left - 10} y={y + 4}
                textAnchor="end" fontSize={12} fill={INK.label}
              >
                {tick.label}
              </text>
            );
          })}

          {xTicks.map((tick) => (
            <text
              key={tick.label} x={tick.value} y={area.bottom + 20}
              textAnchor="middle" fontSize={12} fill={INK.label}
            >
              {tick.label}
            </text>
          ))}

          <text
            x={area.left + area.width / 2} y={box.height - 6}
            textAnchor="middle" fontSize={12} fill={INK.label}
          >
            {xLabel}
          </text>
          <text
            x={14} y={area.top + area.height / 2}
            textAnchor="middle" fontSize={12} fill={INK.label}
            transform={`rotate(-90 14 ${area.top + area.height / 2})`}
          >
            {yLabel}
          </text>
        </svg>

        {tooltip && <Tooltip state={tooltip} box={box} />}
      </div>
    </figure>
  );
}

function Legend({ entries }: { entries: LegendEntry[] }) {
  return (
    <div className="mb-2 flex flex-wrap items-center gap-x-5 gap-y-1.5">
      {entries.map((entry) => (
        <span key={entry.label} className="flex items-center gap-2 text-xs text-slate-600">
          {entry.dashed ? (
            <span
              aria-hidden="true"
              className="inline-block h-0 w-4 border-t-2 border-dashed"
              style={{ borderColor: entry.colour }}
            />
          ) : (
            <span
              aria-hidden="true"
              className="inline-block h-2.5 w-2.5 rounded-sm"
              style={{ backgroundColor: entry.colour }}
            />
          )}
          {entry.label}
        </span>
      ))}
    </div>
  );
}

function Tooltip({ state, box }: { state: TooltipState; box: PlotBox }) {
  // Flip to the left of the cursor near the right edge so it never clips.
  const nearRight = state.x > box.width * 0.62;
  return (
    <div
      role="status"
      className="pointer-events-none absolute z-10 min-w-[9rem] rounded-md border border-slate-200 bg-white/97 px-2.5 py-2 text-xs shadow-md"
      style={{
        left: `${(state.x / box.width) * 100}%`,
        top: `${(state.y / box.height) * 100}%`,
        transform: `translate(${nearRight ? 'calc(-100% - 10px)' : '10px'}, -50%)`,
      }}
    >
      {state.content}
    </div>
  );
}

/**
 * Measures the container and returns plot geometry for that width.
 *
 * `ref` goes on the container (passed to `ChartFrame` as `containerRef`). The
 * measurement happens before first paint and again whenever the container
 * resizes, so charts re-flow on rotation and window changes.
 */
export function useChartBox() {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(DEFAULT_BOX.width);

  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;
    const update = () => setWidth(element.clientWidth);
    update();
    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => observer.disconnect();
  }, [setWidth]);

  return { ref, box: boxFor(width) };
}
