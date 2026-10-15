/**
 * Small presentational building blocks shared by the pages. Grouped in one
 * module because each is a handful of lines and they are always imported
 * together.
 */

import type { ReactNode } from 'react';

export function Card({
  title,
  description,
  children,
  className = '',
}: {
  title?: string;
  description?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6 ${className}`}
    >
      {title && (
        <h2 className="text-base font-semibold text-slate-900">{title}</h2>
      )}
      {description && (
        <p className="mt-1 text-sm text-slate-600">{description}</p>
      )}
      <div className={title || description ? 'mt-4' : ''}>{children}</div>
    </section>
  );
}

export function Spinner({ label = 'Loading…' }: { label?: string }) {
  return (
    <div
      role="status"
      aria-live="polite"
      className="flex items-center gap-3 text-sm text-slate-600"
    >
      <span
        aria-hidden="true"
        className="h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-indigo-600"
      />
      {label}
    </div>
  );
}

export function Alert({
  tone = 'error',
  title,
  children,
  onRetry,
}: {
  tone?: 'error' | 'warning' | 'info';
  title?: string;
  children: ReactNode;
  onRetry?: () => void;
}) {
  const tones = {
    error: 'border-red-200 bg-red-50 text-red-900',
    warning: 'border-amber-200 bg-amber-50 text-amber-900',
    info: 'border-sky-200 bg-sky-50 text-sky-900',
  } as const;

  return (
    <div
      role={tone === 'error' ? 'alert' : 'note'}
      className={`rounded-lg border p-4 text-sm ${tones[tone]}`}
    >
      {title && <p className="font-semibold">{title}</p>}
      <div className={title ? 'mt-1' : ''}>{children}</div>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-3 rounded-md border border-current px-3 py-1.5 text-xs font-medium hover:bg-white/60"
        >
          Try again
        </button>
      )}
    </div>
  );
}

export function MetricTile({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
      <dt className="text-xs font-medium uppercase tracking-wide text-slate-500">
        {label}
      </dt>
      <dd className="mt-1 text-2xl font-semibold tabular-nums text-slate-900">
        {value}
      </dd>
      {hint && <p className="mt-1 text-xs text-slate-500">{hint}</p>}
    </div>
  );
}

/** A labelled definition row, used for model and dataset facts. */
export function DetailRow({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <div className="grid gap-1 border-b border-slate-100 py-3 last:border-0 sm:grid-cols-3 sm:gap-4">
      <dt className="text-sm font-medium text-slate-500">{label}</dt>
      <dd className="min-w-0 text-sm text-slate-900 sm:col-span-2">{children}</dd>
    </div>
  );
}

export function KaggleLink({ href }: { href: string }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer noopener"
      className="break-all font-medium text-indigo-700 underline decoration-indigo-300 underline-offset-2 hover:decoration-indigo-600"
    >
      {href}
    </a>
  );
}

export function PageHeader({
  title,
  children,
}: {
  title: string;
  children?: ReactNode;
}) {
  return (
    <header>
      <h1 className="text-2xl font-semibold tracking-tight text-slate-900 sm:text-3xl">
        {title}
      </h1>
      {children && (
        <p className="mt-2 max-w-3xl text-sm text-slate-600">{children}</p>
      )}
    </header>
  );
}

export const formatPercent = (value: number, digits = 1) =>
  `${(value * 100).toFixed(digits)}%`;

export const formatMetric = (value: number) => value.toFixed(4);
