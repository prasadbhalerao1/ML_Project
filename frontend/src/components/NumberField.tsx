/** A labelled numeric input with inline help and validation messaging. */

import type { FeatureField } from '../services/featureFields';

export function NumberField({
  field,
  value,
  error,
  onChange,
}: {
  field: FeatureField;
  value: string;
  error?: string;
  onChange: (value: string) => void;
}) {
  const inputId = `field-${field.key}`;
  const helpId = field.help ? `${inputId}-help` : undefined;
  const errorId = error ? `${inputId}-error` : undefined;

  return (
    <div>
      <label
        htmlFor={inputId}
        className="flex flex-wrap items-baseline gap-x-2 text-sm font-medium text-slate-800"
      >
        {field.label}
        {field.optional ? (
          <span className="text-xs font-normal text-slate-500">optional</span>
        ) : (
          <span className="text-xs font-normal text-red-600" aria-hidden="true">
            required
          </span>
        )}
        {field.unit && (
          <span className="text-xs font-normal text-slate-400">
            ({field.unit})
          </span>
        )}
      </label>

      <input
        id={inputId}
        type="number"
        inputMode="decimal"
        value={value}
        min={field.min}
        max={field.max}
        step={field.step}
        required={!field.optional}
        aria-required={!field.optional}
        aria-invalid={error ? true : undefined}
        aria-describedby={[helpId, errorId].filter(Boolean).join(' ') || undefined}
        onChange={(event) => onChange(event.target.value)}
        className={`mt-1.5 w-full rounded-lg border px-3 py-2 text-sm text-slate-900 shadow-sm outline-none transition focus:ring-2 ${
          error
            ? 'border-red-400 focus:border-red-500 focus:ring-red-200'
            : 'border-slate-300 focus:border-indigo-500 focus:ring-indigo-200'
        }`}
      />

      {field.help && (
        <p id={helpId} className="mt-1.5 text-xs leading-relaxed text-slate-500">
          {field.help}
        </p>
      )}
      {error && (
        <p id={errorId} role="alert" className="mt-1.5 text-xs font-medium text-red-700">
          {error}
        </p>
      )}
    </div>
  );
}
