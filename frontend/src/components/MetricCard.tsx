import { InfoTip } from './InfoTip';

interface MetricCardProps {
  label: string;
  value: string;
  hint?: string;
  tooltip?: string;
  accent?: 'gold' | 'teal' | 'rose' | 'steel';
}

export function MetricCard({ label, value, hint, tooltip, accent = 'steel' }: MetricCardProps) {
  return (
    <article className={`metric-card accent-${accent}`}>
      <span className="metric-label field-label">
        {label}
        {tooltip ? <InfoTip text={tooltip} /> : null}
      </span>
      <strong className="metric-value">{value}</strong>
      {hint && <span className="metric-hint">{hint}</span>}
    </article>
  );
}
