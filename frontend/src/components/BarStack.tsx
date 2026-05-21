interface BarStackProps {
  label: string;
  value: number;
  max: number;
  subtitle?: string;
  accent?: 'gold' | 'teal' | 'rose' | 'steel';
}

export function BarStack({ label, value, max, subtitle, accent = 'steel' }: BarStackProps) {
  const percent = max > 0 ? Math.min(100, (value / max) * 100) : 0;

  return (
    <div className="bar-stack">
      <div className="bar-stack-copy">
        <strong>{label}</strong>
        {subtitle && <span>{subtitle}</span>}
      </div>
      <div className="bar-track">
        <div className={`bar-fill accent-${accent}`} style={{ width: `${percent}%` }} />
      </div>
      <span className="bar-value">{value.toFixed(2)}</span>
    </div>
  );
}
