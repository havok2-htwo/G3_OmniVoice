function scaleMaxFor(values: number[]) {
  const peak = Math.max(1, ...values.map((value) => Math.max(0, value)));
  const magnitude = 10 ** Math.floor(Math.log10(peak));
  const normalized = peak / magnitude;

  if (normalized <= 1) return 1 * magnitude;
  if (normalized <= 2) return 2 * magnitude;
  if (normalized <= 5) return 5 * magnitude;
  return 10 * magnitude;
}

function buildPath(values: number[], width: number, height: number, scaleMax: number) {
  if (values.length === 0) {
    return '';
  }

  return values
    .map((value, index) => {
      const x = values.length <= 1 ? 0 : (index / (values.length - 1)) * width;
      const y = height - (Math.max(0, value) / Math.max(1, scaleMax)) * height;
      return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(' ');
}

interface SparklineProps {
  title: string;
  subtitle: string;
  primaryLabel: string;
  primaryValues: number[];
  primaryColor: string;
  primaryFormatter?: (value: number) => string;
  secondaryLabel?: string;
  secondaryValues?: number[];
  secondaryColor?: string;
  secondaryFormatter?: (value: number) => string;
}

export function Sparkline({
  title,
  subtitle,
  primaryLabel,
  primaryValues,
  primaryColor,
  primaryFormatter = (value) => value.toFixed(2),
  secondaryLabel,
  secondaryValues = [],
  secondaryColor = '#ffc16c',
  secondaryFormatter = (value) => value.toFixed(0),
}: SparklineProps) {
  const values = primaryValues.slice(-120);
  const secondary = secondaryValues.slice(-120);
  const primaryScale = scaleMaxFor(values);
  const secondaryScale = scaleMaxFor(secondary);
  const width = 220;
  const height = 118;
  const primaryPath = buildPath(values, width, height, primaryScale);
  const secondaryPath = buildPath(secondary, width, height, secondaryScale);
  const latestPrimary = values.at(-1) ?? 0;
  const latestSecondary = secondary.at(-1) ?? 0;

  return (
    <article className="sparkline-card">
      <div className="sparkline-head">
        <div>
          <span className="sparkline-title">{title}</span>
          <strong>{subtitle}</strong>
        </div>
        <div className="sparkline-stats">
          <span className="sparkline-stat">
            <span className="sparkline-dot" style={{ backgroundColor: primaryColor }} />
            {primaryLabel}: {primaryFormatter(latestPrimary)}
          </span>
          {secondaryLabel ? (
            <span className="sparkline-stat">
              <span className="sparkline-dot" style={{ backgroundColor: secondaryColor }} />
              {secondaryLabel}: {secondaryFormatter(latestSecondary)}
            </span>
          ) : null}
        </div>
      </div>

      <svg className="sparkline-graph" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-label={title}>
        <line x1="0" y1={height - 1} x2={width} y2={height - 1} className="sparkline-axis" />
        <line x1="0" y1={height * 0.5} x2={width} y2={height * 0.5} className="sparkline-grid-line" />
        <line x1="0" y1={height * 0.25} x2={width} y2={height * 0.25} className="sparkline-grid-line" />
        <line x1="0" y1={height * 0.75} x2={width} y2={height * 0.75} className="sparkline-grid-line" />
        {secondaryPath ? (
          <path
            d={secondaryPath}
            fill="none"
            stroke={secondaryColor}
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            opacity="0.6"
          />
        ) : null}
        {primaryPath ? (
          <path
            d={primaryPath}
            fill="none"
            stroke={primaryColor}
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        ) : null}
      </svg>
    </article>
  );
}
