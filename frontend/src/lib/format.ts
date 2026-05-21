export function formatMs(value: number): string {
  if (!Number.isFinite(value)) {
    return '-';
  }

  if (value >= 1000) {
    return `${(value / 1000).toFixed(2)} s`;
  }

  return `${Math.round(value)} ms`;
}

export function formatSeconds(value: number): string {
  return `${value.toFixed(1)} s`;
}

export function formatRealtime(value: number): string {
  if (!Number.isFinite(value)) {
    return '-';
  }

  return `${value.toFixed(2)}x`;
}

export function formatPercent(value: number): string {
  return `${Math.round(value)}%`;
}

export function formatDate(value: string): string {
  const date = new Date(value);
  return date.toLocaleString('en-US', {
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  });
}

export function formatBytes(value: number): string {
  if (!Number.isFinite(value)) {
    return '-';
  }

  if (value >= 1024 * 1024) {
    return `${(value / (1024 * 1024)).toFixed(1)} MB`;
  }

  if (value >= 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }

  return `${Math.round(value)} B`;
}

export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
