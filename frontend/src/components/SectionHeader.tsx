import type { ReactNode } from 'react';

interface SectionHeaderProps {
  title: string;
  subtitle: string;
  actions?: ReactNode;
}

export function SectionHeader({ title, subtitle, actions }: SectionHeaderProps) {
  return (
    <div className="section-header">
      <div>
        <p className="eyebrow">Control Room</p>
        <h2>{title}</h2>
        <p>{subtitle}</p>
      </div>
      {actions && <div className="section-actions">{actions}</div>}
    </div>
  );
}
