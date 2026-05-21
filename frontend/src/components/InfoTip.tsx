interface InfoTipProps {
  text: string;
}

export function InfoTip({ text }: InfoTipProps) {
  return (
    <span className="info-tip" data-tooltip={text} aria-label={text} tabIndex={0}>
      i
    </span>
  );
}
