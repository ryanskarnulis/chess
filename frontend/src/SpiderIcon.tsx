/**
 * The agent's mascot: a little spider hanging from a silk thread, drawn in
 * the night-silk palette (thin pale strokes like the corner-web backdrop,
 * glow-dot eyes). Purely decorative — always pair it with text.
 */
export function SpiderIcon({ className }: { className?: string }) {
  const leg = 'none'
  const stroke = '#c7d3e4'
  return (
    <svg
      className={className ? `spider-icon ${className}` : 'spider-icon'}
      viewBox="0 0 64 80"
      aria-hidden="true"
      focusable="false"
    >
      {/* silk thread */}
      <line x1="32" y1="0" x2="32" y2="30" stroke={stroke} strokeOpacity="0.35" strokeWidth="1" />
      {/* legs: four each side, arcing out from the body */}
      <g fill={leg} stroke={stroke} strokeOpacity="0.8" strokeWidth="1.6" strokeLinecap="round">
        <path d="M26 44 Q14 36 10 24" />
        <path d="M25 48 Q10 44 4 34" />
        <path d="M25 52 Q10 54 5 64" />
        <path d="M27 55 Q18 62 16 72" />
        <path d="M38 44 Q50 36 54 24" />
        <path d="M39 48 Q54 44 60 34" />
        <path d="M39 52 Q54 54 59 64" />
        <path d="M37 55 Q46 62 48 72" />
      </g>
      {/* abdomen + head */}
      <ellipse cx="32" cy="52" rx="10" ry="12" fill="var(--surface, #1c2536)" stroke={stroke} strokeOpacity="0.8" strokeWidth="1.6" />
      <circle cx="32" cy="37" r="7" fill="var(--surface, #1c2536)" stroke={stroke} strokeOpacity="0.8" strokeWidth="1.6" />
      {/* glow eyes */}
      <circle cx="29.5" cy="36" r="1.6" fill="var(--glow, #9be7ff)" />
      <circle cx="34.5" cy="36" r="1.6" fill="var(--glow, #9be7ff)" />
      {/* abdomen web mark */}
      <path d="M32 45 v14 M27 52 h10" stroke={stroke} strokeOpacity="0.3" strokeWidth="1" fill="none" />
    </svg>
  )
}
