// Circular gauge: maps a value within [min,max] to an arc.
export default function ProgressRing({ value, min = 0, max = 100, unit = '', label, color = 'var(--blue)', size = 104 }) {
  const r = (size - 14) / 2
  const c = 2 * Math.PI * r
  const has = value != null && !Number.isNaN(value)
  const pct = has ? Math.min(1, Math.max(0, (value - min) / (max - min))) : 0
  return (
    <div className="ring-wrap">
      <svg width={size} height={size} role="img" aria-label={`${label}: ${has ? value + ' ' + unit : 'no data'}`}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--border)" strokeWidth="9" />
        <circle
          cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth="9"
          strokeLinecap="round" strokeDasharray={c} strokeDashoffset={c * (1 - pct)}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
        <text x="50%" y="48%" textAnchor="middle" fontSize="22" fontWeight="800" fill="var(--text)">
          {has ? value : '—'}
        </text>
        <text x="50%" y="64%" textAnchor="middle" fontSize="11" fill="var(--muted)">{unit}</text>
      </svg>
    </div>
  )
}
