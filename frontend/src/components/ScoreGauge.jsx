// Prominent circular score gauge (0–100) with a big centered number + label.
export default function ScoreGauge({ score, label, color = 'var(--blue)', size = 200, caption }) {
  const stroke = 16
  const r = (size - stroke) / 2
  const c = 2 * Math.PI * r
  const has = score != null && !Number.isNaN(score)
  const pct = has ? Math.max(0, Math.min(1, score / 100)) : 0
  return (
    <div className="score-gauge" role="img" aria-label={`${caption || 'Score'}: ${has ? score + ' out of 100, ' + label : 'no data'}`}>
      <svg width={size} height={size}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--border)" strokeWidth={stroke} />
        <circle
          cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth={stroke}
          strokeLinecap="round" strokeDasharray={c} strokeDashoffset={c * (1 - pct)}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{ transition: 'stroke-dashoffset .8s ease' }}
        />
        <text x="50%" y="46%" textAnchor="middle" fontSize={size * 0.26} fontWeight="800" fill="var(--text)">
          {has ? score : '—'}
        </text>
        <text x="50%" y="60%" textAnchor="middle" fontSize={size * 0.08} fill="var(--muted)">/ 100</text>
      </svg>
      <div className="score-label" style={{ color }}>{label}</div>
      {caption && <div className="score-caption muted">{caption}</div>}
    </div>
  )
}
