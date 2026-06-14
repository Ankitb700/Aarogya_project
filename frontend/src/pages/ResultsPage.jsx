import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar, Cell,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine,
} from 'recharts'
import Stepper from '../components/Stepper.jsx'
import ScoreGauge from '../components/ScoreGauge.jsx'
import { usePatient } from '../state/PatientContext.jsx'
import {
  stats, wellnessScore, recoveryScore, readinessScore, radarData,
  signalQuality, insights, dimensions,
} from '../lib/health.js'

const BP_LABELS = {
  normal: { text: 'Normal', range: '< 120 mmHg', tag: 'good', emoji: '🟢' },
  prehypertension: { text: 'Prehypertension', range: '120–139 mmHg', tag: 'warn', emoji: '🟡' },
  hypertension: { text: 'Hypertension', range: '≥ 140 mmHg', tag: 'bad', emoji: '🔴' },
}

function hrTag(hr) {
  if (hr == null) return null
  if (hr < 60) return { t: 'Low', c: 'warn' }
  if (hr <= 100) return { t: 'Normal', c: 'good' }
  return { t: 'Elevated', c: 'bad' }
}
function rrTag(rr) {
  if (rr == null) return null
  if (rr < 12) return { t: 'Low', c: 'warn' }
  if (rr <= 20) return { t: 'Normal', c: 'good' }
  return { t: 'High', c: 'bad' }
}
function stressTag(s) {
  if (s == null) return null
  if (s < 40) return { t: 'Calm', c: 'good', emoji: '😌' }
  if (s <= 70) return { t: 'Moderate', c: 'warn', emoji: '🙂' }
  return { t: 'High', c: 'bad', emoji: '😣' }
}

function Kpi({ emoji, label, value, unit, tag }) {
  return (
    <div className="kpi card">
      <div className="kpi-top">
        <span className="kpi-emoji" aria-hidden="true">{emoji}</span>
        {tag && <span className={`tag ${tag.c}`}>{tag.t}</span>}
      </div>
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">
        {value ?? '—'}{value != null && unit && <span className="kpi-unit">{unit}</span>}
      </div>
    </div>
  )
}

function StatTile({ label, value, unit }) {
  return (
    <div className="stat-tile">
      <div className="stat-val">{value ?? '—'}{value != null && unit ? <small> {unit}</small> : ''}</div>
      <div className="stat-key">{label}</div>
    </div>
  )
}

const tip = {
  contentStyle: { background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 10 },
}

// Dev-only fixture so the dashboard can be previewed without a live scan
// (open /results?demo=1 in `npm run dev`). Never used in production builds.
function demoResults() {
  const series = Array.from({ length: 26 }, (_, i) => {
    const t = i + 5
    return {
      t,
      hr: Math.round(72 + 6 * Math.sin(i / 3) + (i % 5)),
      rr: Math.round(15 + 2 * Math.sin(i / 4)),
      rmssd: Math.round(34 + 6 * Math.sin(i / 5)),
      sdnn: Math.round(52 + 5 * Math.cos(i / 6)),
      stress: Math.round(45 + 12 * Math.sin(i / 4)),
      sqi: Math.round(70 + 12 * Math.sin(i / 7)),
    }
  })
  return {
    metrics: { hr: 74, rr: 15.2, hrv_rmssd: 36, hrv_sdnn: 53, stress_index: 47, sqi: 0.78, beats: 37, elapsed: 30.2, samples: 533 },
    series,
    patient: { gender: 'Male', age: 24, bpCategory: 'normal' },
    finishedAt: new Date().toISOString(),
  }
}

export default function ResultsPage() {
  const navigate = useNavigate()
  const { results: ctxResults, resetForm } = usePatient()

  const demo = import.meta.env.DEV &&
    typeof window !== 'undefined' &&
    new URLSearchParams(window.location.search).has('demo')
  const results = ctxResults || (demo ? demoResults() : null)

  useEffect(() => {
    if (!results) navigate('/', { replace: true })
  }, [results, navigate])

  if (!results) return null

  const m = results.metrics || {}
  const series = results.series || []
  const p = results.patient || {}
  const bp = BP_LABELS[p.bpCategory] || null
  const noData = m.hr == null

  const hrStats = stats(series.map((s) => s.hr))
  const rrStats = stats(series.map((s) => s.rr))
  const rmssdStats = stats(series.map((s) => s.rmssd))

  // KPI cards show the scan average (steadier than the single final reading);
  // fall back to the final value when no trend series is available.
  const hrDisplay = hrStats.avg ?? m.hr
  const rrDisplay = rrStats.avg ?? m.rr

  const wellness = wellnessScore(m)
  const recovery = recoveryScore(m)
  const readiness = readinessScore(m)
  const dim = dimensions(m)
  const radar = radarData(m)
  const sq = signalQuality(m.sqi)
  const tips = insights(m)
  const st = stressTag(m.stress_index)
  const stressEmoji = st?.emoji || '😐'

  const hasRrSeries = series.some((s) => s.rr != null)
  const hasRmssdSeries = series.some((s) => s.rmssd != null)
  const hasSqiSeries = series.some((s) => s.sqi != null)

  const hrvData = [
    { name: 'RMSSD', value: m.hrv_rmssd ?? 0, fill: 'var(--blue)' },
    { name: 'SDNN', value: m.hrv_sdnn ?? 0, fill: 'var(--teal)' },
  ]

  return (
    <div className="results-dash">
      <Stepper current={2} />

      {/* Phase 2 — Assessment summary header */}
      <section className="summary-hero card">
        <div className="sh-main">
          <span className="tag good" style={{ marginBottom: 8 }}>✓ Assessment complete</span>
          <h1>Your health assessment</h1>
          <p className="muted">
            {p.gender || '—'} • {p.age != null ? `${p.age} years` : '—'}
          </p>
        </div>
        <div className="sh-meta">
          <StatTile label="Scan duration" value={m.elapsed} unit="sec" />
          <StatTile label="Frames processed" value={m.samples} />
          <StatTile label="Signal quality" value={sq.label} />
          <StatTile label="Completed" value={results.finishedAt ? new Date(results.finishedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '—'} />
        </div>
      </section>

      {noData && (
        <div className="notice" role="alert" style={{ background: 'var(--warn-bg)', borderColor: '#fde68a', color: '#92400e' }}>
          <span className="ic" aria-hidden="true">⚠️</span>
          <div>We couldn't extract a reliable pulse signal. Ensure your face is well-lit, centered and still, then re-measure.</div>
        </div>
      )}

      {/* Phase 1 — KPI row (4 columns on desktop). HR/RR show the scan
          average (more stable than the single final reading). */}
      <section className="kpi-row">
        <Kpi emoji="❤️" label="Heart Rate" value={hrDisplay} unit="bpm" tag={hrTag(hrDisplay)} />
        <Kpi emoji="🌬️" label="Respiratory Rate" value={rrDisplay} unit="bpm" tag={rrTag(rrDisplay)} />
        <Kpi emoji={stressEmoji} label="Stress Index" value={m.stress_index} unit="/100" tag={st} />
        <Kpi emoji="🩺" label="Blood Pressure" value={bp ? `${bp.emoji} ${bp.text}` : null} unit={bp ? '' : 'mmHg'} tag={bp ? { t: bp.range, c: bp.tag } : null} />
      </section>

      {/* Phase 3 + 4 — Wellness score + insights */}
      <section className="panel-2">
        <div className="card pad score-card">
          <h3>Overall Wellness Score</h3>
          <ScoreGauge score={wellness.score} label={wellness.label} color={wellness.color} caption="Composite of heart, stress, recovery & breathing" />
        </div>
        <div className="card pad">
          <h3>Health Insights</h3>
          <ul className="insights">
            {tips.length === 0 && <li className="muted">No insights available — try a cleaner measurement.</li>}
            {tips.map((t, i) => (
              <li key={i} className={`insight ${t.tone}`}>
                <span className="ins-emoji" aria-hidden="true">{t.emoji}</span>
                <span>{t.text}</span>
              </li>
            ))}
          </ul>
        </div>
      </section>

      {/* Phase 6 — HR & RR trends */}
      <section className="panel-2">
        <div className="card pad">
          <div className="chart-title">❤️ Heart-rate trend</div>
          <div className="chart-stats">
            <span>Avg <b>{hrStats.avg ?? '—'}</b></span>
            <span>Min <b>{hrStats.min ?? '—'}</b></span>
            <span>Max <b>{hrStats.max ?? '—'}</b> <small>bpm</small></span>
          </div>
          <ResponsiveContainer width="100%" height={210}>
            <LineChart data={series} margin={{ top: 6, right: 14, left: -12, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="t" unit="s" stroke="var(--muted)" fontSize={12} />
              <YAxis domain={[40, 140]} stroke="var(--muted)" fontSize={12} />
              <Tooltip {...tip} formatter={(v) => [`${v} bpm`, 'HR']} labelFormatter={(t) => `${t}s`} />
              {hrStats.avg != null && <ReferenceLine y={hrStats.avg} stroke="var(--teal)" strokeDasharray="4 4" />}
              <Line type="monotone" dataKey="hr" stroke="var(--blue)" strokeWidth={2.5} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="card pad">
          <div className="chart-title">🌬️ Respiratory trend</div>
          <div className="chart-stats">
            <span>Avg <b>{rrStats.avg ?? '—'}</b></span>
            <span>Min <b>{rrStats.min ?? '—'}</b></span>
            <span>Max <b>{rrStats.max ?? '—'}</b> <small>bpm</small></span>
          </div>
          {hasRrSeries ? (
            <ResponsiveContainer width="100%" height={210}>
              <AreaChart data={series} margin={{ top: 6, right: 14, left: -12, bottom: 0 }}>
                <defs>
                  <linearGradient id="rrFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--teal)" stopOpacity={0.45} />
                    <stop offset="100%" stopColor="var(--teal)" stopOpacity={0.04} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="t" unit="s" stroke="var(--muted)" fontSize={12} />
                <YAxis domain={[0, 30]} stroke="var(--muted)" fontSize={12} />
                <Tooltip {...tip} formatter={(v) => [`${v} bpm`, 'RR']} labelFormatter={(t) => `${t}s`} />
                <Area type="monotone" dataKey="rr" stroke="var(--teal)" strokeWidth={2} fill="url(#rrFill)" isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          ) : <div className="empty">Not enough breathing data captured.</div>}
        </div>
      </section>

      {/* HRV evolution + Recovery analysis */}
      <section className="panel-2">
        <div className="card pad">
          <div className="chart-title">💓 HRV trend (RMSSD evolution)</div>
          {hasRmssdSeries ? (
            <ResponsiveContainer width="100%" height={210}>
              <LineChart data={series} margin={{ top: 6, right: 14, left: -12, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="t" unit="s" stroke="var(--muted)" fontSize={12} />
                <YAxis stroke="var(--muted)" fontSize={12} unit="ms" />
                <Tooltip {...tip} formatter={(v) => [`${v} ms`, 'RMSSD']} labelFormatter={(t) => `${t}s`} />
                <Line type="monotone" dataKey="rmssd" stroke="var(--blue)" strokeWidth={2.5} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : <div className="empty">Not enough variability data captured.</div>}
        </div>

        <div className="card pad recovery-card">
          <h3>Recovery analysis</h3>
          <div className="recovery-row">
            <ScoreGauge score={recovery.score} label={recovery.label} color={recovery.color} size={150} />
            <div className="recovery-side">
              <div className="mini-bar"><span>Readiness</span><i style={{ width: `${readiness.score ?? 0}%`, background: readiness.color }} /><b>{readiness.score ?? '—'}</b></div>
              <div className="mini-bar"><span>Recovery</span><i style={{ width: `${dim.recovery ?? 0}%`, background: 'var(--blue)' }} /><b>{Math.round(dim.recovery ?? 0)}</b></div>
              <div className="mini-bar"><span>Calmness</span><i style={{ width: `${dim.stress ?? 0}%`, background: 'var(--teal)' }} /><b>{Math.round(dim.stress ?? 0)}</b></div>
            </div>
          </div>
        </div>
      </section>

      {/* Phase 7 — Radar + Signal quality */}
      <section className="panel-2">
        <div className="card pad">
          <h3>Health radar</h3>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radar} outerRadius="72%">
              <PolarGrid stroke="var(--border)" />
              <PolarAngleAxis dataKey="axis" tick={{ fill: 'var(--muted)', fontSize: 12 }} />
              <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
              <Radar dataKey="value" stroke="var(--blue)" fill="var(--blue)" fillOpacity={0.3} isAnimationActive={false} />
              <Tooltip {...tip} formatter={(v) => [`${v}/100`, 'Score']} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        <div className="card pad">
          <div className="chart-title">📶 Signal quality during scan</div>
          {hasSqiSeries ? (
            <ResponsiveContainer width="100%" height={210}>
              <AreaChart data={series} margin={{ top: 6, right: 14, left: -12, bottom: 0 }}>
                <defs>
                  <linearGradient id="sqiFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={sq.color} stopOpacity={0.45} />
                    <stop offset="100%" stopColor={sq.color} stopOpacity={0.04} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="t" unit="s" stroke="var(--muted)" fontSize={12} />
                <YAxis domain={[0, 100]} stroke="var(--muted)" fontSize={12} unit="%" />
                <Tooltip {...tip} formatter={(v) => [`${v}%`, 'Confidence']} labelFormatter={(t) => `${t}s`} />
                <Area type="monotone" dataKey="sqi" stroke={sq.color} strokeWidth={2} fill="url(#sqiFill)" isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          ) : <div className="empty">Signal confidence: <b>{sq.label}</b>{sq.pct != null ? ` (${sq.pct}%)` : ''}</div>}
        </div>
      </section>

      {/* Phase 5 — Advanced HRV analytics */}
      <section className="card pad">
        <h3>Advanced HRV analytics</h3>
        <div className="panel-2" style={{ marginBottom: 0 }}>
          <div>
            <div className="adv-metric">
              <div>
                <div className="adv-name">RMSSD <a href="https://en.wikipedia.org/wiki/Heart_rate_variability" target="_blank" rel="noreferrer">Learn more ↗</a></div>
                <div className="adv-desc">Measures short-term heart-rhythm variability. Typical resting range ≈ 20–90 ms.</div>
              </div>
              <div className="adv-val">{m.hrv_rmssd ?? '—'}<small> ms</small></div>
            </div>
            <div className="adv-metric">
              <div>
                <div className="adv-name">SDNN <a href="https://en.wikipedia.org/wiki/Heart_rate_variability" target="_blank" rel="noreferrer">Learn more ↗</a></div>
                <div className="adv-desc">Measures overall heart-rhythm variability. Typical short-recording range ≈ 30–100 ms.</div>
              </div>
              <div className="adv-val">{m.hrv_sdnn ?? '—'}<small> ms</small></div>
            </div>
            <div className="stat-row">
              <StatTile label="RMSSD avg" value={rmssdStats.avg} unit="ms" />
              <StatTile label="RMSSD min" value={rmssdStats.min} unit="ms" />
              <StatTile label="RMSSD max" value={rmssdStats.max} unit="ms" />
            </div>
          </div>
          <div>
            <div className="chart-title">HRV comparison</div>
            <ResponsiveContainer width="100%" height={210}>
              <BarChart data={hrvData} margin={{ top: 6, right: 14, left: -12, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="name" stroke="var(--muted)" fontSize={12} />
                <YAxis stroke="var(--muted)" fontSize={12} unit="ms" />
                <Tooltip {...tip} formatter={(v) => [`${v} ms`, 'Value']} />
                <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                  {hrvData.map((d) => <Cell key={d.name} fill={d.fill} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      {/* Phase 8 — Vitals summary */}
      <section className="panel-2">
        <div className="card pad">
          <h3>Vitals summary</h3>
          <div className="summary-table">
            <div className="st-head"><span>Metric</span><span>Avg</span><span>Min</span><span>Max</span></div>
            <div className="st-row"><span>❤️ Heart Rate <small>bpm</small></span><span>{hrStats.avg ?? '—'}</span><span>{hrStats.min ?? '—'}</span><span>{hrStats.max ?? '—'}</span></div>
            <div className="st-row"><span>🌬️ Respiratory <small>bpm</small></span><span>{rrStats.avg ?? '—'}</span><span>{rrStats.min ?? '—'}</span><span>{rrStats.max ?? '—'}</span></div>
            <div className="st-row"><span>💓 RMSSD <small>ms</small></span><span>{rmssdStats.avg ?? '—'}</span><span>{rmssdStats.min ?? '—'}</span><span>{rmssdStats.max ?? '—'}</span></div>
          </div>
        </div>
      </section>

      <div className="disclaimer" role="note">
        Signal quality (SQI): {sq.pct != null ? `${sq.pct}% — ${sq.label}` : 'n/a'}. These figures
        are camera-based estimates for wellness and demonstration only and must not be used for
        diagnosis or treatment decisions.
      </div>

      <div className="form-actions" style={{ marginTop: 18 }}>
        <button className="btn btn-ghost" onClick={() => { resetForm(); navigate('/') }}>Start over</button>
        <button className="btn btn-primary" onClick={() => navigate('/measure')}>Measure again →</button>
      </div>
    </div>
  )
}
