// Derived health analytics: scores, normalized radar values, insights.
// All scores are 0–100 where higher = better. These are heuristic wellness
// indicators for demonstration, not clinical scoring.

export const clamp = (v, a, b) => Math.max(a, Math.min(b, v))

// 100 inside [lo,hi]; falls linearly to 0 at the hard limits.
export function bandScore(v, lo, hi, hardLo, hardHi) {
  if (v == null || Number.isNaN(v)) return null
  if (v >= lo && v <= hi) return 100
  if (v < lo) return clamp((100 * (v - hardLo)) / (lo - hardLo), 0, 100)
  return clamp((100 * (hardHi - v)) / (hardHi - hi), 0, 100)
}

// Higher is better, saturating from `min` (=0) to `good` (=100).
export function highScore(v, min, good) {
  if (v == null || Number.isNaN(v)) return null
  return clamp((100 * (v - min)) / (good - min), 0, 100)
}

export function stats(arr) {
  const xs = arr.filter((x) => x != null && !Number.isNaN(x))
  if (!xs.length) return { avg: null, min: null, max: null }
  const sum = xs.reduce((a, b) => a + b, 0)
  return {
    avg: Math.round((sum / xs.length) * 10) / 10,
    min: Math.round(Math.min(...xs) * 10) / 10,
    max: Math.round(Math.max(...xs) * 10) / 10,
  }
}

// Individual wellness dimensions (0–100, higher better).
export function dimensions(m) {
  return {
    cardiovascular: bandScore(m.hr, 60, 100, 40, 160),
    respiratory: bandScore(m.rr, 12, 20, 6, 30),
    recovery: highScore(m.hrv_rmssd, 10, 60),
    variability: highScore(m.hrv_sdnn, 20, 100),
    stress: m.stress_index != null ? clamp(100 - m.stress_index, 0, 100) : null,
  }
}

function avgDefined(vals, weights) {
  let s = 0, w = 0
  vals.forEach((v, i) => { if (v != null) { s += v * weights[i]; w += weights[i] } })
  return w ? s / w : null
}

export function scoreLabel(score) {
  if (score == null) return { label: 'No data', color: 'var(--muted)' }
  if (score >= 85) return { label: 'Excellent', color: 'var(--good)' }
  if (score >= 70) return { label: 'Good', color: 'var(--good)' }
  if (score >= 50) return { label: 'Fair', color: 'var(--warn)' }
  return { label: 'Needs attention', color: 'var(--bad)' }
}

export function wellnessScore(m) {
  const d = dimensions(m)
  const s = avgDefined(
    [d.cardiovascular, d.stress, d.recovery, d.respiratory],
    [0.3, 0.25, 0.25, 0.2],
  )
  const score = s == null ? null : Math.round(s)
  return { score, ...scoreLabel(score) }
}

export function recoveryScore(m) {
  const d = dimensions(m)
  const s = avgDefined([d.recovery, d.variability, d.stress], [0.4, 0.3, 0.3])
  const score = s == null ? null : Math.round(s)
  return { score, ...scoreLabel(score) }
}

export function readinessScore(m) {
  const d = dimensions(m)
  const s = avgDefined([d.recovery, d.stress, d.cardiovascular], [0.4, 0.3, 0.3])
  const score = s == null ? null : Math.round(s)
  return { score, ...scoreLabel(score) }
}

export function radarData(m) {
  const d = dimensions(m)
  const w = wellnessScore(m).score
  const rd = readinessScore(m).score
  return [
    { axis: 'Cardiovascular', value: Math.round(d.cardiovascular ?? 0) },
    { axis: 'Recovery', value: Math.round(d.recovery ?? 0) },
    { axis: 'Stress', value: Math.round(d.stress ?? 0) },
    { axis: 'Respiratory', value: Math.round(d.respiratory ?? 0) },
    { axis: 'Wellness', value: Math.round(w ?? 0) },
    { axis: 'Readiness', value: Math.round(rd ?? 0) },
  ]
}

export function signalQuality(sqi) {
  if (sqi == null) return { label: 'Unknown', pct: null, color: 'var(--muted)' }
  const pct = Math.round(sqi * 100)
  if (sqi >= 0.7) return { label: 'Excellent', pct, color: 'var(--good)' }
  if (sqi >= 0.5) return { label: 'Good', pct, color: 'var(--good)' }
  if (sqi >= 0.3) return { label: 'Fair', pct, color: 'var(--warn)' }
  return { label: 'Low', pct, color: 'var(--bad)' }
}

// Plain-language, jargon-free insights.
export function insights(m) {
  const out = []
  if (m.hr != null) {
    if (m.hr < 60) out.push({ emoji: '❤️', tone: 'warn', text: 'Your heart rate is on the lower side of normal.' })
    else if (m.hr <= 100) out.push({ emoji: '❤️', tone: 'good', text: 'Your heart rate is within the normal range.' })
    else out.push({ emoji: '❤️', tone: 'bad', text: 'Your heart rate is elevated — try to rest and breathe slowly.' })
  }
  if (m.rr != null) {
    if (m.rr < 12) out.push({ emoji: '🌬️', tone: 'warn', text: 'Your breathing rate is slightly lower than average.' })
    else if (m.rr <= 20) out.push({ emoji: '🌬️', tone: 'good', text: 'Your breathing rate looks healthy and steady.' })
    else out.push({ emoji: '🌬️', tone: 'warn', text: 'Your breathing rate is a little faster than average.' })
  }
  if (m.stress_index != null) {
    if (m.stress_index < 40) out.push({ emoji: '😌', tone: 'good', text: 'Your stress level appears low and relaxed.' })
    else if (m.stress_index <= 70) out.push({ emoji: '🙂', tone: 'warn', text: 'Your stress level appears moderate.' })
    else out.push({ emoji: '😣', tone: 'bad', text: 'Your stress level appears high right now.' })
  }
  if (m.hrv_rmssd != null) {
    if (m.hrv_rmssd >= 30) out.push({ emoji: '💪', tone: 'good', text: 'Heart-rhythm variability suggests good recovery.' })
    else out.push({ emoji: '🛌', tone: 'warn', text: 'Heart-rhythm variability suggests recovery could improve with rest.' })
  }
  if (m.sqi != null && m.sqi < 0.5) {
    out.push({ emoji: '📶', tone: 'warn', text: 'Signal quality was modest — better lighting and stillness improve accuracy.' })
  }
  return out
}
