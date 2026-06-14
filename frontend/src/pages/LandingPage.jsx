import { useNavigate } from 'react-router-dom'
import { usePatient } from '../state/PatientContext.jsx'

const FEATURES = [
  { ic: '📷', title: 'Contactless Measurement', desc: 'No wearables or cuffs — just your camera.' },
  { ic: '❤️', title: 'Heart Rate Analysis', desc: 'Beat-to-beat pulse from facial blood flow.' },
  { ic: '🩺', title: 'Blood Pressure Context', desc: 'Self-reported BP category factored into your report.' },
  { ic: '🫁', title: 'Respiratory Monitoring', desc: 'Breathing rate derived from the pulse signal.' },
]

const STEPS = [
  { n: 1, t: 'Tell us about you', d: 'A short intake form (≈30s).' },
  { n: 2, t: 'Face the camera', d: 'Hold still in good light.' },
  { n: 3, t: '30-second scan', d: 'We read your pulse signal.' },
  { n: 4, t: 'Get your report', d: 'Clear, dashboard-style results.' },
]

function HeroArt() {
  return (
    <svg viewBox="0 0 400 300" role="img" aria-label="Illustration of contactless vital sign measurement">
      <defs>
        <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#2563eb" />
          <stop offset="1" stopColor="#0ea5a4" />
        </linearGradient>
      </defs>
      <rect width="400" height="300" fill="url(#bg)" />
      <circle cx="200" cy="120" r="58" fill="#ffffff" opacity="0.95" />
      <circle cx="183" cy="112" r="6" fill="#0f172a" />
      <circle cx="217" cy="112" r="6" fill="#0f172a" />
      <path d="M182 138 q18 16 36 0" stroke="#0f172a" strokeWidth="4" fill="none" strokeLinecap="round" />
      <rect x="120" y="60" width="160" height="120" rx="14" fill="none" stroke="#ffffff" strokeWidth="3" strokeDasharray="10 8" opacity="0.85" />
      <path d="M40 235 h70 l14 -34 l20 64 l18 -84 l16 54 h150"
        fill="none" stroke="#ffffff" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
      <text x="200" y="285" textAnchor="middle" fill="#ffffff" fontSize="16" fontWeight="700" opacity="0.9">rPPG signal</text>
    </svg>
  )
}

export default function LandingPage() {
  const navigate = useNavigate()
  const { resetForm } = usePatient()
  // Start a brand-new assessment with a blank form (clears any saved data).
  const startAssessment = () => { resetForm(); navigate('/intake') }
  return (
    <div>
      <section className="hero">
        <div>
          <span className="eyebrow">Camera-based health screening</span>
          <h1>Check your vitals in 30 seconds — no contact required.</h1>
          <p className="lead">
            VitalScan estimates your heart rate, respiration and heart-rate variability
            straight from your camera using remote photoplethysmography (rPPG).
          </p>
          <div className="benefits">
            <span className="pill">✓ No wearables</span>
            <span className="pill">✓ ~30s scan</span>
            <span className="pill">✓ Private — processed locally</span>
          </div>
          <button className="btn btn-primary" onClick={startAssessment}>
            Start Health Assessment →
          </button>
        </div>
        <div className="hero-art"><HeroArt /></div>
      </section>

      <h2>What you get</h2>
      <div className="feature-grid">
        {FEATURES.map((f) => (
          <div className="feature" key={f.title}>
            <div className="ic" aria-hidden="true">{f.ic}</div>
            <h3>{f.title}</h3>
            <p>{f.desc}</p>
          </div>
        ))}
      </div>

      <h2>How it works</h2>
      <div className="how">
        {STEPS.map((s) => (
          <div className="step" key={s.n}>
            <div className="num" aria-hidden="true">{s.n}</div>
            <h3 style={{ margin: '0 0 4px', fontSize: 15 }}>{s.t}</h3>
            <p className="muted" style={{ margin: 0, fontSize: 14 }}>{s.d}</p>
          </div>
        ))}
      </div>

      <div className="notice" role="note">
        <span className="ic" aria-hidden="true">🔒</span>
        <div>
          <strong>Your privacy matters.</strong> Video frames are streamed only to your
          own analysis server to compute the pulse signal and are not stored. Results are
          estimates for wellness and demonstration, not a medical diagnosis.
        </div>
      </div>

      <button className="btn btn-primary btn-block" onClick={startAssessment}>
        Start Health Assessment →
      </button>
    </div>
  )
}
