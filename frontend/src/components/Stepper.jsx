// Simple 3-step progress indicator: Intake → Measure → Results.
const STEPS = ['Patient info', 'Measurement', 'Results']

export default function Stepper({ current }) {
  return (
    <div className="stepper" role="group" aria-label={`Step ${current + 1} of ${STEPS.length}`}>
      {STEPS.map((label, i) => (
        <div key={label} style={{ display: 'contents' }}>
          <div
            className={`dot ${i < current ? 'done' : ''} ${i === current ? 'active' : ''}`}
            aria-current={i === current ? 'step' : undefined}
          >
            {i < current ? '✓' : i + 1}
          </div>
          <span className="label">{label}</span>
          {i < STEPS.length - 1 && <span className={`bar ${i < current ? 'done' : ''}`} />}
        </div>
      ))}
    </div>
  )
}
