import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Stepper from '../components/Stepper.jsx'
import { usePatient } from '../state/PatientContext.jsx'

const CURRENT_YEAR = new Date().getFullYear()
const MIN_YEAR = CURRENT_YEAR - 120
const MAX_YEAR = CURRENT_YEAR - 5

const BP_OPTIONS = [
  { value: 'normal', title: 'Normal', desc: 'SBP < 120 mmHg' },
  { value: 'prehypertension', title: 'Prehypertension', desc: 'SBP 120–139 mmHg' },
  { value: 'hypertension', title: 'Hypertension', desc: 'SBP ≥ 140 mmHg' },
]

function validate(form) {
  const e = {}
  if (!form.gender) e.gender = 'Please select a gender.'

  const w = Number(form.weight)
  if (!form.weight) e.weight = 'Weight is required.'
  else if (Number.isNaN(w) || w < 20 || w > 250) e.weight = 'Enter a weight between 20 and 250 kg.'

  const h = Number(form.height)
  if (!form.height) e.height = 'Height is required.'
  else if (Number.isNaN(h) || h < 100 || h > 250) e.height = 'Enter a height between 100 and 250 cm.'

  const y = Number(form.birthYear)
  if (!form.birthYear) e.birthYear = 'Birth year is required.'
  else if (Number.isNaN(y) || y < MIN_YEAR || y > MAX_YEAR) e.birthYear = `Enter a year between ${MIN_YEAR} and ${MAX_YEAR}.`

  if (!form.bpCategory) e.bpCategory = 'Please select your usual blood pressure category.'
  return e
}

export default function IntakeForm() {
  const navigate = useNavigate()
  const { form, updateForm, age } = usePatient()
  const [touched, setTouched] = useState({})
  const [submitAttempted, setSubmitAttempted] = useState(false)

  const errors = validate(form)
  const showErr = (k) => (touched[k] || submitAttempted) && errors[k]

  const onBlur = (k) => setTouched((t) => ({ ...t, [k]: true }))

  const submit = (e) => {
    e.preventDefault()
    setSubmitAttempted(true)
    if (Object.keys(errors).length === 0) navigate('/measure')
  }

  return (
    <form className="form-card card" onSubmit={submit} noValidate>
      <Stepper current={0} />
      <h1 style={{ marginTop: 0 }}>Patient information</h1>
      <p className="muted" style={{ marginTop: -6 }}>
        Step 1 of 2 — this helps us contextualize your results. Your data stays on this device.
      </p>

      {/* Gender */}
      <fieldset className="field" style={{ border: 0, padding: 0, margin: '0 0 22px' }}>
        <legend className="q">Gender</legend>
        <div className="radio-grid cols">
          {['Male', 'Female', 'Other'].map((g) => (
            <label key={g} className={`radio-card ${form.gender === g ? 'sel' : ''}`}>
              <input
                type="radio" name="gender" value={g}
                checked={form.gender === g}
                onChange={() => { updateForm({ gender: g }); onBlur('gender') }}
              />
              <span className="rc-title">{g}</span>
            </label>
          ))}
        </div>
        {showErr('gender') && <div className="err" role="alert">{errors.gender}</div>}
      </fieldset>

      {/* Weight */}
      <div className="field">
        <label className="q" htmlFor="weight">Weight</label>
        <div className="input-row">
          <input
            id="weight" type="number" inputMode="decimal" min={20} max={250}
            className={`input ${showErr('weight') ? 'invalid' : ''}`}
            value={form.weight}
            onChange={(e) => updateForm({ weight: e.target.value })}
            onBlur={() => onBlur('weight')}
            aria-describedby="weight-err"
          />
          <span className="suffix">kg</span>
        </div>
        {showErr('weight') && <div className="err" id="weight-err" role="alert">{errors.weight}</div>}
      </div>

      {/* Height */}
      <div className="field">
        <label className="q" htmlFor="height">Height</label>
        <div className="input-row">
          <input
            id="height" type="number" inputMode="decimal" min={100} max={250}
            className={`input ${showErr('height') ? 'invalid' : ''}`}
            value={form.height}
            onChange={(e) => updateForm({ height: e.target.value })}
            onBlur={() => onBlur('height')}
            aria-describedby="height-err"
          />
          <span className="suffix">cm</span>
        </div>
        {showErr('height') && <div className="err" id="height-err" role="alert">{errors.height}</div>}
      </div>

      {/* Birth year */}
      <div className="field">
        <label className="q" htmlFor="birthYear">Birth year</label>
        <input
          id="birthYear" type="number" inputMode="numeric" min={MIN_YEAR} max={MAX_YEAR}
          placeholder={`e.g. ${CURRENT_YEAR - 30}`}
          className={`input ${showErr('birthYear') ? 'invalid' : ''}`}
          value={form.birthYear}
          onChange={(e) => updateForm({ birthYear: e.target.value })}
          onBlur={() => onBlur('birthYear')}
          aria-describedby="byear-err"
        />
        {age != null && !errors.birthYear && (
          <div className="age-chip" aria-live="polite">Age: {age} years</div>
        )}
        {showErr('birthYear') && <div className="err" id="byear-err" role="alert">{errors.birthYear}</div>}
      </div>

      {/* BP category */}
      <fieldset className="field" style={{ border: 0, padding: 0, margin: '0 0 22px' }}>
        <legend className="q">Usual blood pressure category</legend>
        <p className="help">If unsure, choose Normal. This is only used to contextualize results.</p>
        <div className="radio-grid">
          {BP_OPTIONS.map((o) => (
            <label key={o.value} className={`radio-card ${form.bpCategory === o.value ? 'sel' : ''}`}>
              <input
                type="radio" name="bp" value={o.value}
                checked={form.bpCategory === o.value}
                onChange={() => { updateForm({ bpCategory: o.value }); onBlur('bpCategory') }}
              />
              <span>
                <span className="rc-title">{o.title}</span>
                <span className="rc-desc"> — {o.desc}</span>
              </span>
            </label>
          ))}
        </div>
        {showErr('bpCategory') && <div className="err" role="alert">{errors.bpCategory}</div>}
      </fieldset>

      <div className="form-actions">
        <button type="button" className="btn btn-ghost" onClick={() => navigate('/')}>← Back</button>
        <button type="submit" className="btn btn-primary">Continue to Measurement →</button>
      </div>
    </form>
  )
}
