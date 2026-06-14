import { createContext, useContext, useEffect, useState } from 'react'

// Global patient + results state, persisted to localStorage so an accidental
// refresh during intake doesn't lose data.
const PatientContext = createContext(null)

const FORM_KEY = 'vitals.patient.form'

const EMPTY_FORM = {
  gender: '',
  weight: '',       // kg
  height: '',       // cm
  birthYear: '',
  bpCategory: '',   // normal | prehypertension | hypertension
}

function loadForm() {
  try {
    const raw = localStorage.getItem(FORM_KEY)
    return raw ? { ...EMPTY_FORM, ...JSON.parse(raw) } : { ...EMPTY_FORM }
  } catch {
    return { ...EMPTY_FORM }
  }
}

export function PatientProvider({ children }) {
  const [form, setForm] = useState(loadForm)
  const [results, setResults] = useState(null)

  // Autosave form to localStorage.
  useEffect(() => {
    try { localStorage.setItem(FORM_KEY, JSON.stringify(form)) } catch { /* noop */ }
  }, [form])

  const updateForm = (patch) => setForm((f) => ({ ...f, ...patch }))
  const resetForm = () => setForm({ ...EMPTY_FORM })

  const age = form.birthYear
    ? new Date().getFullYear() - Number(form.birthYear)
    : null

  const value = { form, updateForm, resetForm, age, results, setResults }
  return <PatientContext.Provider value={value}>{children}</PatientContext.Provider>
}

export function usePatient() {
  const ctx = useContext(PatientContext)
  if (!ctx) throw new Error('usePatient must be used within PatientProvider')
  return ctx
}

export { EMPTY_FORM }
