import { Link, Route, Routes } from 'react-router-dom'
import LandingPage from './pages/LandingPage.jsx'
import IntakeForm from './pages/IntakeForm.jsx'
import CameraPage from './pages/CameraPage.jsx'
import ResultsPage from './pages/ResultsPage.jsx'

// App shell: top bar + routed pages + footer.
export default function App() {
  return (
    <div className="shell">
      <header className="topbar">
        <Link to="/" className="brand" aria-label="VitalScan home">
          <span className="logo" aria-hidden="true">✚</span>
          <span>
            VitalScan
            <small>Contactless vitals assessment</small>
          </span>
        </Link>
      </header>

      <main className="page">
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/intake" element={<IntakeForm />} />
          <Route path="/measure" element={<CameraPage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="*" element={<LandingPage />} />
        </Routes>
      </main>

      <footer className="appfooter">
        ⚠️ For wellness &amp; demonstration only — these are estimates, not medical
        measurements. Consult a clinician for diagnosis.
      </footer>
    </div>
  )
}
