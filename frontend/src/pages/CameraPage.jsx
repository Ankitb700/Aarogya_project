import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Stepper from '../components/Stepper.jsx'
import { AnalyzeSocket } from '../lib/ws.js'
import { FrameCapturer } from '../lib/capture.js'
import { usePatient } from '../state/PatientContext.jsx'

const SCAN_SECONDS = 30

function StatusPill({ ok, pending, warn, label }) {
  const cls = ok ? 'ok' : warn ? 'warn' : pending ? 'pending' : 'pending'
  const mark = ok ? '✓' : warn ? '!' : '…'
  return (
    <div className={`spill ${cls}`}>
      <span className="ic" aria-hidden="true">{mark}</span>
      <span>{label}</span>
    </div>
  )
}

export default function CameraPage() {
  const navigate = useNavigate()
  const { form, age, setResults } = usePatient()

  const videoRef = useRef(null)
  const sockRef = useRef(null)
  const capRef = useRef(null)
  const autostopRef = useRef(null)

  const [phase, setPhase] = useState('idle') // idle|connecting|running|done|error
  const [status, setStatus] = useState('Press Start when you are ready.')
  const [elapsed, setElapsed] = useState(0)
  const [faceDetected, setFaceDetected] = useState(false)
  const [metrics, setMetrics] = useState({})
  const [series, setSeries] = useState([])

  // Redirect to intake if the form wasn't completed.
  useEffect(() => {
    if (!form.gender || !form.weight || !form.height || !form.birthYear || !form.bpCategory) {
      navigate('/intake', { replace: true })
    }
  }, [form, navigate])

  const cleanup = useCallback(() => {
    if (autostopRef.current) clearTimeout(autostopRef.current)
    autostopRef.current = null
    capRef.current?.stop()
    sockRef.current?.close()
    capRef.current = null
    sockRef.current = null
  }, [])

  useEffect(() => () => cleanup(), [cleanup])

  const finish = useCallback((finalMsg) => {
    // Prefer the backend-computed windowed trend (proper time-series over the
    // full 30s signal); fall back to the client-collected samples if absent.
    const trend = Array.isArray(finalMsg.trend) && finalMsg.trend.length
      ? finalMsg.trend
      : series
    setResults({
      metrics: finalMsg, series: trend, patient: { ...form, age },
      finishedAt: new Date().toISOString(),
    })
    cleanup()
    navigate('/results')
  }, [series, form, age, setResults, cleanup, navigate])

  const handleMessage = useCallback((msg) => {
    if (msg.type === 'status') {
      setStatus(msg.message)
      if (['busy', 'warming', 'error'].includes(msg.state)) {
        setPhase('error'); cleanup()
      }
    } else if (msg.type === 'metrics') {
      setElapsed(msg.elapsed)
      setFaceDetected(!!msg.face_detected)
      setMetrics(msg)
      setStatus(msg.face_detected ? 'Measuring — keep still…' : 'Center your face and hold still.')
      if (msg.hr != null) {
        setSeries((s) => [...s.slice(-120), {
          t: Math.round(msg.elapsed),
          hr: msg.hr,
          rr: msg.rr ?? null,
          rmssd: msg.hrv_rmssd ?? null,
          sdnn: msg.hrv_sdnn ?? null,
          stress: msg.stress_index ?? null,
          sqi: msg.sqi != null ? Math.round(msg.sqi * 100) : null,
        }])
      }
    } else if (msg.type === 'final') {
      setMetrics(msg)
      setPhase('done')
      finish(msg)
    }
  }, [cleanup, finish])

  const stop = useCallback(() => {
    capRef.current?.stop()
    sockRef.current?.stop() // request final summary from server
  }, [])

  const start = useCallback(() => {
    setPhase('connecting'); setStatus('Connecting…')
    setMetrics({}); setSeries([]); setElapsed(0)

    const sock = new AnalyzeSocket({
      onOpen: async () => {
        try {
          const cap = new FrameCapturer({ fps: 18 })
          capRef.current = cap
          await cap.start(videoRef.current, (blob) => sock.sendFrame(blob))
          setPhase('running')
          setStatus('Hold still and look at the camera.')
          // Client-side auto-stop at 30s (backend also caps at 30s).
          autostopRef.current = setTimeout(stop, SCAN_SECONDS * 1000)
        } catch (e) {
          setPhase('error')
          if (e?.code === 'INSECURE_CONTEXT' || !window.isSecureContext) {
            setStatus('Camera needs a secure (HTTPS) connection. On a phone, open the https ngrok link — a plain http://LAN-IP address will not work.')
          } else if (e?.name === 'NotAllowedError') {
            setStatus('Camera permission was denied. Allow camera access in your browser settings and try again.')
          } else if (e?.name === 'NotFoundError') {
            setStatus('No camera found on this device.')
          } else {
            setStatus('Could not start the camera: ' + (e?.message || 'unknown error'))
          }
          cleanup()
        }
      },
      onMessage: handleMessage,
      onError: () => { setPhase('error'); setStatus('Connection error. Is the backend running?'); cleanup() },
    })
    sockRef.current = sock
    sock.connect()
  }, [handleMessage, cleanup, stop])

  const running = phase === 'running'
  const remaining = Math.max(0, SCAN_SECONDS - elapsed)
  const progress = Math.min(100, (elapsed / SCAN_SECONDS) * 100)
  const signalStable = (metrics.sqi ?? 0) > 0.5

  const insecure = typeof window !== 'undefined' && !window.isSecureContext

  return (
    <div>
      <Stepper current={1} />
      {insecure && (
        <div className="notice" role="alert"
          style={{ background: 'var(--warn-bg)', borderColor: '#fde68a', color: '#92400e', marginBottom: 18 }}>
          <span className="ic" aria-hidden="true">🔒</span>
          <div>
            <strong>Camera needs HTTPS.</strong> This page is on an insecure
            connection, so your browser blocks the camera. On a phone, open the
            <strong> https ngrok link</strong> instead of the <code>http://</code> LAN address.
          </div>
        </div>
      )}
      <div className="cam-layout">
        <div>
          <div className="camera-wrap">
            <video ref={videoRef} className="camera-video" playsInline muted />
            {!running && phase !== 'connecting' && (
              <div className="camera-placeholder">Camera is off — press Start measurement</div>
            )}
            {(running || phase === 'connecting') && (
              <div className={`face-guide ${faceDetected ? 'ok' : 'searching'}`} aria-hidden="true" />
            )}
            {running && (
              <>
                <div className="cam-countdown" aria-live="polite">⏱ {Math.ceil(remaining)}s</div>
                <div className="cam-progress"><i style={{ width: `${progress}%` }} /></div>
              </>
            )}
          </div>

          <div style={{ display: 'flex', gap: 12, marginTop: 16 }}>
            {!running ? (
              <button
                className="btn btn-primary" onClick={start}
                disabled={phase === 'connecting'}
              >
                {phase === 'connecting' ? 'Connecting…' : 'Start measurement'}
              </button>
            ) : (
              <button className="btn btn-danger" onClick={stop}>Stop &amp; see results</button>
            )}
          </div>
          <div className={`statusline ${phase === 'error' ? 'error' : ''}`} aria-live="polite">{status}</div>
        </div>

        <div>
          <div className="summary-card card">
            <h3>Patient summary</h3>
            <div className="summary-grid">
              <span className="k">Gender</span><span className="v">{form.gender || '—'}</span>
              <span className="k">Age</span><span className="v">{age != null ? `${age} yrs` : '—'}</span>
              <span className="k">Height</span><span className="v">{form.height ? `${form.height} cm` : '—'}</span>
              <span className="k">Weight</span><span className="v">{form.weight ? `${form.weight} kg` : '—'}</span>
              <span className="k">Usual BP</span><span className="v" style={{ textTransform: 'capitalize' }}>{form.bpCategory || '—'}</span>
            </div>
          </div>

          <div className="status-pills" role="status">
            <StatusPill ok={running} pending={!running} label={running ? 'Camera running' : 'Camera idle'} />
            <StatusPill ok={running && faceDetected} warn={running && !faceDetected} pending={!running} label={running && faceDetected ? 'Face detected' : 'Detecting face'} />
            <StatusPill ok={running && signalStable} warn={running && !signalStable} pending={!running} label={signalStable ? 'Signal stable' : 'Stabilizing signal'} />
            <StatusPill ok={running && elapsed >= 5} pending={!running || elapsed < 5} label="Measurement running" />
          </div>
        </div>
      </div>
    </div>
  )
}
