# UI Redesign Changelog

Transformed the rPPG vitals app from a single-page technical prototype into a
multi-step, healthcare-grade web experience: **Landing → Patient Intake →
Camera Measurement → Results**, with a clean clinical design system.

---

## Files Created

### Pages (`frontend/src/pages/`)
- `LandingPage.jsx` — hero with inline medical SVG, benefit pills, 4 feature
  cards (Contactless, Heart Rate, Blood Pressure context, Respiratory),
  "How it works" steps, privacy notice, **Start Health Assessment** CTA.
- `IntakeForm.jsx` — patient intake (Step 1 of 2): gender (radio cards),
  weight (kg, 20–250), height (cm, 100–250), birth year → auto age, BP category
  (Normal / Prehypertension / Hypertension with SBP ranges). Inline validation,
  progress stepper, autosave via context/localStorage.
- `CameraPage.jsx` — modernized measurement screen: patient summary card,
  status pills (camera / face detected / signal stable / measurement running),
  face-guide overlay, 30s countdown + progress bar. Routes to results on finish.
- `ResultsPage.jsx` — dashboard: KPI cards with circular gauges (HR, RR, Stress),
  HRV (RMSSD/SDNN) cards, BP-category card, HR trend chart, SQI + disclaimer.

### Components (`frontend/src/components/`)
- `Stepper.jsx` — 3-step progress indicator (Patient info → Measurement → Results).
- `ProgressRing.jsx` — accessible circular gauge (SVG) for KPI values.

### State
- `frontend/src/state/PatientContext.jsx` — Context API store for patient form
  + results, with localStorage autosave and automatic age calculation.

### Other
- `frontend/src/lib/config.js` — host-derived API/WS base (LAN-friendly).
- `.claude/launch.json` — preview/run config for the frontend.
- `UI_REDESIGN_CHANGELOG.md` — this file.

## Files Modified

- `frontend/src/main.jsx` — wrapped app in `BrowserRouter` + `PatientProvider`.
- `frontend/src/App.jsx` — now an app shell (top bar + routed pages + footer)
  instead of the monolithic camera page.
- `frontend/src/styles.css` — **full rewrite**: light, clinical healthcare design
  system (Medical Blue #2563EB, Healthcare Teal #0EA5A4, soft grays, white
  surfaces, status colors), glassmorphism, shadows, rounded corners, responsive
  grids, reduced-motion support.
- `frontend/vite.config.js` — `server.host = true` to expose on the LAN (phone testing).
- `backend/app.py` — `MAX_DURATION_SEC` 60 → **30s**; CORS now allows any
  `http://*:5173` origin (LAN devices).

## Files Removed

- `frontend/src/components/CameraView.jsx`, `VitalsDashboard.jsx`,
  `LiveChart.jsx`, `SessionControls.jsx` — superseded by the new page-based UI.

## Features Added

- **Multi-step onboarding flow** — app no longer opens directly to the camera.
- **Landing page** introducing purpose, method, duration, privacy, and steps.
- **Patient intake form** with validation, auto-age, progress, and autosave.
- **Healthcare theme** — consistent, calm, professional, accessible palette.
- **Enhanced camera page** — patient summary, live status pills, face-detection
  + signal-quality indicators, 30-second countdown and progress bar.
- **Modernized results** — KPI cards, circular gauges, trend chart, status tags.
- **30-second scan** (client auto-stop + backend cap).
- **LAN exposure** for phone/tablet testing.

## Accessibility

- Semantic `<fieldset>/<legend>`, `<label>`-bound inputs, `role="alert"` errors,
  `aria-live` status regions, `aria-label` on icons/SVGs.
- Visible focus ring (`:focus-visible`) on all interactive elements.
- Large touch targets (≥48px) for buttons and inputs.
- Responsive typography (`clamp()`), mobile-first grids (breakpoints at 900/560px).
- `prefers-reduced-motion` respected.

---

## Results Dashboard Redesign (v2)

Rebuilt the results page into a premium, responsive healthcare analytics dashboard
(Apple Health / Oura / WHOOP style) using CSS Grid that scales from mobile (1 col)
→ tablet (2 col) → desktop (4-col KPI row + multi-panel) and full-bleed-widens to
~1340–1480px on large monitors.

### Created
- `frontend/src/lib/health.js` — derived analytics: wellness/recovery/readiness
  scores, normalized radar dimensions, signal-quality grading, plain-language
  insights, and series stats (avg/min/max).
- `frontend/src/components/ScoreGauge.jsx` — prominent circular score gauge.

### Modified
- `frontend/src/pages/ResultsPage.jsx` — full rewrite:
  - **Assessment summary hero** (status, gender/age, scan duration, frames, signal quality, time).
  - **4-up KPI row** — HR (bpm), RR (bpm), Stress (/100), BP (mmHg) with emojis + status tags.
  - **Wellness score gauge** (0–100 + label) and **plain-language Health Insights** panel.
  - **Time-series charts**: HR trend (with avg/min/max + average reference line),
    Respiratory trend (area), HRV/RMSSD evolution, Signal-quality trend.
  - **Health radar** (Cardiovascular, Recovery, Stress, Respiratory, Wellness, Readiness).
  - **Recovery analysis** panel (recovery gauge + readiness/recovery/calmness bars).
  - **Advanced HRV analytics** section — RMSSD & SDNN moved out of primary vitals,
    with descriptions, "Learn more" links, reference ranges, and a comparison bar chart.
  - **Stress distribution** chart + **Vitals summary** table (avg/min/max).
  - Includes a dev-only `?demo=1` fixture (gated by `import.meta.env.DEV`) for previewing.
- `frontend/src/pages/CameraPage.jsx` — the live series now records the full metric
  set per tick (hr, rr, rmssd, sdnn, stress, sqi) to drive all dashboard charts.
- `frontend/src/styles.css` — dashboard design system: KPI/score/insight/recovery/
  advanced-metric/summary-table styles, hover states, responsive grids, full-bleed
  widening, `overflow-x: clip` guard.

### Units
Heart Rate **bpm**, Respiratory Rate **bpm**, HRV **ms**, Blood Pressure **mmHg**.

### Verified
Rendered via preview: charts + 2 gauges, radar, wellness score, insights, signal-
quality trend all render with no console errors and no horizontal overflow; grid
collapses correctly at mobile width.

### Time-series data fix (backend-computed trends)
The per-second live metrics are *cumulative* (recomputed over the whole signal,
null until ~300 samples), which left the trend charts nearly empty. Fixed by
computing the trend on the backend from the full 30s BVP signal:
- `backend/session.py` — new `compute_trend()` slides a window (sized from the
  measured sample rate to always hold ≥300 samples) over the full signal and runs
  `extract_vitals` per window → `[{t, hr, rr, rmssd, sdnn, stress}, …]`.
  Validated on a synthetic 72-bpm signal: 13 points, HR 71.8–72.2.
- `backend/schemas.py` / `backend/app.py` — `FinalMessage.trend` carries the series.
- `frontend/src/pages/CameraPage.jsx` — prefers `final.trend` for the chart series
  (falls back to client samples).
- **Removed** the Stress Distribution card (per request).

## Remaining / Future Improvements

- **Live BP estimation** — currently the engine does not estimate blood pressure;
  the Results BP card shows the self-reported category. Wire a BP model later.
- High-contrast (`prefers-contrast`) theme variant.
- Code-split the recharts bundle (build warns >500 kB).
- Optional downloadable PDF/printable report (engine already has
  `generate_visual_report`).
- Persist/replay historical sessions; multi-user backend pool sizing.
- Silence React Router v7 future-flag warnings by opting into the flags.
