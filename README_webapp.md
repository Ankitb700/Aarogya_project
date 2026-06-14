# rPPG Vitals Web App

Live contactless vitals (HR, RR, HRV, stress) from a webcam, built on the
existing `open_rppg_inference` engine.

- **Backend**: FastAPI + WebSocket (`backend/`). Holds a pre-warmed `rppg.Model`
  pool; receives JPEG frames over `/ws/analyze`, feeds them to `model.update_frame`,
  and streams vitals back (reusing `vitals_analyzer.extract_vitals`).
- **Frontend**: React + Vite (`frontend/`). Captures webcam → canvas → JPEG,
  streams over WebSocket, renders a live dashboard + HR chart.

> ⚠️ Wellness/demo only — estimates, not medical measurements.

## Prerequisites

The Python engine deps live in the **`.infer_venv`** virtualenv at the repo root
(jax, keras, onnxruntime, opencv, scipy, heartpy, av). Web deps (fastapi,
uvicorn, websockets) are installed there too.

## Run the backend

```powershell
cd E:\phase_1\backend
$env:KERAS_BACKEND = "jax"
# --host 0.0.0.0 exposes the backend on the LAN so a phone can reach it.
..\.infer_venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000
```

First start takes ~30–60 s while the model loads + JIT-compiles. Check readiness:

```powershell
curl http://localhost:8000/health      # {"status":"ok","ready":true}
curl http://localhost:8000/api/info
```

## Run the frontend

```powershell
cd E:\phase_1\frontend
npm install      # first time only
npm run dev      # exposes Local + Network (LAN) URLs
```

The app flow is **Landing → Patient info → Camera → Results**. Open the app,
fill the intake form, then run a **30-second** contactless scan.

### Single-origin proxy (important)
The Vite dev server proxies `/ws`, `/api` and `/health` to the backend
(`vite.config.js`). So the page **and** the WebSocket share one origin — the
frontend connects to `(ws|wss)://<same-host>/ws/analyze`. This means:
- No CORS / mixed-content issues.
- Only the frontend (`:5173`) needs to be reachable/tunneled; it forwards to the
  backend on `localhost:8000` internally.

### Testing on a phone — use ngrok (HTTPS)
Mobile browsers **block the camera on insecure origins**, so a plain LAN IP
(`http://192.168.x.x:5173`) will show the UI but the camera stays off. Use an
HTTPS tunnel to the **frontend port only**:

```powershell
# 1) backend (this machine)
..\.infer_venv\Scripts\python.exe -m uvicorn app:app --port 8000   # localhost is fine
# 2) frontend
npm run dev
# 3) tunnel the frontend
ngrok http 5173
```

Open the `https://<id>.ngrok-free.app` URL on your phone. Camera works (secure
context) and the WebSocket rides the same tunnel as `wss://…/ws/analyze`.
`server.allowedHosts: true` in `vite.config.js` permits the ngrok host.

> LAN IP without HTTPS only works for camera on desktop `localhost`. For phones,
> the ngrok (or any TLS) path is required.

## Verify the engine path without a browser

```powershell
cd E:\phase_1\backend
..\.infer_venv\Scripts\python.exe verify_engine.py [optional_face_video.mp4]
```

Replays a video through the same `update_frame` path the WebSocket uses. Note the
bundled `Recording 2026-06-09 ...mp4` is a screen capture with **no face**, so it
validates ingestion but cannot produce HR — use a real face recording for that.

## Notes / next steps
- Single concurrent session by default (`POOL_SIZE = 1` in `app.py`); raise it for
  more simultaneous users (each instance costs memory).
- BP is intentionally not included yet (the engine/`vitals_analyzer` don't estimate it).
- Off-localhost deployment needs HTTPS + `wss://` for `getUserMedia` to work.
