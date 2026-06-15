"""FastAPI backend for live rPPG vitals estimation.

Browser captures webcam frames and streams JPEGs over /ws/analyze; the backend
feeds them into a pre-warmed rppg.Model and streams HR / RR / HRV / stress back.
"""
import asyncio
import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState

from engine import DEFAULT_MODEL
from pool import ModelPool
from schemas import FinalMessage, MetricsMessage, StatusMessage
from session import AnalysisSession

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rppg.app")

POOL_SIZE = 1
MAX_DURATION_SEC = 30.0       # hard cap on a session
METRICS_INTERVAL_SEC = 1.0    # how often to push vitals
MIN_COLLECT_SEC = 5.0         # don't bother computing vitals before this

app = FastAPI(title="rPPG Vitals API")
app.add_middleware(
    CORSMiddleware,
    # Permissive: any origin. With the Vite dev proxy the WS is same-origin anyway,
    # but this keeps direct API access (LAN IP, tunnels) unrestricted.
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pool = ModelPool(size=POOL_SIZE, model_name=DEFAULT_MODEL)


@app.on_event("startup")
async def _startup():
    asyncio.create_task(pool.warmup())


@app.get("/health")
async def health():
    return {"status": "ok" if pool.ready else "warming", "ready": pool.ready}


@app.get("/api/info")
async def info():
    return {
        "model": DEFAULT_MODEL,
        "metrics": ["hr", "rr", "hrv_rmssd", "hrv_sdnn", "stress_index", "sqi"],
        "min_collect_sec": MIN_COLLECT_SEC,
        "max_duration_sec": MAX_DURATION_SEC,
        "note": "Estimates only. Not a medical device.",
    }


async def _send(ws: WebSocket, msg) -> bool:
    """Send a JSON message; return False if the socket is already gone."""
    if ws.client_state != WebSocketState.CONNECTED:
        return False
    try:
        await ws.send_text(msg.model_dump_json())
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False


@app.websocket("/ws/analyze")
async def analyze(ws: WebSocket):
    await ws.accept()

    if not pool.ready:
        await _send(ws, StatusMessage(state="warming", message="Model still loading, try again shortly."))
        await ws.close()
        return

    model = pool.try_acquire()
    if model is None:
        await _send(ws, StatusMessage(state="busy", message="All analysis slots are in use. Please retry."))
        await ws.close()
        return

    sess = AnalysisSession(model)
    try:
        await asyncio.to_thread(sess.open)
        await _send(ws, StatusMessage(state="ready", message="Hold still and look at the camera."))

        last_metrics = 0.0
        disconnected = False
        while True:
            if sess.elapsed >= MAX_DURATION_SEC:
                break

            try:
                packet = await asyncio.wait_for(ws.receive(), timeout=METRICS_INTERVAL_SEC)
            except asyncio.TimeoutError:
                packet = None

            if packet is not None:
                if packet.get("type") == "websocket.disconnect":
                    disconnected = True
                    break
                if "bytes" in packet and packet["bytes"] is not None:
                    await asyncio.to_thread(sess.push_jpeg, packet["bytes"])
                elif "text" in packet and packet["text"] is not None:
                    try:
                        ctrl = json.loads(packet["text"])
                    except json.JSONDecodeError:
                        ctrl = {}
                    if ctrl.get("action") == "stop":
                        break

            # Periodic metrics push.
            if sess.elapsed - last_metrics >= METRICS_INTERVAL_SEC:
                last_metrics = sess.elapsed
                if sess.elapsed < MIN_COLLECT_SEC:
                    await _send(ws, StatusMessage(
                        state="collecting", message="Collecting signal...",
                        elapsed=round(sess.elapsed, 1),
                    ))
                else:
                    vit = await asyncio.to_thread(sess.compute_vitals)
                    await _send(ws, MetricsMessage(
                        elapsed=round(sess.elapsed, 1),
                        face_detected=sess.face_detected, **vit,
                    ))

        # Final summary (skip if the client already went away).
        if not disconnected:
            vit = await asyncio.to_thread(sess.compute_vitals)
            trend = await asyncio.to_thread(sess.compute_trend)
            await _send(ws, FinalMessage(
                elapsed=round(sess.elapsed, 1), samples=sess.frame_count,
                trend=trend, **vit,
            ))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.exception("Session error")
        if ws.client_state == WebSocketState.CONNECTED:
            await _send(ws, StatusMessage(state="error", message=str(e)))
    finally:
        await asyncio.to_thread(sess.close)
        pool.release(model)
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.close()
