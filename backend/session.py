"""Drives one analysis session over a single rppg.Model instance.

A session owns the `with model:` lifecycle, paces incoming browser frames into
`model.update_frame`, and periodically derives vitals via the existing
`extract_vitals` helper. All blocking model/numpy work is delegated to threads
by the caller (app.py) so the event loop stays responsive.
"""
import time

import cv2
import numpy as np

from engine import extract_vitals


class AnalysisSession:
    def __init__(self, model):
        self.model = model
        self._entered = False
        self._t0 = None
        self.frame_count = 0
        self.last_box_seen = 0.0  # wall time of last successful face box

    # --- lifecycle -------------------------------------------------------
    def open(self):
        self.model.__enter__()
        self._entered = True
        self._t0 = time.monotonic()
        self.last_box_seen = self._t0

    def close(self):
        if self._entered:
            try:
                self.model.__exit__(None, None, None)
            except Exception:
                pass
            self._entered = False

    @property
    def elapsed(self) -> float:
        return 0.0 if self._t0 is None else time.monotonic() - self._t0

    # --- frame ingestion -------------------------------------------------
    def push_jpeg(self, data: bytes, ts: float = None) -> bool:
        """Decode a JPEG frame and feed it to the model. Returns True on success.

        If ``ts`` is None the frame is timestamped by real elapsed capture time
        (live mode); pass an explicit ts to replay a recording at native timing.
        """
        arr = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return False
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Pace by real elapsed capture time so the engine's frame-rate logic
        # stays consistent regardless of how fast frames arrive.
        if ts is None:
            ts = self.elapsed
        self.model.update_frame(rgb, ts=ts)
        self.frame_count += 1
        if self.model.box is not None:
            self.last_box_seen = time.monotonic()
        return True

    @property
    def face_detected(self) -> bool:
        return (time.monotonic() - self.last_box_seen) < 1.5

    # --- vitals ----------------------------------------------------------
    def compute_vitals(self) -> dict:
        """Return current vitals + SQI. Keys are None when not enough data yet."""
        result = {
            "hr": None, "rr": None, "hrv_rmssd": None, "hrv_sdnn": None,
            "stress_index": None, "sqi": None, "beats": None,
        }
        try:
            signal, ts = self.model.bvp()
        except Exception:
            signal, ts = [], []

        if signal is not None and len(signal) >= 300:
            vit = extract_vitals(signal, ts)
            if vit:
                result.update(
                    hr=vit["hr"], rr=vit["rr"], hrv_rmssd=vit["hrv_rmssd"],
                    hrv_sdnn=vit["hrv_sdnn"], stress_index=vit["stress_index"],
                    beats=vit["beats"],
                )

        try:
            hr_info = self.model.hr()
            if hr_info:
                result["sqi"] = (
                    round(float(hr_info["SQI"]), 3)
                    if hr_info.get("SQI") is not None else None
                )
        except Exception:
            pass

        return result

    def compute_trend(self, step: float = 1.5) -> list:
        """Slide a window over the full BVP signal and compute per-window vitals,
        producing a real time-series for the charts.

        Returns a list of {t, hr, rr, rmssd, sdnn, stress} sampled across the scan.
        `extract_vitals` requires >=300 samples, so the window is sized from the
        measured sample rate to always contain enough data.
        """
        try:
            signal, ts = self.model.bvp()
        except Exception:
            return []
        signal = np.asarray(signal)
        ts = np.asarray(ts)
        if len(signal) < 300:
            return []

        duration = ts[-1] - ts[0]
        if duration <= 0:
            return []
        fps = len(signal) / duration
        # Window large enough to hold ~330 samples (>= the 300 minimum).
        window = max(10.0, 330.0 / fps)
        if window >= duration:
            return []

        t0 = ts[0]
        out = []
        we = t0 + window
        while we <= ts[-1] + 1e-6:
            ws = we - window
            mask = (ts >= ws) & (ts <= we)
            seg = signal[mask]
            segt = ts[mask]
            if len(seg) >= 300:
                v = extract_vitals(seg, segt)
                if v:
                    out.append({
                        "t": round(float(we - t0), 1),
                        "hr": v["hr"], "rr": v["rr"],
                        "rmssd": v["hrv_rmssd"], "sdnn": v["hrv_sdnn"],
                        "stress": v["stress_index"],
                    })
            we += step
        return out
