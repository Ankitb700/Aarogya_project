"""Outgoing WebSocket message schemas."""
from typing import List, Literal, Optional

from pydantic import BaseModel


class StatusMessage(BaseModel):
    type: Literal["status"] = "status"
    state: str                       # e.g. "ready", "collecting", "no_face", "busy", "error"
    message: str
    elapsed: float = 0.0


class MetricsMessage(BaseModel):
    type: Literal["metrics"] = "metrics"
    elapsed: float
    hr: Optional[float] = None
    rr: Optional[float] = None
    hrv_rmssd: Optional[float] = None
    hrv_sdnn: Optional[float] = None
    stress_index: Optional[float] = None
    sqi: Optional[float] = None
    beats: Optional[int] = None
    face_detected: bool = True


class FinalMessage(BaseModel):
    type: Literal["final"] = "final"
    elapsed: float
    hr: Optional[float] = None
    rr: Optional[float] = None
    hrv_rmssd: Optional[float] = None
    hrv_sdnn: Optional[float] = None
    stress_index: Optional[float] = None
    sqi: Optional[float] = None
    beats: Optional[int] = None
    samples: int = 0
    trend: List[dict] = []
