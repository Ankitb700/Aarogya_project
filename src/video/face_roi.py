from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ROIResult:
    frame_index: int
    detected: bool
    forehead: np.ndarray | None
    left_cheek: np.ndarray | None
    right_cheek: np.ndarray | None
    brightness: float
    roi_variance: float


def _landmark_box(landmarks, indices: list[int], width: int, height: int, pad: int = 6) -> tuple[int, int, int, int]:
    xs = [int(landmarks[i].x * width) for i in indices]
    ys = [int(landmarks[i].y * height) for i in indices]
    x1, x2 = max(min(xs) - pad, 0), min(max(xs) + pad, width)
    y1, y2 = max(min(ys) - pad, 0), min(max(ys) + pad, height)
    return x1, y1, x2, y2


def extract_mediapipe_rois(frame: np.ndarray, face_mesh, frame_index: int = 0) -> ROIResult:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    brightness = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
    if not results.multi_face_landmarks:
        return ROIResult(frame_index, False, None, None, None, brightness, np.nan)

    h, w = frame.shape[:2]
    landmarks = results.multi_face_landmarks[0].landmark
    regions = {
        "forehead": [10, 67, 69, 104, 108, 109, 151],
        "left_cheek": [117, 118, 119, 120, 123, 147, 187, 205],
        "right_cheek": [346, 347, 348, 349, 352, 376, 411, 425],
    }
    rois = {}
    variances = []
    for name, indices in regions.items():
        x1, y1, x2, y2 = _landmark_box(landmarks, indices, w, h)
        roi = frame[y1:y2, x1:x2]
        rois[name] = roi if roi.size else None
        if roi.size:
            variances.append(float(np.var(roi)))
    return ROIResult(
        frame_index=frame_index,
        detected=True,
        forehead=rois["forehead"],
        left_cheek=rois["left_cheek"],
        right_cheek=rois["right_cheek"],
        brightness=brightness,
        roi_variance=float(np.mean(variances)) if variances else np.nan,
    )


def mean_rgb(roi: np.ndarray | None) -> tuple[float, float, float]:
    if roi is None or roi.size == 0:
        return np.nan, np.nan, np.nan
    b, g, r = cv2.split(roi)
    return float(np.mean(r)), float(np.mean(g)), float(np.mean(b))
