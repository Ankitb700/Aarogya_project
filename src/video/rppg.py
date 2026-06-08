from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.signals.filtering import butter_bandpass, detrend, zscore
from .face_roi import extract_mediapipe_rois, mean_rgb


def _haar_face_roi(frame: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return frame[y:y + h, x:x + w]


def green_rppg(rgb: np.ndarray, fps: float) -> np.ndarray:
    return butter_bandpass(zscore(detrend(rgb[:, 1])), fps)


def chrom_rppg(rgb: np.ndarray, fps: float, window_seconds: float = 1.6) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=float)
    norm = rgb / (np.nanmean(rgb, axis=0) + 1e-12)
    x = 3 * norm[:, 0] - 2 * norm[:, 1]
    y = 1.5 * norm[:, 0] + norm[:, 1] - 1.5 * norm[:, 2]
    alpha = np.nanstd(x) / (np.nanstd(y) + 1e-12)
    return butter_bandpass(zscore(x - alpha * y), fps)


def pos_rppg(rgb: np.ndarray, fps: float) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=float)
    norm = rgb / (np.nanmean(rgb, axis=0) + 1e-12)
    s1 = norm[:, 1] - norm[:, 2]
    s2 = norm[:, 1] + norm[:, 2] - 2 * norm[:, 0]
    alpha = np.nanstd(s1) / (np.nanstd(s2) + 1e-12)
    return butter_bandpass(zscore(s1 + alpha * s2), fps)


def extract_video_rgb_traces(video_path: str | Path, max_frames: int | None = None) -> pd.DataFrame:
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows = []
    previous_center = None
    face_mesh_factory = getattr(getattr(mp, "solutions", None), "face_mesh", None)
    face_mesh_context = (
        face_mesh_factory.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        if face_mesh_factory is not None
        else None
    )

    class _NullContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    with (face_mesh_context or _NullContext()) as face_mesh:
        frame_index = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_index >= max_frames:
                break
            if face_mesh is not None:
                result = extract_mediapipe_rois(frame, face_mesh, frame_index)
                region_rgbs = [mean_rgb(result.forehead), mean_rgb(result.left_cheek), mean_rgb(result.right_cheek)]
                arr = np.array(region_rgbs, dtype=float)
                rgb_mean = np.nanmean(arr, axis=0)
                detected = result.detected
                brightness = result.brightness
                roi_variance = result.roi_variance
            else:
                roi = _haar_face_roi(frame)
                rgb_mean = np.array(mean_rgb(roi), dtype=float)
                detected = roi is not None
                brightness = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
                roi_variance = float(np.var(roi)) if roi is not None else np.nan
            motion = np.nan
            if detected:
                center = np.array([rgb_mean[0], rgb_mean[1]])
                if previous_center is not None:
                    motion = float(np.linalg.norm(center - previous_center))
                previous_center = center
            rows.append(
                {
                    "frame_index": frame_index,
                    "time_seconds": frame_index / fps,
                    "detected": detected,
                    "r_mean": rgb_mean[0],
                    "g_mean": rgb_mean[1],
                    "b_mean": rgb_mean[2],
                    "brightness": brightness,
                    "roi_variance": roi_variance,
                    "face_motion_score": motion,
                }
            )
            frame_index += 1
    cap.release()
    traces = pd.DataFrame(rows)
    if len(traces) >= 12:
        rgb = traces[["r_mean", "g_mean", "b_mean"]].interpolate(limit_direction="both").to_numpy()
        traces["rppg_green"] = green_rppg(rgb, fps)
        traces["rppg_chrom"] = chrom_rppg(rgb, fps)
        traces["rppg_pos"] = pos_rppg(rgb, fps)
    return traces
