from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA

from src.signals.filtering import butter_bandpass, detrend, zscore
from .face_roi import extract_mediapipe_rois, mean_rgb


def ica_rppg(rgb: np.ndarray, fps: float) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=float)
    rgb = rgb - np.nanmean(rgb, axis=0)
    ica = FastICA(n_components=3, max_iter=1000, random_state=42)
    components = ica.fit_transform(rgb)
    best = None
    best_score = -np.inf
    for i in range(components.shape[1]):
        comp = components[:, i]
        freqs, psd = __import__("scipy").signal.welch(
            comp[np.isfinite(comp)], fs=fps, nperseg=min(256, len(comp))
        )
        hr_band = (freqs >= 0.7) & (freqs <= 4.0)
        if not hr_band.any():
            continue
        peak_power = np.max(psd[hr_band])
        total_power = np.sum(psd)
        score = peak_power / (total_power + 1e-12)
        if score > best_score:
            best_score = score
            best = comp
    if best is None:
        best = components[:, 0]
    return zscore(butter_bandpass(best, fps))


def extract_multi_roi_rgb_traces(
    video_path: str | Path, max_frames: int | None = None
) -> pd.DataFrame:
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows = []
    face_mesh_factory = getattr(getattr(mp, "solutions", None), "face_mesh", None)
    face_mesh_context = (
        face_mesh_factory.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True
        )
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
                rgb_forehead = mean_rgb(result.forehead)
                rgb_left_cheek = mean_rgb(result.left_cheek)
                rgb_right_cheek = mean_rgb(result.right_cheek)
                detected = result.detected
                brightness = result.brightness
                roi_variance = result.roi_variance
            else:
                from .rppg import _haar_face_roi
                roi = _haar_face_roi(frame)
                rgb_forehead = mean_rgb(roi) if roi is not None else (np.nan, np.nan, np.nan)
                rgb_left_cheek = (np.nan, np.nan, np.nan)
                rgb_right_cheek = (np.nan, np.nan, np.nan)
                detected = roi is not None
                brightness = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
                roi_variance = float(np.var(roi)) if roi is not None else np.nan
            rows.append({
                "frame_index": frame_index,
                "time_seconds": frame_index / fps,
                "detected": detected,
                "brightness": brightness,
                "roi_variance": roi_variance,
                "forehead_r": rgb_forehead[0],
                "forehead_g": rgb_forehead[1],
                "forehead_b": rgb_forehead[2],
                "left_cheek_r": rgb_left_cheek[0],
                "left_cheek_g": rgb_left_cheek[1],
                "left_cheek_b": rgb_left_cheek[2],
                "right_cheek_r": rgb_right_cheek[0],
                "right_cheek_g": rgb_right_cheek[1],
                "right_cheek_b": rgb_right_cheek[2],
            })
            frame_index += 1
    cap.release()
    df = pd.DataFrame(rows)
    if len(df) >= 12:
        for roi_name in ["forehead", "left_cheek", "right_cheek"]:
            rgb = df[[f"{roi_name}_r", f"{roi_name}_g", f"{roi_name}_b"]].interpolate(limit_direction="both").to_numpy()
            if np.isnan(rgb).all():
                continue
            df[f"{roi_name}_rppg_green"] = __import__("src.video.rppg", fromlist=["green_rppg"]).green_rppg(rgb, fps)
            df[f"{roi_name}_rppg_chrom"] = __import__("src.video.rppg", fromlist=["chrom_rppg"]).chrom_rppg(rgb, fps)
            df[f"{roi_name}_rppg_pos"] = __import__("src.video.rppg", fromlist=["pos_rppg"]).pos_rppg(rgb, fps)
            df[f"{roi_name}_rppg_ica"] = ica_rppg(rgb, fps)
        full_face_rgb = df[[
            col for col in df.columns
            if col.endswith("_r") or col.endswith("_g") or col.endswith("_b")
        ]].groupby(lambda x: x.split("_")[-1], axis=1).mean().interpolate(limit_direction="both").to_numpy()
        df["rppg_green"] = __import__("src.video.rppg", fromlist=["green_rppg"]).green_rppg(full_face_rgb, fps)
        df["rppg_chrom"] = __import__("src.video.rppg", fromlist=["chrom_rppg"]).chrom_rppg(full_face_rgb, fps)
        df["rppg_pos"] = __import__("src.video.rppg", fromlist=["pos_rppg"]).pos_rppg(full_face_rgb, fps)
        df["rppg_ica"] = ica_rppg(full_face_rgb, fps)
    return df


def compute_per_roi_snr(df: pd.DataFrame, fps: float, hr_low: float = 0.7, hr_high: float = 4.0) -> dict:
    from scipy import signal as scipy_signal
    results = {}
    rois = ["forehead", "left_cheek", "right_cheek"]
    methods = ["rppg_green", "rppg_chrom", "rppg_pos", "rppg_ica"]
    signal_cols = [c for c in df.columns if any(c.endswith(m) for m in methods)]
    for col in signal_cols:
        values = df[col].dropna().to_numpy()
        if len(values) < 12:
            continue
        freqs, psd = scipy_signal.welch(values, fs=fps, nperseg=min(256, len(values)))
        hr_band = (freqs >= hr_low) & (freqs <= hr_high)
        if not hr_band.any():
            continue
        total_band = (freqs > 0) & (freqs <= fps / 2)
        signal_power = np.sum(psd[hr_band])
        total_power = np.sum(psd[total_band])
        noise_power = total_power - signal_power
        snr = 10 * np.log10(signal_power / (noise_power + 1e-12))
        peak_freq = freqs[hr_band][np.argmax(psd[hr_band])]
        results[col] = {"snr_db": snr, "peak_freq_hz": peak_freq}
    return results
