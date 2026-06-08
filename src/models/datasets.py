from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset


BVP_COLUMNS = ["bvp"]
PHYSIOLOGY_COLUMNS = ["eda", "temp", "acc_x", "acc_y", "acc_z", "ibi"]
TARGET_COLUMNS = ["hr", "cv_load_index", "cv_load_class", "bp_proxy_score"]


@dataclass(frozen=True)
class WindowSpec:
    subject_id: str
    trial_id: str
    aligned_path: Path
    start_row: int
    end_row: int
    video_path: Path | None = None


def _safe_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in columns:
        out[col] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
    return out.interpolate(limit_direction="both").fillna(0.0)


def _zscore_frame(df: pd.DataFrame) -> np.ndarray:
    arr = df.to_numpy(dtype=np.float32)
    mean = np.nanmean(arr, axis=0, keepdims=True)
    std = np.nanstd(arr, axis=0, keepdims=True)
    std[std == 0] = 1.0
    return ((arr - mean) / std).astype(np.float32)


def fit_standard_scaler(values: pd.DataFrame | pd.Series | np.ndarray) -> dict[str, list[float] | float]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    mean[~np.isfinite(mean)] = 0.0
    return {"mean": mean.tolist(), "std": std.tolist()}


def apply_standard_scaler(values: np.ndarray, scaler: dict | None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if scaler is None:
        return arr
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    return ((arr - mean) / std).astype(np.float32)


def inverse_standard_scaler(values: np.ndarray | float, scaler: dict | None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if scaler is None:
        return arr
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    return arr * std + mean


def fit_target_scalers(windows: pd.DataFrame) -> dict[str, dict]:
    return {
        "hr": fit_standard_scaler(windows["hr_target"].to_numpy(dtype=np.float32)),
        "bp_proxy": fit_standard_scaler(windows["bp_proxy_target"].to_numpy(dtype=np.float32)),
    }


def load_feature_table(features_path: str | Path | None) -> pd.DataFrame | None:
    if features_path is None:
        return None
    path = Path(features_path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def static_feature_columns(feature_table: pd.DataFrame) -> list[str]:
    excluded = {"subject_id", "trial_id", "hr_mean", "cv_load_class", "bp_proxy_score"}
    return [c for c in feature_table.columns if c not in excluded and pd.api.types.is_numeric_dtype(feature_table[c])]


def fit_static_scaler(feature_table: pd.DataFrame) -> dict:
    cols = static_feature_columns(feature_table)
    if not cols:
        return {"mean": [], "std": []}
    return fit_standard_scaler(feature_table[cols].fillna(0.0).to_numpy(dtype=np.float32))


def create_window_index(
    aligned_dir: str | Path,
    archive_records: list | None = None,
    window_seconds: int = 64,
    stride_seconds: int = 32,
    sample_rate_hz: float = 1.0,
) -> pd.DataFrame:
    aligned_dir = Path(aligned_dir)
    video_lookup = {}
    if archive_records is not None:
        video_lookup = {(r.subject_id, r.trial_id): str(r.video_path) if r.video_path else None for r in archive_records}

    rows = []
    window_rows = max(1, int(window_seconds * sample_rate_hz))
    stride_rows = max(1, int(stride_seconds * sample_rate_hz))
    for path in sorted(aligned_dir.glob("*_aligned_*hz.csv")):
        df = pd.read_csv(path, usecols=lambda col: col in {"subject_id", "trial_id", "hr"})
        if df.empty:
            continue
        subject_id = str(df["subject_id"].iloc[0])
        trial_id = str(df["trial_id"].iloc[0])
        for start in range(0, max(len(df) - window_rows + 1, 1), stride_rows):
            end = min(start + window_rows, len(df))
            if end - start < max(4, window_rows // 2):
                continue
            rows.append(
                {
                    "subject_id": subject_id,
                    "trial_id": trial_id,
                    "aligned_path": str(path),
                    "start_row": start,
                    "end_row": end,
                    "video_path": video_lookup.get((subject_id, trial_id)),
                }
            )
    return pd.DataFrame(rows)


def add_window_targets(window_index: pd.DataFrame, feature_table: pd.DataFrame | None) -> pd.DataFrame:
    index = window_index.copy()
    rows = []
    for _, row in index.iterrows():
        aligned = pd.read_csv(row["aligned_path"]).iloc[int(row["start_row"]): int(row["end_row"])]
        target = {
            "hr_target": float(pd.to_numeric(aligned.get("hr"), errors="coerce").mean()),
        }
        if feature_table is not None:
            match = feature_table[
                (feature_table["subject_id"] == row["subject_id"])
                & (feature_table["trial_id"] == row["trial_id"])
            ]
            if not match.empty:
                target["cv_load_index"] = float(match["cv_load_index"].iloc[0])
                target["cv_load_class"] = int(match["cv_load_class"].iloc[0])
                target["bp_proxy_target"] = float(match["bp_proxy_score"].iloc[0])
        if "cv_load_index" not in target:
            hr = target["hr_target"]
            target["cv_load_index"] = float(np.clip((hr - 60.0) / 60.0, 0.0, 1.0))
            target["cv_load_class"] = int(np.digitize(target["cv_load_index"], [0.33, 0.66]))
            target["bp_proxy_target"] = target["cv_load_index"]
        rows.append(target)
    targets = pd.DataFrame(rows)
    return pd.concat([index.reset_index(drop=True), targets], axis=1)


def split_window_index(windows: pd.DataFrame, test_size: float = 0.3, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    if windows.empty or windows["subject_id"].nunique() < 2:
        midpoint = max(1, int(len(windows) * (1 - test_size)))
        return windows.iloc[:midpoint].reset_index(drop=True), windows.iloc[midpoint:].reset_index(drop=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(windows, groups=windows["subject_id"]))
    return windows.iloc[train_idx].reset_index(drop=True), windows.iloc[val_idx].reset_index(drop=True)


class AlignedWindowDataset(Dataset):
    def __init__(
        self,
        windows: pd.DataFrame,
        feature_table: pd.DataFrame | None = None,
        sequence_length: int = 64,
        frame_size: int = 64,
        video_frames: int = 16,
        use_video: bool = True,
        use_face_roi: bool = True,
        static_scaler: dict | None = None,
        target_scalers: dict | None = None,
    ):
        self.windows = windows.reset_index(drop=True)
        self.feature_table = feature_table
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.video_frames = video_frames
        self.use_video = use_video
        self.use_face_roi = use_face_roi
        self.static_scaler = static_scaler
        self.target_scalers = target_scalers or {}
        self.static_columns = self._static_columns(feature_table)

    def _static_columns(self, feature_table: pd.DataFrame | None) -> list[str]:
        if feature_table is None:
            return []
        return static_feature_columns(feature_table)

    def __len__(self) -> int:
        return len(self.windows)

    def _resize_sequence(self, arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] == self.sequence_length:
            return arr
        source = np.linspace(0, 1, arr.shape[0])
        target = np.linspace(0, 1, self.sequence_length)
        return np.vstack([np.interp(target, source, arr[:, i]) for i in range(arr.shape[1])]).T.astype(np.float32)

    def _load_aligned_window(self, row: pd.Series) -> pd.DataFrame:
        aligned = pd.read_csv(row["aligned_path"])
        return aligned.iloc[int(row["start_row"]): int(row["end_row"])].reset_index(drop=True)

    def _load_static(self, row: pd.Series) -> np.ndarray:
        if self.feature_table is None or not self.static_columns:
            return np.zeros((0,), dtype=np.float32)
        match = self.feature_table[
            (self.feature_table["subject_id"] == row["subject_id"])
            & (self.feature_table["trial_id"] == row["trial_id"])
        ]
        if match.empty:
            return np.zeros((len(self.static_columns),), dtype=np.float32)
        values = match.iloc[0][self.static_columns].fillna(0.0).to_numpy(dtype=np.float32)
        return apply_standard_scaler(values, self.static_scaler)

    def _center_crop(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        side = min(h, w)
        y1 = max((h - side) // 2, 0)
        x1 = max((w - side) // 2, 0)
        return frame[y1:y1 + side, x1:x1 + side]

    def _face_roi_crop(self, frame: np.ndarray, face_mesh) -> np.ndarray:
        if face_mesh is not None:
            results = face_mesh.process(frame)
            if results.multi_face_landmarks:
                h, w = frame.shape[:2]
                landmarks = results.multi_face_landmarks[0].landmark
                xs = [int(lm.x * w) for lm in landmarks]
                ys = [int(lm.y * h) for lm in landmarks]
                pad_x = int((max(xs) - min(xs)) * 0.12)
                pad_y = int((max(ys) - min(ys)) * 0.18)
                x1 = max(min(xs) - pad_x, 0)
                x2 = min(max(xs) + pad_x, w)
                y1 = max(min(ys) - pad_y, 0)
                y2 = min(max(ys) + pad_y, h)
                crop = frame[y1:y2, x1:x2]
                if crop.size:
                    return crop

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        if len(faces):
            x, y, w, h = faces[0]
            pad_x = int(w * 0.15)
            pad_y = int(h * 0.2)
            x1 = max(x - pad_x, 0)
            y1 = max(y - pad_y, 0)
            x2 = min(x + w + pad_x, frame.shape[1])
            y2 = min(y + h + pad_y, frame.shape[0])
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                return crop
        return self._center_crop(frame)

    def _load_video_clip(self, row: pd.Series) -> np.ndarray:
        video_path = row.get("video_path")
        if not self.use_video or not isinstance(video_path, str) or not video_path or not Path(video_path).exists():
            return np.zeros((3, self.video_frames, self.frame_size, self.frame_size), dtype=np.float32)
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            cap.release()
            return np.zeros((3, self.video_frames, self.frame_size, self.frame_size), dtype=np.float32)
        frame_indices = np.linspace(0, total - 1, self.video_frames).astype(int)
        frames = []
        face_mesh = None
        if self.use_face_roi:
            try:
                import mediapipe as mp

                face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
            except Exception:
                face_mesh = None
        try:
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, frame = cap.read()
                if not ok:
                    frames.append(np.zeros((self.frame_size, self.frame_size, 3), dtype=np.float32))
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                crop = self._face_roi_crop(frame, face_mesh)
                crop = cv2.resize(crop, (self.frame_size, self.frame_size))
                frames.append(crop.astype(np.float32) / 255.0)
        finally:
            if face_mesh is not None:
                face_mesh.close()
            cap.release()
        arr = np.stack(frames, axis=0)
        return np.transpose(arr, (3, 0, 1, 2)).astype(np.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.windows.iloc[idx]
        aligned = self._load_aligned_window(row)
        bvp = self._resize_sequence(_zscore_frame(_safe_numeric(aligned, BVP_COLUMNS))).T
        physiology = self._resize_sequence(_zscore_frame(_safe_numeric(aligned, PHYSIOLOGY_COLUMNS))).T
        static = self._load_static(row)
        video = self._load_video_clip(row)

        hr_actual = np.float32(row["hr_target"])
        bp_actual = np.float32(row["bp_proxy_target"])
        hr_scaled = apply_standard_scaler(np.array([hr_actual], dtype=np.float32), self.target_scalers.get("hr"))[0]
        bp_scaled = apply_standard_scaler(np.array([bp_actual], dtype=np.float32), self.target_scalers.get("bp_proxy"))[0]

        return {
            "video": torch.from_numpy(video),
            "bvp": torch.from_numpy(bvp),
            "physiology": torch.from_numpy(physiology),
            "static": torch.from_numpy(static),
            "hr_target": torch.tensor(np.float32(hr_scaled)),
            "hr_target_raw": torch.tensor(hr_actual),
            "cv_load_index": torch.tensor(np.float32(row["cv_load_index"])),
            "cv_load_class": torch.tensor(int(row["cv_load_class"]), dtype=torch.long),
            "bp_proxy_target": torch.tensor(np.float32(bp_scaled)),
            "bp_proxy_target_raw": torch.tensor(bp_actual),
        }


class MultimodalFeatureDataset(AlignedWindowDataset):
    """Backward-compatible alias for earlier notebooks."""
