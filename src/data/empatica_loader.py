from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


MULTI_COLUMN_NAMES = {
    "ACC": ["acc_x", "acc_y", "acc_z"],
}

SINGLE_COLUMN_NAMES = {
    "BVP": "bvp",
    "EDA": "eda",
    "HR": "hr",
    "TEMP": "temp",
}


def _read_raw_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, header=None)


def load_timeseries(path: str | Path, signal_name: str) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    signal_name = signal_name.upper()
    raw = _read_raw_csv(path)
    start_time = float(raw.iloc[0, 0])
    sample_rate = float(raw.iloc[1, 0])
    values = raw.iloc[2:].reset_index(drop=True).apply(pd.to_numeric, errors="coerce")

    if signal_name in MULTI_COLUMN_NAMES:
        values = values.iloc[:, : len(MULTI_COLUMN_NAMES[signal_name])]
        values.columns = MULTI_COLUMN_NAMES[signal_name]
    else:
        column = SINGLE_COLUMN_NAMES.get(signal_name, signal_name.lower())
        values = values.iloc[:, [0]]
        values.columns = [column]

    values.insert(0, "sample_index", np.arange(len(values)))
    values.insert(1, "timestamp", start_time + values["sample_index"] / sample_rate)
    meta = {"signal": signal_name, "start_time": start_time, "sample_rate": sample_rate, "path": str(path)}
    return values, meta


def load_ibi(path: str | Path) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    raw = pd.read_csv(path, header=None)
    start_time = float(raw.iloc[0, 0])
    values = raw.iloc[1:, :2].reset_index(drop=True).apply(pd.to_numeric, errors="coerce")
    values.columns = ["offset_seconds", "ibi"]
    values["timestamp"] = start_time + values["offset_seconds"]
    meta = {"signal": "IBI", "start_time": start_time, "sample_rate": None, "path": str(path)}
    return values[["timestamp", "offset_seconds", "ibi"]], meta


def load_tags(path: str | Path) -> pd.Series:
    raw = pd.read_csv(path, header=None)
    return pd.to_numeric(raw.iloc[:, 0], errors="coerce").dropna().reset_index(drop=True)


def load_track_notes(path: str | Path | None) -> str:
    if path is None:
        return ""
    path = Path(path)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def parse_track_alignment(track_text: str) -> dict:
    alignment: dict[str, float | int | None] = {
        "video_event_frame": None,
        "start_event_index": 1,
        "end_event_index": 2,
    }
    frame_match = re.search(r"(\d+)(?:st|nd|rd|th)?\s+frames?", track_text, flags=re.IGNORECASE)
    if frame_match:
        alignment["video_event_frame"] = int(frame_match.group(1))
    if "2nd eventmark to the 3rd eventmark" in track_text.lower():
        alignment["start_event_index"] = 1
        alignment["end_event_index"] = 2
    return alignment


def load_trial_signals(files: dict[str, Path]) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    signals: dict[str, pd.DataFrame] = {}
    metadata: dict[str, dict] = {}
    for name, path in files.items():
        upper = name.upper()
        if upper == "TAGS":
            continue
        if upper == "IBI":
            signals[upper], metadata[upper] = load_ibi(path)
        else:
            signals[upper], metadata[upper] = load_timeseries(path, upper)
    return signals, metadata
