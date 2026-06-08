from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .archive_scanner import TrialRecord
from .empatica_loader import load_tags, load_track_notes, load_trial_signals, parse_track_alignment


def determine_analysis_window(tags: pd.Series, track_text: str, fallback_start: float, fallback_end: float) -> tuple[float, float]:
    parsed = parse_track_alignment(track_text)
    if len(tags) > int(parsed["end_event_index"]):
        return float(tags.iloc[int(parsed["start_event_index"])]), float(tags.iloc[int(parsed["end_event_index"])])
    if len(tags) >= 2:
        return float(tags.iloc[0]), float(tags.iloc[-1])
    return fallback_start, fallback_end


def resample_signal(df: pd.DataFrame, columns: list[str], timeline: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"timestamp": timeline})
    source = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if source.empty:
        for col in columns:
            out[col] = np.nan
        return out
    for col in columns:
        valid = source[["timestamp", col]].dropna()
        if len(valid) < 2:
            out[col] = valid[col].iloc[0] if len(valid) == 1 else np.nan
        else:
            out[col] = np.interp(timeline, valid["timestamp"], valid[col], left=np.nan, right=np.nan)
    return out


def align_trial(record: TrialRecord, hz: float = 1.0) -> tuple[pd.DataFrame, dict]:
    signals, metadata = load_trial_signals(record.files)
    tags = load_tags(record.files["tags"]) if "tags" in record.files else pd.Series(dtype=float)
    track_text = load_track_notes(record.track_path)

    starts = [df["timestamp"].min() for df in signals.values() if "timestamp" in df]
    ends = [df["timestamp"].max() for df in signals.values() if "timestamp" in df]
    fallback_start = float(np.nanmax(starts))
    fallback_end = float(np.nanmin(ends))
    start, end = determine_analysis_window(tags, track_text, fallback_start, fallback_end)
    start = max(start, fallback_start)
    end = min(end, fallback_end)
    if end <= start:
        start, end = fallback_start, fallback_end

    timeline = np.arange(start, end, 1.0 / hz)
    aligned = pd.DataFrame({"timestamp": timeline})
    aligned["elapsed_seconds"] = aligned["timestamp"] - start
    aligned["subject_id"] = record.subject_id
    aligned["trial_id"] = record.trial_id

    for name, df in signals.items():
        if name == "IBI":
            continue
        columns = [c for c in df.columns if c not in {"sample_index", "timestamp"}]
        sampled = resample_signal(df, columns, timeline)
        for col in columns:
            aligned[col] = sampled[col]

    if "IBI" in signals:
        ibi = signals["IBI"]
        aligned["ibi"] = np.nan
        for _, row in ibi.iterrows():
            idx = np.searchsorted(timeline, row["timestamp"])
            if 0 <= idx < len(aligned):
                aligned.loc[idx, "ibi"] = row["ibi"]
        aligned["ibi"] = aligned["ibi"].ffill()

    meta = {
        "subject_id": record.subject_id,
        "trial_id": record.trial_id,
        "start_timestamp": start,
        "end_timestamp": end,
        "duration_seconds": end - start,
        "tags": tags.tolist(),
        "track_alignment": parse_track_alignment(track_text),
        "signal_metadata": metadata,
        "video_path": str(record.video_path) if record.video_path else None,
    }
    return aligned, meta


def save_aligned_trial(record: TrialRecord, output_dir: str | Path, hz: float = 1.0) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aligned, _ = align_trial(record, hz=hz)
    path = output_dir / f"{record.subject_id}_{record.trial_id}_aligned_{hz:g}hz.csv"
    aligned.to_csv(path, index=False)
    return path
