from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd


EMPATICA_DIR_NAMES = ("empatica_e4", "empatica_data")
SIGNAL_FILES = ("ACC", "BVP", "EDA", "HR", "IBI", "TEMP", "tags")


@dataclass(frozen=True)
class TrialRecord:
    subject_id: str
    trial_id: str
    trial_path: Path
    empatica_path: Path
    video_path: Path | None
    track_path: Path | None
    files: dict[str, Path]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("trial_path", "empatica_path", "video_path", "track_path"):
            if data[key] is not None:
                data[key] = str(data[key])
        data["files"] = {name: str(path) for name, path in self.files.items()}
        return data


def find_empatica_dir(trial_path: Path) -> Path | None:
    for name in EMPATICA_DIR_NAMES:
        candidate = trial_path / name
        if candidate.exists():
            return candidate
    candidates = [p for p in trial_path.iterdir() if p.is_dir() and p.name.startswith("empatica")]
    return candidates[0] if candidates else None


def find_video_path(trial_path: Path) -> Path | None:
    video_dir = trial_path / "video"
    if not video_dir.exists():
        return None
    for suffix in ("*.MOV", "*.mov", "*.MP4", "*.mp4", "*.avi"):
        matches = sorted(video_dir.glob(suffix))
        if matches:
            return matches[0]
    return None


def scan_archive(archive_dir: str | Path) -> list[TrialRecord]:
    archive_dir = Path(archive_dir)
    trials: list[TrialRecord] = []
    for subject_outer in sorted(archive_dir.glob("subject_*")):
        if not subject_outer.is_dir():
            continue
        subject_id = subject_outer.name
        for trial_path in sorted(subject_outer.glob(f"{subject_id}/trial_*")):
            empatica_path = find_empatica_dir(trial_path)
            if empatica_path is None:
                continue
            files = {
                name: empatica_path / f"{name}.csv"
                for name in SIGNAL_FILES
                if (empatica_path / f"{name}.csv").exists()
            }
            trials.append(
                TrialRecord(
                    subject_id=subject_id,
                    trial_id=trial_path.name,
                    trial_path=trial_path,
                    empatica_path=empatica_path,
                    video_path=find_video_path(trial_path),
                    track_path=empatica_path / "track.txt" if (empatica_path / "track.txt").exists() else None,
                    files=files,
                )
            )
    return trials


def trial_manifest(archive_dir: str | Path) -> pd.DataFrame:
    records = scan_archive(archive_dir)
    rows = []
    for record in records:
        row = record.to_dict()
        for name in SIGNAL_FILES:
            row[f"has_{name.lower()}"] = name in record.files
        rows.append(row)
    return pd.DataFrame(rows)
