from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.archive_scanner import scan_archive
from src.data.alignment import align_trial


def validate_archive(archive_dir: str | Path) -> dict[str, int]:
    records = scan_archive(archive_dir)
    return {
        "n_trials": len(records),
        "n_subjects": len({r.subject_id for r in records}),
        "n_with_video": sum(r.video_path is not None for r in records),
        "n_with_tags": sum("tags" in r.files for r in records),
    }


def validate_alignment(record, hz: float = 1.0) -> dict[str, object]:
    aligned, meta = align_trial(record, hz=hz)
    return {
        "rows": len(aligned),
        "columns": list(aligned.columns),
        "duration_seconds": meta["duration_seconds"],
        "has_hr": "hr" in aligned.columns and aligned["hr"].notna().any(),
    }


def assert_feature_table_ok(features: pd.DataFrame) -> None:
    required = ["subject_id", "trial_id", "hr_mean", "cv_load_index", "bp_proxy_score"]
    missing = [c for c in required if c not in features.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    all_null = [c for c in required if features[c].isna().all()]
    if all_null:
        raise ValueError(f"Required feature columns are all null: {all_null}")
