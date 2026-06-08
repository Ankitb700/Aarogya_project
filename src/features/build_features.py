from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.archive_scanner import TrialRecord, scan_archive
from src.data.empatica_loader import load_trial_signals
from src.signals.bvp_features import pulse_features
from src.signals.hrv import hrv_features
from src.signals.quality import motion_magnitude, snr_estimate, stability_score
from .pseudo_targets import add_pseudo_targets


def summarize_series(prefix: str, values: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(values, errors="coerce")
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
    }


def build_trial_features(record: TrialRecord) -> dict[str, float | str]:
    signals, metadata = load_trial_signals(record.files)
    row: dict[str, float | str] = {"subject_id": record.subject_id, "trial_id": record.trial_id}

    if "HR" in signals:
        row.update(summarize_series("hr", signals["HR"]["hr"]))
    if "EDA" in signals:
        row.update(summarize_series("eda", signals["EDA"]["eda"]))
    if "TEMP" in signals:
        row.update(summarize_series("temp", signals["TEMP"]["temp"]))
        row["optical_stability_score"] = stability_score(signals["TEMP"]["temp"].to_numpy())
    if "ACC" in signals:
        mag = motion_magnitude(signals["ACC"]["acc_x"], signals["ACC"]["acc_y"], signals["ACC"]["acc_z"])
        row.update(summarize_series("motion", pd.Series(mag)))
    if "IBI" in signals:
        row.update(hrv_features(signals["IBI"]["ibi"]))
    if "BVP" in signals:
        sample_rate = metadata["BVP"]["sample_rate"]
        bvp = signals["BVP"]["bvp"].to_numpy()
        row.update({f"bvp_{k}": v for k, v in pulse_features(bvp, sample_rate).items()})
        row["bvp_snr"] = snr_estimate(bvp)
    return row


def build_feature_table(archive_dir: str | Path) -> pd.DataFrame:
    rows = [build_trial_features(record) for record in scan_archive(archive_dir)]
    features = pd.DataFrame(rows)
    return add_pseudo_targets(features)


def save_feature_table(archive_dir: str | Path, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features = build_feature_table(archive_dir)
    features.to_csv(output_path, index=False)
    return output_path
