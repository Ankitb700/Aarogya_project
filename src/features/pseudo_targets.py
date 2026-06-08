from __future__ import annotations

import numpy as np
import pandas as pd


def minmax(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    low = series.min()
    high = series.max()
    if not np.isfinite(low) or not np.isfinite(high) or high == low:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - low) / (high - low)


def add_pseudo_targets(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    hr_norm = minmax(df.get("hr_mean", pd.Series(index=df.index, dtype=float)))
    low_hrv = 1 - minmax(df.get("rmssd", pd.Series(index=df.index, dtype=float)))
    eda_norm = minmax(df.get("eda_mean", pd.Series(index=df.index, dtype=float)))
    motion_norm = minmax(df.get("motion_mean", pd.Series(index=df.index, dtype=float)))
    df["cv_load_index"] = 0.45 * hr_norm + 0.25 * low_hrv + 0.2 * eda_norm + 0.1 * motion_norm
    df["cv_load_class"] = pd.cut(df["cv_load_index"], bins=[-0.01, 0.33, 0.66, 1.01], labels=[0, 1, 2]).astype(int)
    snr_norm = minmax(df.get("bvp_snr", pd.Series(index=df.index, dtype=float)))
    optical = df.get("optical_stability_score", pd.Series(0.5, index=df.index))
    df["signal_reliability_score"] = (0.5 * snr_norm + 0.3 * optical + 0.2 * (1 - motion_norm)).clip(0, 1)
    df["stress_recovery_index"] = (0.5 * minmax(df.get("rmssd", pd.Series(index=df.index, dtype=float))) + 0.5 * (1 - eda_norm)).clip(0, 1)
    df["bp_proxy_score"] = (0.55 * df["cv_load_index"] + 0.25 * motion_norm + 0.2 * (1 - df["signal_reliability_score"])).clip(0, 1)
    if "subject_id" in df and "hr_mean" in df:
        baseline = df.groupby("subject_id")["hr_mean"].transform("mean")
        df["hr_baseline_deviation"] = df["hr_mean"] - baseline
    return df
