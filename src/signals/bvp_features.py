from __future__ import annotations

import numpy as np
from scipy import signal

from .filtering import dominant_frequency, spectral_entropy


def pulse_features(values: np.ndarray, sample_rate: float) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    clean = values[np.isfinite(values)]
    if len(clean) < 4:
        return {
            "pulse_amplitude_mean": np.nan,
            "pulse_amplitude_std": np.nan,
            "pulse_width_mean": np.nan,
            "peak_interval_mean": np.nan,
            "peak_interval_std": np.nan,
            "pulse_skewness": np.nan,
            "dominant_frequency": np.nan,
            "spectral_entropy": np.nan,
        }
    distance = max(1, int(sample_rate * 0.35))
    peaks, props = signal.find_peaks(clean, distance=distance, prominence=np.nanstd(clean) * 0.1)
    widths = signal.peak_widths(clean, peaks, rel_height=0.5)[0] / sample_rate if len(peaks) else np.array([])
    intervals = np.diff(peaks) / sample_rate if len(peaks) > 1 else np.array([])
    centered = clean - np.mean(clean)
    skew = np.mean(centered**3) / (np.std(clean) ** 3 + 1e-12)
    return {
        "pulse_amplitude_mean": float(np.nanmean(props.get("prominences", [np.nan]))),
        "pulse_amplitude_std": float(np.nanstd(props.get("prominences", [np.nan]))),
        "pulse_width_mean": float(np.nanmean(widths)) if len(widths) else np.nan,
        "peak_interval_mean": float(np.nanmean(intervals)) if len(intervals) else np.nan,
        "peak_interval_std": float(np.nanstd(intervals)) if len(intervals) else np.nan,
        "pulse_skewness": float(skew),
        "dominant_frequency": dominant_frequency(clean, sample_rate),
        "spectral_entropy": spectral_entropy(clean, sample_rate),
    }
