from __future__ import annotations

import numpy as np


def motion_magnitude(acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray) -> np.ndarray:
    return np.sqrt(np.asarray(acc_x, dtype=float) ** 2 + np.asarray(acc_y, dtype=float) ** 2 + np.asarray(acc_z, dtype=float) ** 2)


def snr_estimate(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 4:
        return np.nan
    signal_power = np.nanvar(values)
    noise = np.diff(values)
    noise_power = np.nanvar(noise)
    if noise_power <= 0:
        return np.nan
    return float(10 * np.log10(signal_power / noise_power))


def stability_score(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    cv = np.nanstd(values) / (abs(np.nanmean(values)) + 1e-12)
    return float(1.0 / (1.0 + cv))
