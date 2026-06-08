from __future__ import annotations

import numpy as np
from scipy import signal


def detrend(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return values - np.nanmean(values)
    return signal.detrend(values)


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std == 0:
        return values * 0
    return (values - mean) / std


def butter_bandpass(values: np.ndarray, sample_rate: float, low_hz: float = 0.7, high_hz: float = 4.0, order: int = 3) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if valid.sum() < max(order * 6, 12):
        return values.copy()
    filled = values.copy()
    if not valid.all():
        x = np.arange(len(values))
        filled[~valid] = np.interp(x[~valid], x[valid], values[valid])
    nyquist = sample_rate / 2.0
    high_hz = min(high_hz, nyquist * 0.95)
    if low_hz <= 0 or high_hz <= low_hz:
        return filled
    b, a = signal.butter(order, [low_hz / nyquist, high_hz / nyquist], btype="band")
    padlen = 3 * max(len(a), len(b))
    if len(filled) <= padlen:
        return filled
    return signal.filtfilt(b, a, filled)


def dominant_frequency(values: np.ndarray, sample_rate: float, low_hz: float = 0.7, high_hz: float = 4.0) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 4:
        return np.nan
    freqs, psd = signal.welch(values, fs=sample_rate, nperseg=min(256, len(values)))
    band = (freqs >= low_hz) & (freqs <= high_hz)
    if not band.any():
        return np.nan
    return float(freqs[band][np.argmax(psd[band])])


def spectral_entropy(values: np.ndarray, sample_rate: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 4:
        return np.nan
    _, psd = signal.welch(values, fs=sample_rate, nperseg=min(256, len(values)))
    total = np.sum(psd)
    if total <= 0:
        return np.nan
    p = psd / total
    return float(-np.sum(p * np.log2(p + 1e-12)))
