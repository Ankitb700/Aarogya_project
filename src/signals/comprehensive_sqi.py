from __future__ import annotations

import numpy as np
from scipy import signal, stats
from scipy.spatial.distance import euclidean


def n_sqi(values: np.ndarray) -> float:
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


def s_sqi(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 4:
        return np.nan
    return float(stats.skew(values))


def k_sqi(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 4:
        return np.nan
    return float(stats.kurtosis(values, fisher=False))


def e_sqi(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 4:
        return np.nan
    power = values ** 2
    power = power / (np.sum(power) + 1e-12)
    return float(-np.sum(power * np.log2(power + 1e-12)))


def z_sqi(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 4:
        return np.nan
    zero_crossings = np.sum(np.diff(np.sign(values)) != 0)
    return float(zero_crossings / len(values))


def r_sqi(values: np.ndarray, sample_rate: float, low_hz: float = 1.0, high_hz: float = 2.25) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 4:
        return np.nan
    freqs, psd = signal.welch(values, fs=sample_rate, nperseg=min(256, len(values)))
    heart_band = (freqs >= low_hz) & (freqs <= high_hz)
    total_band = freqs <= 8.0
    heart_power = np.sum(psd[heart_band]) if heart_band.any() else 0
    total_power = np.sum(psd[total_band]) if total_band.any() else 1
    if total_power <= 0:
        return np.nan
    return float(heart_power / total_power)


def m_sqi(values: np.ndarray, sample_rate: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 4:
        return np.nan
    min_dist = int(sample_rate * 0.35)
    min_prom = np.nanstd(values) * 0.1
    peaks1, props1 = signal.find_peaks(values, distance=min_dist, prominence=min_prom)
    peaks2, props2 = signal.find_peaks(values, distance=int(sample_rate * 0.3), prominence=min_prom * 0.5)
    if len(peaks1) == 0:
        return np.nan
    overlap = len(set(peaks1) & set(peaks2))
    return float(overlap / len(peaks1))


def periodicity(values: np.ndarray, sample_rate: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 4:
        return np.nan
    min_dist = int(sample_rate * 0.35)
    min_prom = np.nanstd(values) * 0.1
    peaks, _ = signal.find_peaks(values, distance=min_dist, prominence=min_prom)
    if len(peaks) < 2:
        return np.nan
    ibi = np.diff(peaks) / sample_rate
    if len(ibi) < 2:
        return np.nan
    ibi_var = np.var(ibi)
    ibi_mean = np.mean(ibi)
    if ibi_mean <= 0:
        return np.nan
    return float(1.0 - ibi_var / (ibi_mean ** 2 + 1e-12))


def peak_stability(values: np.ndarray, sample_rate: float, window_seconds: float = 5.0) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < int(sample_rate * window_seconds * 2):
        return np.nan
    window_samples = int(sample_rate * window_seconds)
    freqs_list = []
    for start in range(0, len(values) - window_samples, window_samples // 2):
        segment = values[start:start + window_samples]
        if np.std(segment) < 1e-8:
            continue
        seg_freqs, seg_psd = signal.welch(segment, fs=sample_rate, nperseg=min(128, len(segment)))
        hr_band = (seg_freqs >= 0.7) & (seg_freqs <= 4.0)
        if not hr_band.any():
            continue
        freqs_list.append(seg_freqs[hr_band][np.argmax(seg_psd[hr_band])])
    if len(freqs_list) < 2:
        return np.nan
    return float(-np.std(freqs_list))


def pulse_consistency(values: np.ndarray, sample_rate: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 4:
        return np.nan
    min_dist = int(sample_rate * 0.35)
    min_prom = np.nanstd(values) * 0.1
    peaks, _ = signal.find_peaks(values, distance=min_dist, prominence=min_prom)
    if len(peaks) < 3:
        return np.nan
    beat_length = int(np.mean(np.diff(peaks)))
    if beat_length < 2:
        return np.nan
    cycles = []
    for i in range(len(peaks) - 1):
        start, end = peaks[i], peaks[i] + beat_length
        if end < len(values):
            cycle = values[start:end]
            if len(cycle) > 0:
                cycles.append(cycle)
    if len(cycles) < 2:
        return np.nan
    min_len = min(len(c) for c in cycles)
    aligned = np.array([c[:min_len] for c in cycles])
    corrs = []
    for i in range(len(aligned) - 1):
        cc = np.corrcoef(aligned[i], aligned[i + 1])[0, 1]
        if np.isfinite(cc):
            corrs.append(cc)
    if not corrs:
        return np.nan
    return float(np.mean(corrs))


def compute_all_sqis(values: np.ndarray, sample_rate: float, prefix: str = "") -> dict[str, float]:
    sqis = {
        "n_sqi": n_sqi(values),
        "s_sqi": s_sqi(values),
        "k_sqi": k_sqi(values),
        "e_sqi": e_sqi(values),
        "z_sqi": z_sqi(values),
        "r_sqi": r_sqi(values, sample_rate),
        "m_sqi": m_sqi(values, sample_rate),
        "periodicity": periodicity(values, sample_rate),
        "peak_stability": peak_stability(values, sample_rate),
        "pulse_consistency": pulse_consistency(values, sample_rate),
    }
    if prefix:
        sqis = {f"{prefix}_{k}": v for k, v in sqis.items()}
    return sqis


def classify_signal_quality(sqis: dict[str, float]) -> str:
    def _get(key):
        for k, v in sqis.items():
            if k.endswith(key):
                return v
        return np.nan

    n = _get("n_sqi")
    s = _get("s_sqi")
    k = _get("k_sqi")
    r = _get("r_sqi")
    p = _get("periodicity")
    good_count = 0
    total = 0
    if np.isfinite(n):
        total += 1
        if n > 10:
            good_count += 1
    if np.isfinite(s):
        total += 1
        if abs(s) < 2.0:
            good_count += 1
    if np.isfinite(k):
        total += 1
        if k > 3.0:
            good_count += 1
    if np.isfinite(r):
        total += 1
        if r > 0.4:
            good_count += 1
    if np.isfinite(p):
        total += 1
        if p > 0.4:
            good_count += 1
    if total == 0:
        return "Unfit"
    score = good_count / total
    if score >= 0.8:
        return "Excellent"
    elif score >= 0.5:
        return "Acceptable"
    else:
        return "Unfit"


def extract_sqi_features(values: np.ndarray, sample_rate: float) -> np.ndarray:
    sqis = compute_all_sqis(values, sample_rate)
    features = []
    for key in ["n_sqi", "s_sqi", "k_sqi", "e_sqi", "z_sqi", "r_sqi", "periodicity", "peak_stability", "pulse_consistency"]:
        v = sqis.get(key, np.nan)
        features.append(v if np.isfinite(v) else 0.0)
    return np.array(features, dtype=float)
