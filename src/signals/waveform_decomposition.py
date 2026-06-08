from __future__ import annotations

import numpy as np
from scipy import signal, ndimage
from scipy.signal import find_peaks, peak_widths


def second_derivative_ppg(values: np.ndarray, sample_rate: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    dt = 1.0 / sample_rate
    first = np.gradient(values, dt)
    second = np.gradient(first, dt)
    return second


def detect_sdppg_peaks(sdppg: np.ndarray, sample_rate: float) -> dict[str, np.ndarray]:
    sdppg = np.asarray(sdppg, dtype=float)
    min_dist = int(sample_rate * 0.1)
    prom = np.nanstd(sdppg) * 0.3
    if not np.isfinite(prom) or prom == 0:
        prom = 0.01
    all_peaks, props = find_peaks(sdppg, distance=min_dist, prominence=prom)
    all_troughs, _ = find_peaks(-sdppg, distance=min_dist, prominence=prom)
    if len(all_peaks) == 0:
        return {"a": np.array([]), "b": np.array([]), "c": np.array([]), "d": np.array([]), "e": np.array([])}
    prominences = props.get("prominences", np.ones(len(all_peaks)))
    sorted_idx = np.argsort(prominences)[::-1]
    a_peak_idx = sorted_idx[0] if len(sorted_idx) > 0 else -1
    a_peak = np.array([all_peaks[a_peak_idx]]) if a_peak_idx >= 0 else np.array([])
    remaining = np.delete(all_peaks, a_peak_idx) if len(all_peaks) > 1 else np.array([])
    remaining_prom = np.delete(prominences, a_peak_idx) if len(prominences) > 1 else np.array([])
    sorted_remaining = np.argsort(remaining_prom)[::-1] if len(remaining_prom) > 0 else np.array([])
    c_peak = np.array([remaining[sorted_remaining[0]]]) if len(sorted_remaining) > 0 and len(remaining) > 0 else np.array([])
    remaining2 = np.delete(remaining, sorted_remaining[0]) if len(sorted_remaining) > 0 and len(remaining) > 1 else remaining
    e_peak = np.array([remaining2[-1]]) if len(remaining2) > 0 else np.array([])
    b_trough = np.array([])
    d_trough = np.array([])
    if len(a_peak) > 0:
        a_pos = a_peak[0]
        b_candidates = all_troughs[all_troughs < a_pos]
        b_trough = np.array([b_candidates[-1]]) if len(b_candidates) > 0 else np.array([])
        d_candidates = all_troughs[all_troughs > a_pos]
        d_trough = np.array([d_candidates[0]]) if len(d_candidates) > 0 else np.array([])
    return {"a": a_peak, "b": b_trough, "c": c_peak, "d": d_trough, "e": e_peak}


def detect_fiducial_points(values: np.ndarray, sample_rate: float) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=float)
    min_dist = int(sample_rate * 0.35)
    prom = np.nanstd(values) * 0.15
    peaks, _ = find_peaks(values, distance=min_dist, prominence=prom)
    troughs, _ = find_peaks(-values, distance=min_dist, prominence=prom)
    systolic_peaks = peaks
    diastolic_notches = np.array([])
    diastolic_peaks = np.array([])
    if len(peaks) >= 2:
        systolic_peaks = np.array([peaks[0]])
        diastolic_peaks = np.array([peaks[1]])
        if len(troughs) > 0:
            notch_candidates = troughs[(troughs > peaks[0]) & (troughs < peaks[1])]
            diastolic_notches = notch_candidates if len(notch_candidates) > 0 else np.array([])
    return {
        "systolic_peaks": systolic_peaks,
        "diastolic_notches": diastolic_notches,
        "diastolic_peaks": diastolic_peaks,
    }


def compute_augmentation_index(values: np.ndarray, sample_rate: float) -> float:
    values = np.asarray(values, dtype=float)
    min_dist = int(sample_rate * 0.35)
    prom = np.nanstd(values) * 0.15
    peaks, props = find_peaks(values, distance=min_dist, prominence=prom)
    if len(peaks) < 2:
        return np.nan
    prominences = props.get("prominences", np.ones(len(peaks)))
    sorted_idx = np.argsort(prominences)[::-1]
    p1 = prominences[sorted_idx[0]]
    p2 = prominences[sorted_idx[1]] if len(sorted_idx) > 1 else p1
    if p1 <= 0:
        return np.nan
    return float(p2 / p1)


def compute_compliance_proxy(values: np.ndarray, sample_rate: float) -> float:
    values = np.asarray(values, dtype=float)
    min_dist = int(sample_rate * 0.35)
    prom = np.nanstd(values) * 0.15
    peaks, _ = find_peaks(values, distance=min_dist, prominence=prom)
    if len(peaks) < 2:
        return np.nan
    first_peak = peaks[0]
    end_systole = first_peak + int(sample_rate * 0.3)
    if end_systole >= len(values):
        end_systole = len(values) - 1
    pulse_area = np.trapezoid(values[first_peak:end_systole] - np.min(values[:end_systole]))
    pulse_height = np.max(values[first_peak:end_systole]) - np.min(values[:end_systole])
    if pulse_height <= 0:
        return np.nan
    compliance = pulse_area / pulse_height
    return float(compliance) if np.isfinite(compliance) else np.nan


def compute_resistance_proxy(values: np.ndarray, sample_rate: float) -> float:
    values = np.asarray(values, dtype=float)
    min_dist = int(sample_rate * 0.35)
    prom = np.nanstd(values) * 0.15
    peaks, _ = find_peaks(values, distance=min_dist, prominence=prom)
    if len(peaks) < 2:
        return np.nan
    systolic_peak_val = values[peaks[0]]
    trough_before = max(0, peaks[0] - int(sample_rate * 0.15))
    pulse_amp = systolic_peak_val - values[trough_before]
    if pulse_amp <= 0:
        return np.nan
    diastolic_start = peaks[0] + int(sample_rate * 0.15)
    diastolic_end = min(peaks[1], len(values) - 1)
    if diastolic_start >= diastolic_end:
        return np.nan
    decay = values[diastolic_start:diastolic_end]
    if len(decay) < 2:
        return np.nan
    decay_frac = (decay[0] - decay[-1]) / pulse_amp
    decay_time = len(decay) / sample_rate
    decay_rate = decay_frac / decay_time if decay_time > 0 else np.nan
    if not np.isfinite(decay_rate) or decay_rate <= 0:
        return pulse_amp / (1.0 + diastolic_start / sample_rate)
    return float(pulse_amp / decay_rate)


def compute_stiffness_index(values: np.ndarray, sample_rate: float) -> float:
    values = np.asarray(values, dtype=float)
    min_dist = int(sample_rate * 0.35)
    prom = np.nanstd(values) * 0.15
    peaks, _ = find_peaks(values, distance=min_dist, prominence=prom)
    if len(peaks) < 2:
        return np.nan
    systolic_peak_idx = peaks[0]
    troughs, _ = find_peaks(-values[:peaks[1]], distance=int(sample_rate * 0.1))
    onset = 0
    if len(troughs) > 0:
        onset_candidates = troughs[troughs < systolic_peak_idx]
        if len(onset_candidates) > 0:
            onset = onset_candidates[-1]
    systolic_amp = values[systolic_peak_idx] - values[onset]
    rise_time = (systolic_peak_idx - onset) / sample_rate
    if rise_time <= 0 or systolic_amp <= 0:
        return np.nan
    return float(systolic_amp / rise_time)


def compute_pulse_area_ratio(values: np.ndarray, sample_rate: float) -> float:
    values = np.asarray(values, dtype=float)
    min_dist = int(sample_rate * 0.35)
    prom = np.nanstd(values) * 0.15
    peaks, _ = find_peaks(values, distance=min_dist, prominence=prom)
    if len(peaks) < 2:
        return np.nan
    onset = max(0, peaks[0] - int(sample_rate * 0.1))
    end_cycle = peaks[1]
    cycle = values[onset:end_cycle]
    if len(cycle) < 4:
        return np.nan
    notch_idx = np.argmin(cycle[len(cycle) // 3: 2 * len(cycle) // 3]) + len(cycle) // 3
    if notch_idx >= len(cycle):
        return np.nan
    baseline = np.min(cycle)
    systolic_area = np.trapezoid(np.maximum(cycle[:notch_idx] - baseline, 0))
    diastolic_area = np.trapezoid(np.maximum(cycle[notch_idx:] - baseline, 0))
    if diastolic_area <= 0:
        return np.nan
    return float(systolic_area / diastolic_area)


def compute_windkessel_features(values: np.ndarray, sample_rate: float) -> dict[str, float]:
    return {
        "augmentation_index": compute_augmentation_index(values, sample_rate),
        "compliance_proxy": compute_compliance_proxy(values, sample_rate),
        "resistance_proxy": compute_resistance_proxy(values, sample_rate),
        "stiffness_index": compute_stiffness_index(values, sample_rate),
        "pulse_area_ratio": compute_pulse_area_ratio(values, sample_rate),
    }


def wavelet_denoise(values: np.ndarray, sample_rate: float, wavelet: str = "db4", level: int = 4) -> np.ndarray:
    import pywt
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 ** level:
        return values
    coeffs = pywt.wavedec(values, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(values)))
    coeffs_thresh = [coeffs[0]]
    for i in range(1, len(coeffs)):
        coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode="soft"))
    return pywt.waverec(coeffs_thresh, wavelet)[:len(values)]


def emd_decompose(values: np.ndarray, max_imfs: int = 5) -> list[np.ndarray]:
    try:
        from PyEMD import EMD
        emd = EMD()
        imfs = emd(np.asarray(values, dtype=float))
        if imfs.ndim == 1:
            imfs = imfs.reshape(1, -1)
        return [imfs[i] for i in range(min(len(imfs), max_imfs))]
    except ImportError:
        return []


def hilbert_envelope(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.abs(signal.hilbert(values))


def hilbert_instantaneous_phase(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.unwrap(np.angle(signal.hilbert(values)))


def hilbert_instantaneous_frequency(values: np.ndarray, sample_rate: float) -> np.ndarray:
    phase = hilbert_instantaneous_phase(values)
    return np.diff(phase) / (2.0 * np.pi) * sample_rate
