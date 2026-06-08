# Model Output CSVs — Field Reference

This document describes every column in the two generated feature CSVs located under `outputs/`.

| File | Purpose |
|------|---------|
| `outputs/features/trial_features.csv` | One row per subject × trial — 38 engineered features + derived targets |
| `outputs/features/dl_windows.csv` | 64-second sliding windows (stride 32 s) with aligned CSV paths and per-window targets |

---

## Table 1 — `outputs/features/trial_features.csv`

| # | Column | Type | Description | Derivation | Source file |
|---|--------|------|-------------|------------|-------------|
| 1 | `subject_id` | str | Unique subject identifier | Raw archive folder name (e.g. `subject_001`) | `src/data/archive_scanner.py` |
| 2 | `trial_id` | str | Trial number within the subject | Folder name (e.g. `trial_001`) | `src/data/archive_scanner.py` |
| 3 | `hr_mean` | float | **Average heart rate** across the full trial window (BPM) | `values.mean()` on the `hr` column of the 1 Hz aligned CSV | `src/features/build_features.py:31` |
| 4 | `hr_std` | float | Standard deviation of HR (BPM) | `values.std()` on the `hr` column | `src/features/build_features.py:31` |
| 5 | `hr_min` | float | Minimum observed HR (BPM) | `values.min()` on the `hr` column | `src/features/build_features.py:31` |
| 6 | `hr_max` | float | Maximum observed HR (BPM) | `values.max()` on the `hr` column | `src/features/build_features.py:31` |
| 7 | `eda_mean` | float | **Average electrodermal activity** (μS) | `values.mean()` on the `eda` column | `src/features/build_features.py:33` |
| 8 | `eda_std` | float | SD of EDA (μS) | `values.std()` on the `eda` column | `src/features/build_features.py:33` |
| 9 | `eda_min` | float | Minimum EDA (μS) | `values.min()` on the `eda` column | `src/features/build_features.py:33` |
| 10 | `eda_max` | float | Maximum EDA (μS) | `values.max()` on the `eda` column | `src/features/build_features.py:33` |
| 11 | `temp_mean` | float | **Average skin temperature** (°C) | `values.mean()` on the `temp` column | `src/features/build_features.py:35` |
| 12 | `temp_std` | float | SD of temperature (°C) | `values.std()` on the `temp` column | `src/features/build_features.py:35` |
| 13 | `temp_min` | float | Minimum temperature (°C) | `values.min()` on the `temp` column | `src/features/build_features.py:35` |
| 14 | `temp_max` | float | Maximum temperature (°C) | `values.max()` on the `temp` column | `src/features/build_features.py:35` |
| 15 | `optical_stability_score` | float [0, 1] | **Video skin-ROI stability** — high when temperature time-series is smooth (low CV) | `1 / (1 + cv)` where CV = std / \|mean\| of the `temp` signal | `src/features/build_features.py:36` → `src/signals/quality.py` |
| 16 | `motion_mean` | float | **Mean 3-axis accelerometer magnitude** across trial (g) | `sqrt(acc_x² + acc_y² + acc_z²).mean()` | `src/features/build_features.py:39` → `src/signals/quality.py` |
| 17 | `motion_std` | float | SD of accelerometer magnitude (g) | `values.std()` on the magnitude series | `src/features/build_features.py:39` |
| 18 | `motion_min` | float | Minimum accelerometer magnitude (g) | `values.min()` on the magnitude series | `src/features/build_features.py:39` |
| 19 | `motion_max` | float | Maximum accelerometer magnitude (g) | `values.max()` on the magnitude series | `src/features/build_features.py:39` |
| 20 | `rmssd` | float | **Root Mean Square of Successive Differences** of IBI — parasympathetic nervous system index (ms) | `sqrt(mean(diff(ibi)²))` | `src/features/build_features.py:41` → `src/signals/hrv.py` |
| 21 | `sdnn` | float | **Standard Deviation of NN intervals** — overall HRV (ms) | `pd.Series.std(ddof=1)` on IBI series | `src/features/build_features.py:41` → `src/signals/hrv.py` |
| 22 | `pnn50` | float | **Percentage of successive NN intervals > 50 ms** (0–1) | `mean(abs(diff(ibi)) > 0.05)` | `src/features/build_features.py:41` → `src/signals/hrv.py` |
| 23 | `lf_hf` | float | **LF/HF ratio** — sympathovagal balance. LF = 0.04–0.15 Hz, HF = 0.15–0.4 Hz | Welch PSD on 4 Hz interpolated IBI grid; `lf_band_power / hf_band_power` | `src/features/build_features.py:41` → `src/signals/hrv.py` |
| 24 | `bvp_pulse_amplitude_mean` | float | **Mean peak prominence** in the BVP waveform (arbitrary units) | `find_peaks(prominence ≥ 0.1σ).prominences.mean()` | `src/features/build_features.py:45` → `src/signals/bvp_features.py` |
| 25 | `bvp_pulse_amplitude_std` | float | SD of BVP peak prominence | `find_peaks.prominences.std()` | `src/features/build_features.py:45` → `src/signals/bvp_features.py` |
| 26 | `bvp_pulse_width_mean` | float | **Mean full-width half-maximum pulse duration** (seconds) | `peak_widths(rel_height=0.5) / sample_rate` averaged across peaks | `src/features/build_features.py:45` → `src/signals/bvp_features.py` |
| 27 | `bvp_peak_interval_mean` | float | **Mean inter-peak interval** (seconds) | `diff(peak_indices) / sample_rate` averaged | `src/features/build_features.py:45` → `src/signals/bvp_features.py` |
| 28 | `bvp_peak_interval_std` | float | SD of inter-peak interval (seconds) | `std` of successive peak differences | `src/features/build_features.py:45` → `src/signals/bvp_features.py` |
| 29 | `bvp_pulse_skewness` | float | **Skewness of BVP pulse** — measures waveform asymmetry (positive = right-skewed) | `E[(x−μ)³] / σ³` on detrended BVP | `src/features/build_features.py:45` → `src/signals/bvp_features.py` |
| 30 | `bvp_dominant_frequency` | float | **Dominant frequency of BVP** PSD in HR band 0.7–4 Hz (Hz) | Welch PSD; argmax of freq in [0.7, 4] Hz band | `src/features/build_features.py:45` → `src/signals/bvp_features.py` → `src/signals/filtering.py` |
| 31 | `bvp_spectral_entropy` | float | **Spectral entropy of BVP** — higher means more spectral spread / less periodic | `−Σ(p_i · log₂(p_i))` where p = normalized PSD bins | `src/features/build_features.py:45` → `src/signals/bvp_features.py` → `src/signals/filtering.py` |
| 32 | `bvp_snr` | float | **SNR of BVP signal** in dB = `10·log₁₀(var_signal / var_noise)`. noise = first-difference variance | `10 * log10(var(values) / var(diff(values)))` | `src/features/build_features.py:46` → `src/signals/quality.py` |
| 33 | `cv_load_index` | float [0, 1] | **Cardiovascular load score** — composite index combining HR, HRV, EDA, and motion | `0.45·hr_norm + 0.25·(1−rmssd_norm) + 0.2·eda_norm + 0.1·motion_norm` (all min-max scaled across all trials) | `src/features/pseudo_targets.py:22` |
| 34 | `cv_load_class` | int {0, 1, 2} | **3-tier cardiovascular load class**: 0=low, 1=moderate, 2=elevated | `pd.cut(cv_load_index, bins=[−0.01, 0.33, 0.66, 1.01])` | `src/features/pseudo_targets.py:23` |
| 35 | `signal_reliability_score` | float [0, 1] | **Overall signal trustworthiness** — high when BVP SNR is good, video stability is high, and motion is low | `0.5·snr_norm + 0.3·optical_stability + 0.2·(1−motion_norm)`, clipped to [0,1] | `src/features/pseudo_targets.py:26` |
| 36 | `stress_recovery_index` | float [0, 1] | **Recovery / de-stress indicator** — high when parasympathetic RMSSD is high and EDA is low | `0.5·rmssd_norm + 0.5·(1−eda_norm)`, clipped to [0,1] | `src/features/pseudo_targets.py:27` |
| 37 | `bp_proxy_score` | float [0, 1] | **Blood-pressure proxy score** — synthetic target approximating elevated cardiovascular pressure | `0.55·cv_load_index + 0.25·motion_norm + 0.2·(1−signal_reliability_score)`, clipped to [0,1] | `src/features/pseudo_targets.py:28` |
| 38 | `hr_baseline_deviation` | float (BPM) | **Subject-personalised HR offset** — current HR minus this subject's trial mean HR | `hr_mean − subject_mean_hr` (grouped by `subject_id`) | `src/features/pseudo_targets.py:30–31` |

---

## Table 2 — `outputs/features/dl_windows.csv`

| # | Column | Type | Description | Derivation | Source file |
|---|--------|------|-------------|------------|-------------|
| 1 | `subject_id` | str | Same as Table 1 | Copied from aligned CSV row 0 | `src/models/datasets.py:72` |
| 2 | `trial_id` | str | Same as Table 1 | Copied from aligned CSV row 0 | `src/models/datasets.py:73` |
| 3 | `aligned_path` | str | Absolute path to the 1 Hz per-trial aligned CSV | `outputs/aligned/{subject}_{trial}_aligned_1hz.csv` | `src/models/datasets.py:82` |
| 4 | `start_row` | int | First row index (inclusive) of the 64-second window inside the aligned CSV | Window loop: `range(0, n_rows−64+1, stride=32)` | `src/models/datasets.py:74` |
| 5 | `end_row` | int | Last row index (exclusive) of the window | `start_row + 64` (clipped to CSV length) | `src/models/datasets.py:75` |
| 6 | `video_path` | str | Absolute path to the source `.MOV` video for this trial | Matched from `archive/{subject}/{subject}/{trial}/video/*.MOV` | `src/models/datasets.py:85` → `src/data/archive_scanner.py` |
| 7 | `hr_target` | float | **Window-average HR** (BPM) — training target for the HR regression head | `mean(hr)` over rows [start_row, end_row) in the aligned CSV | `src/models/datasets.py:97` |
| 8 | `cv_load_index` | float | **Cardiovascular load index** for this trial (same value across all windows of the same trial) | Merged from `trial_features.csv` via subject_id × trial_id join | `src/models/datasets.py:105` |
| 9 | `cv_load_class` | int | **3-class cardiovascular load label** for this trial (same across all windows of the same trial) | Merged from `trial_features.csv` via subject_id × trial_id join | `src/models/datasets.py:106` |
| 10 | `bp_proxy_target` | float | **BP proxy score** for this trial (same across all windows of the same trial) | Merged from `trial_features.csv` via subject_id × trial_id join | `src/models/datasets.py:107` |

---

## How the Two Tables Are Related

```
trial_features.csv    ← one row per (subject_id, trial_id), trial-level stats
        │
        │  joined on [subject_id, trial_id]
        ▼
dl_windows.csv        ← multiple sliding-window rows per trial,
                         each referencing a [start_row, end_row) slice
                         of the aligned 1 Hz CSV + 3 per-trial targets copied
                         from trial_features.csv
```

`dl_windows.csv` is built by `create_window_index()` (`src/models/datasets.py:53`) then enriched with target columns by `add_window_targets()` (`src/models/datasets.py:91`). The enriched table is written by notebook 6 at `src/notebooks/06_deep_learning_pipeline.ipynb`.

---

## Derived / Pseudo Targets — Shared Naming Note

The columns `cv_load_index`, `cv_load_class`, `signal_reliability_score`, `stress_recovery_index`, `bp_proxy_score`, and `hr_baseline_deviation` are all created in a **single pass** by `add_pseudo_targets()` in `src/features/pseudo_targets.py`. They use subject-level baselines derived from the same `trial_features.csv` table, so they are valid for the current cohort but scale poorly to out-of-cohort subjects without recalibration.

---

## Data Flow Summary (all output files)

```
archive/  (raw Empatica E4 CSVs + .MOV videos)
   │
   │  load_trial_signals()          ← empatica_loader.py
   │  scan_archive()                ← archive_scanner.py
   │  align_trial()                 ← alignment.py  (resample + interpolate to 1 Hz)
   │
   ▼
outputs/aligned/*_aligned_1hz.csv  (one 1 Hz CSV per trial)
   │
   │  build_trial_features()        ← build_features.py
   │  [resample signal stats, IBI→HRV, BVP→pulse features]
   │  add_pseudo_targets()          ← pseudo_targets.py
   │
   ▼
outputs/features/trial_features.csv
   │
   │  create_window_index()         ← datasets.py  (64 s windows, stride 32 s)
   │  add_window_targets()          ← datasets.py
   │
   ▼
outputs/features/dl_windows.csv
```

---

*Schema version: generated from `outputs/features/trial_features.csv` (8 rows, 38 columns) and `outputs/features/dl_windows.csv` (74 rows, 11 columns).*
*Last updated: pipeline committed after notebook 6 (`06_deep_learning_pipeline.ipynb`).*
