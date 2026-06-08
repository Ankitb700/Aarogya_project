# Prediction CSVs — Column Reference

Files in `outputs/predictions/`:

| File | Rows | Origin |
|------|------|--------|
| `dl_random_sample_predictions.csv` | 8 windows | Notebook `09_dl_prediction.ipynb` → `predict_dl_samples()` |
| `ml_random_sample_predictions.csv` | 2 trials | Notebook `08_ml_prediction.ipynb` → `predict_ml_samples()` |

Both tables were created by:
1. Loading a trained model (`.joblib` or `.pt` checkpoint)
2. Drawing a **random sample from the held-out test set** (80/20 split, `random_state=42`, stratified by `cv_load_class` where counts allow)
3. Running a forward pass or `.predict()` call
4. Collecting actuals, predictions, and auxiliary outputs into a single row per sample

The split function is `split_80_20()` in `src/models/prediction.py:16`; sampling is `random_sample()` at `src/models/prediction.py:26`.

---

## Table 1 — `outputs/predictions/dl_random_sample_predictions.csv` (15 columns, 8 rows)

Each row = **one 64-second, 1 Hz window** (stride 32 s) from the DL test set.

| # | Column | Type | Representation | Derivation | Source file |
|---|--------|------|----------------|------------|-------------|
| 1 | `subject_id` | str | Subject identifier matching the aligned CSV | Copied from the `subject_id` column of the windowed `dl_windows.csv` row | `src/models/prediction.py:116` |
| 2 | `trial_id` | str | Trial identifier matching the aligned CSV | Copied from the `trial_id` column of the windowed `dl_windows.csv` row | `src/models/prediction.py:117` |
| 3 | `start_row` | int | First row index (**inclusive**) of this window slice inside the 1 Hz aligned CSV | Copied from `start_row` in `dl_windows.csv` | `src/models/prediction.py:118` |
| 4 | `end_row` | int | Last row index (**exclusive**) of this window slice inside the aligned CSV | Copied from `end_row` in `dl_windows.csv` | `src/models/prediction.py:119` |
| 5 | `actual_hr` | float (BPM) | **Ground-truth average HR** for the 64 s window | Mean of the `hr` column over aligned rows [start_row, end_row) — read from the raw un-scaled tensor `hr_target_raw` carried by `AlignedWindowDataset.__getitem__` | `src/models/prediction.py:120` → `src/models/datasets.py:316–327` |
| 6 | `predicted_hr` | float (BPM) | **Model-predicted HR** for the window — in real-world BPM units | Raw neural-network output on the HR head passed back through `inverse_standard_scaler()` using the target mean/std stored in the `.pt` checkpoint during training | `src/models/prediction.py:107,121` → `src/models/datasets.py:64–71` |
| 7 | `predicted_hr_model_scale` | float (normalised) | **HR head output before inverse-scaling** — directly comparable to the z-scored training target | `outputs["hr"][i]`, i.e. `MultimodalPhysiologyNet.hr_head(fused)` — a z-scored residual, not yet un-transformed | `src/models/prediction.py:122` |
| 8 | `actual_cv_load_class` | int {0, 1, 2} | **Ground-truth 3-class cardiovascular load** for this trial | `cv_load_class` column of `trial_features.csv` (same for every window of the same trial) — 0 = low, 1 = moderate, 2 = elevated | `src/models/prediction.py:123` → `src/features/pseudo_targets.py:23` |
| 9 | `predicted_cv_load_class` | int {0, 1, 2} | **Model-predicted cardiovascular load class** for the window | `argmax` over `softmax(cv_load_logits)` — the most probable of the 3 classes | `src/models/prediction.py:110,124` |
| 10 | `actual_bp_proxy` | float [0, 1] | **Ground-truth BP proxy score** for the trial | `bp_proxy_score` column of `trial_features.csv` (same for every window of the same trial) | `src/models/prediction.py:125` → `src/features/pseudo_targets.py:28` |
| 11 | `predicted_bp_proxy` | float [0, 1] | **Model-predicted BP proxy score** — in real-world 0–1 units | Raw BP proxy head output passed back through `inverse_standard_scaler()` using the per-target training statistics saved in the `.pt` checkpoint | `src/models/prediction.py:108,126` → `src/models/datasets.py:64–71` |
| 12 | `predicted_bp_proxy_model_scale` | float (normalised) | **BP proxy head output before inverse-scaling** — directly comparable to the z-scored training target | `outputs["bp_proxy"][i]`, i.e. `MultimodalPhysiologyNet.bp_proxy_head(fused)` | `src/models/prediction.py:127` |
| 13 | `predicted_cv_load_proba_0` | float [0, 1] | **Predicted probability of class 0** (low cardiovascular load) | `softmax(cv_load_logits)[class_idx=0]` | `src/models/prediction.py:109,129–130` |
| 14 | `predicted_cv_load_proba_1` | float [0, 1] | **Predicted probability of class 1** (moderate cardiovascular load) | `softmax(cv_load_logits)[class_idx=1]` | `src/models/prediction.py:109,129–130` |
| 15 | `predicted_cv_load_proba_2` | float [0, 1] | **Predicted probability of class 2** (elevated cardiovascular load) | `softmax(cv_load_logits)[class_idx=2]` | `src/models/prediction.py:109,129–130` |

> **`_model_scale` pair semantics**: columns 7 and 12 hold the network's raw linear outputs, which are z-scored with respect to the training-target statistics. Columns 6 and 11 hold the same values after `inverse_standard_scaler()` reconstructs the original BPM / 0–1 range. Inspect `_model_scale` values alongside the checkpoint's saved `target_scalers` dict (`outputs/models/multimodal_physio_net.pt`) to verify the scaler parameters.

---

## Table 2 — `outputs/predictions/ml_random_sample_predictions.csv` (11 columns, 2 rows)

Each row = **one trial** (subject × trial) sampled from the ML test set.

| # | Column | Type | Representation | Derivation | Source file |
|---|--------|------|----------------|------------|-------------|
| 1 | `subject_id` | str | Subject identifier | Copied from `subject_id` column of the sample DataFrame | `src/models/prediction.py:37` |
| 2 | `trial_id` | str | Trial identifier | Copied from `trial_id` column of the sample DataFrame | `src/models/prediction.py:37` |
| 3 | `actual_hr` | float (BPM) | **Ground-truth average HR** for the trial | `hr_mean` column of `trial_features.csv` | `src/models/prediction.py:39,45` |
| 4 | `actual_cv_load` | int {0, 1, 2} | **Ground-truth cardiovascular load class** for the trial | `cv_load_class` column of `trial_features.csv` — 0 = low, 1 = moderate, 2 = elevated | `src/models/prediction.py:40,45` |
| 5 | `actual_bp_proxy` | float [0, 1] | **Ground-truth BP proxy score** for the trial | `bp_proxy_score` column of `trial_features.csv` | `src/models/prediction.py:41,45` |
| 6 | `predicted_hr` | float (BPM) | **HR regression prediction** from the XGBoost `hr_regressor.joblib` model | `hr_regressor.predict(X)` on the 38-feature row; output is in BPM | `src/models/prediction.py:48,56,57` |
| 7 | `predicted_cv_load` | int {0, 1, 2} | **CV load classification prediction** from the XGBoost `cv_load_classifier.joblib` model | `cv_load_classifier.predict(X)` on the 38-feature row; returns the most probable class | `src/models/prediction.py:48,56,57` |
| 8 | `predicted_cv_load_proba_0` | float [0, 1] | **Predicted probability of class 0** (low CV load) | `cv_load_classifier.predict_proba(X)[:, 0]` | `src/models/prediction.py:58–61` |
| 9 | `predicted_cv_load_proba_1` | float [0, 1] | **Predicted probability of class 1** (moderate CV load) | `cv_load_classifier.predict_proba(X)[:, 1]` | `src/models/prediction.py:58–61` |
| 10 | `predicted_cv_load_proba_2` | float [0, 1] | **Predicted probability of class 2** (elevated CV load) | `cv_load_classifier.predict_proba(X)[:, 2]` | `src/models/prediction.py:58–61` |
| 11 | `predicted_bp_proxy` | float [0, 1] | **BP proxy score regression prediction** from the XGBoost `bp_proxy_regressor.joblib` model | `bp_proxy_regressor.predict(X)` on the 38-feature row | `src/models/prediction.py:48,56,57` |

---

## Model-Output Difference at a Glance

| Dimension | ML predictions | DL predictions |
|-----------|---------------|----------------|
| **Prediction unit** | Trial (one row per subject × trial) | Window (one row per 64 s slice) |
| **Primary input** | 38 pre-computed engineered features from `trial_features.csv` | Aligned 1 Hz CSV window + video frames (if `use_video=True`) + raw BVP segment |
| **HR output** | `predicted_hr` — already in BPM | `predicted_hr` — un-normalised BPM; `predicted_hr_model_scale` — z-scored head output |
| **CV load output** | `predicted_cv_load` (class label) + `predicted_cv_load_proba_{0,1,2}` | `predicted_cv_load_class` (class label) + `predicted_cv_load_proba_{0,1,2}` |
| **BP proxy output** | `predicted_bp_proxy` — already in [0, 1] | `predicted_bp_proxy` — un-normalised [0, 1]; `predicted_bp_proxy_model_scale` — z-scored head output |
| **Scale reversal** | Not needed — XGBoost operates on raw feature scale (after StandardScaler inverse) | Applied via `inverse_standard_scaler()` — see `src/models/datasets.py:64–71` |
| **Source model** | `hr_regressor.joblib`, `cv_load_classifier.joblib`, `bp_proxy_regressor.joblib` | `multimodal_physio_net.pt` + checkpoint scalers |
| **Random seed** | `random_state=7` (8 samples drawn from 80/20 split) | `random_state=7` (8 samples drawn from 80/20 split) |

---

## Critical Notes on the `_model_scale` columns (DL only)

`predicted_hr_model_scale` and `predicted_bp_proxy_model_scale` exist in the DL table but **not** in the ML table because:

- The ML pipeline uses a scikit-learn `Pipeline` with a `StandardScaler` step. The `.predict()` call already applies the inverse transform before returning, so only the final real-world value is available.
- The DL checkpoint saves `target_scalers` (mean and std per target computed over the training set). `inverse_standard_scaler()` in `src/models/datasets.py:64–71` manually reverses this transform. It is called once for the `predicted_hr`/`predicted_bp_proxy` columns (`src/models/prediction.py:107–108`) and **not** for the `_model_scale` columns (`src/models/prediction.py:122,127`), so those remain in the training z-score space.

If you need the exact scaler values to reconstruct real-world values from `_model_scale`:

```python
import torch
ckpt = torch.load("outputs/models/multimodal_physio_net.pt", map_location="cpu", weights_only=False)
hr_scaler      = ckpt["target_scalers"]["hr"]          # {"mean": float, "std": float}
bp_proxy_scaler = ckpt["target_scalers"]["bp_proxy"]

# inverse_standard_scaler applies:  x_real = x_scaled * std + mean
# source: src/models/datasets.py:64–71
```

---

## Key Limitation (both tables)

Both prediction files are drawn from a **non-stratified 80/20 split of only 8 trials total** (DL: 8-window sample; ML: 2-trial sample). The DL checkpoint was trained with `use_video=False` (minimum viable path at `09_dl_prediction.ipynb:96`), meaning the video branch was zeroed out and the model used BVP + physiology + static features only. Performance figures in these CSVs should be treated as **qualitative sanity checks**, not production benchmarks.
