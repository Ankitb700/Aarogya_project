---

# OVERALL SYSTEM GOAL

Your actual architecture should become:

```text id="d2y5m7"
Video + Physiological Signals
            ↓
Physiological Representation Learning
            ↓
HR / HRV / Cardiovascular State / BP Proxy
```
initially.
---

# COMPLETE PIPELINE LAYOUT

---

# STAGE 1 — DATA ALIGNMENT

This is the MOST IMPORTANT stage.

You already have:

* event markers
* timestamps
* frame alignment references

This is excellent.

---

# Inputs

| File      | Purpose                       |
| --------- | ----------------------------- |
| video     | visual rPPG                   |
| BVP.csv   | physiological pulse reference |
| HR.csv    | supervision                   |
| IBI.csv   | HRV                           |
| EDA.csv   | stress/autonomic              |
| TEMP.csv  | perfusion                     |
| ACC.csv   | motion                        |
| tags.csv  | synchronization               |
| track.txt | alignment metadata            |

---

# Output

```text id="0ef0qy"
aligned multimodal timeline
```

Example:

| Timestamp | Video Frame | BVP | HR | EDA | TEMP | ACC |
| --------- | ----------- | --- | -- | --- | ---- | --- |

---

# STAGE 2 — VIDEO PROCESSING

---

# Goal

Extract:

* stable facial ROI
* physiological color traces

---

# Recommended ROI

| ROI      | Importance |
| -------- | ---------- |
| forehead | BEST       |
| cheeks   | VERY GOOD  |
| nose     | optional   |

---

# Recommended Tools

| Tool               | Why         |
| ------------------ | ----------- |
| MediaPipe FaceMesh | lightweight |
| dlib               | stable      |
| RetinaFace         | accurate    |

---

# Output

```text id="7r0t1i"
face_roi_frames
```

---

# STAGE 3 — rPPG EXTRACTION

This is the core.

---

# Extract Signals

| Method        | Importance     |
| ------------- | -------------- |
| Green channel | MUST           |
| CHROM         | VERY IMPORTANT |
| POS           | VERY IMPORTANT |
| ICA           | optional       |

---

# Output

```text id="cdz3pk"
rppg_signal(t)
```

---

# STAGE 4 — SIGNAL CLEANING

VERY important.

---

# Apply

| Method              | Purpose           |
| ------------------- | ----------------- |
| Butterworth filter  | remove noise      |
| Bandpass filter     | heart frequencies |
| Detrending          | baseline removal  |
| Motion compensation | ACC correction    |

---

# Target Frequency

Typical HR range:

0.7\text{ Hz} \leq f \leq 4\text{ Hz}

Equivalent:

* ~42 BPM to 240 BPM

---

# STAGE 5 — FEATURE ENGINEERING

This is where your dataset becomes powerful.

---

# A. VIDEO FEATURES

---

## Raw Features

| Feature            | Meaning            |
| ------------------ | ------------------ |
| RGB mean traces    | pulse info         |
| green intensity    | strongest pulse    |
| skin reflectance   | optical properties |
| temporal variation | blood flow         |

---

## Derived Features

| Feature                | Use                 |
| ---------------------- | ------------------- |
| rPPG SNR               | signal quality      |
| illumination stability | robustness          |
| pulse confidence       | reliability         |
| face motion score      | artifact estimation |

---

# B. BVP FEATURES

VERY important.

---

## Time-Domain Features

| Feature            | Importance           |
| ------------------ | -------------------- |
| pulse amplitude    | vascular strength    |
| pulse width        | vessel elasticity    |
| systolic rise time | cardiovascular state |
| peak intervals     | HR                   |
| pulse skewness     | waveform shape       |

---

## Frequency Features

| Feature            | Importance         |
| ------------------ | ------------------ |
| dominant frequency | HR                 |
| spectral entropy   | variability        |
| PSD energy         | autonomic activity |

---

# C. HRV FEATURES (FROM IBI)

VERY IMPORTANT.

---

# Extract

| Feature | Use               |
| ------- | ----------------- |
| RMSSD   | parasympathetic   |
| SDNN    | HR variability    |
| pNN50   | recovery          |
| LF/HF   | autonomic balance |

---

# D. ACC FEATURES

Critical for production systems.

---

# Extract

| Feature               | Use               |
| --------------------- | ----------------- |
| motion magnitude      | signal corruption |
| movement intensity    | context           |
| acceleration variance | activity level    |

---

# FINAL MODEL DESIGN

---

# OPTION 1 — Classical ML (BEST START)

---

# INPUT

```text id="dxlr6p"
[
 HRV_features,
 pulse_features,
 motion_features,
 EDA_features,
 TEMP_features,
 skin_tone_label
]
```

---

# OUTPUTS

---

## Output A — HR

Best initial target.

---

## Output B — Cardiovascular State

Example:

| Label | Meaning         |
| ----- | --------------- |
| 0     | relaxed         |
| 1     | moderate        |
| 2     | elevated stress |

---

## Output C — BP Proxy

Synthetic target:

| Label | Meaning                 |
| ----- | ----------------------- |
| 0     | low cardiovascular load |
| 1     | medium                  |
| 2     | elevated                |

---

# MODEL

Use:

* XGBoost

BEST for first phase.

---

# OPTION 2 — CUSTOM DL MODEL

This is your future architecture.

---

# INPUT

Multi-branch architecture:

```text id="fms13f"
Video ROI Frames
        ↓
CNN Encoder

BVP Sequence
        ↓
1D CNN

EDA + TEMP + ACC
        ↓
MLP

All Features
        ↓
Fusion Layer
        ↓
Temporal Model
        ↓
Output Head
```

---

# BEST TEMPORAL MODELS

| Model        | Use         |
| ------------ | ----------- |
| Temporal CNN | fast        |


---

# OUTPUT HEADS

---

## Head 1

```text id="n0w76n"
HR regression
```

---

## Head 2

```text id="0lq0ec"
stress classification
```

---

## Head 3

```text id="b6ivvs"
BP proxy regression
```

---

# PSEUDO FEATURES YOU SHOULD CREATE

This is where your custom value comes from.

---

# 1. Cardiovascular Load Index

Combine:

* HR
* HRV
* EDA
* motion

Example:

```text id="4e0c6s"
CV_load =
(HR_norm)
+
(low_HRV_weight)
+
(EDA_weight)
```

---

# 2. Signal Reliability Score

Using:

* motion
* illumination
* pulse SNR

---

# 3. Stress Recovery Index

Using:

* HRV normalization
* EDA recovery
* pulse stabilization

---

# 4. Optical Stability Score

Using:

* RGB consistency
* frame brightness
* skin ROI variance

---

# 5. Personalized Baseline Deviation

VERY important.

Instead of:

```text id="8fawgx"
HR = 92
```

use:

```text id="vs0k6d"
HR deviation from subject baseline
```

This dramatically improves wearable models.

---

# BEST INITIAL TARGETS

---

# FIRST MODEL

## Input

```text id="6a47jf"
engineered physiological features
```

## Output

```text id="dzlqz2"
HR
```

---

# SECOND MODEL

## Input

same features

## Output

```text id="6gwavm"
cardiovascular load
```

---

# THIRD MODEL

## Input

video + physiology

## Output

```text id="fd8wcz"
BP proxy score
```

---

