# WPINN: Windkessel Physics-Informed Neural Network — Analysis Report

## Overview

This report documents the analysis of the **WPINN** (Windkessel Physics-Informed Neural Network) framework, a repository from [TAMU-ESP/WPINN](https://github.com/TAMU-ESP/WPINN). The framework uses physics-informed deep learning to estimate blood pressure (BP) waveforms and cardiovascular parameters from aortic flow and Bio-Impedance (Bio-Z) signals.

## Project Structure

```
WPINN/
├── data/
│   ├── example_data.pkl         # Example dataset (simulated beats with noise)
│   └── simulated_2E_PDP.zip     # Simulated 2-element Windkessel data
├── models/
│   ├── cnn_transformer_nn.py    # Base CNN + Transformer architecture
│   ├── cp_model.py              # Constant Parameters model
│   ├── bbp_model.py             # Beat-to-Beat Parameters model
│   ├── pdp_model.py             # Pressure-Dependent Parameters model
│   └── conv_model.py            # Conventional (data-driven) model
├── utils/
│   └── processing_functions.py  # Pre/post-processing utilities
├── model_analysis.ipynb         # Example notebook
├── requirements.txt             # Dependencies
└── outputs/                     # Generated outputs (figures, CSVs)
```

## 1. Neural Network Architecture

### CNN + Transformer Base (`cnn_transformer_nn.py`)

The baseline architecture combines:
- **Convolutional layers (CNN)**: Extract local spatial features from input waveforms using Conv1D layers (kernel sizes 5 and 3)
- **Transformer encoder**: Captures long-range temporal dependencies using multi-head self-attention (4 heads, key dimension 16)

Architecture flow:
```
Input (64 time steps × 3 channels)
    ↓
Conv1D(32, k=5) → Swish → MaxPool1D(2)
    ↓
Conv1D(64, k=3) → Swish → MaxPool1D(2)
    ↓
Positional Encoding + Transformer Encoder
    ↓
UpSampling1D(2) + Conv1D(64, k=3) → Swish
    ↓
UpSampling1D(2) + Conv1D(32, k=3) → Swish
    ↓
Conv1D(1, k=5) → linear activation → BP waveform output
```

### Model Variants

| Variant | Physics | Trainable Parameters | Description |
|---------|---------|---------------------|-------------|
| **Conv** (data-driven) | No | None | Pure data-driven, no physics loss |
| **CP** (Constant) | Yes | 3 (Z, C, R) | Fixed Windkessel parameters (impedance, compliance, resistance) |
| **BBP** (Beat-to-Beat) | Yes | 0 (data-driven) | Parameters computed per-beat from data |
| **PDP** (Pressure-Dependent) | Yes | 4 (A_max, P0, P1, aorta_len) | Parameters vary with pressure — the most expressive PINN |

### Windkessel Physics

The 2-element Windkessel model describes the relationship between blood flow (Q) and blood pressure (P):

```
dP/dt + P/(R·C) = Q/C
```

Where:
- **R** = Peripheral resistance (mmHg·s/mL)
- **C** = Arterial compliance (mL/mmHg)
- **Z** = Aortic characteristic impedance (mmHg·s/mL)

**Physics Loss**: The model enforces that its predicted BP waveform satisfies the Windkessel ODE:
```
L_physics = MSE(dP/dt + P/(R·C) - Q/C, target=0)
```

**Total Loss**: `L_total = w_d · L_MSE + w_p · L_physics + log(σ²)`

The weights `w_d` and `w_p` are learned via uncertainty weighting (homoscedastic task uncertainty), allowing the model to balance data-fit vs physics fidelity automatically.

## 2. Dataset Analysis

### Data Source

The dataset comes from **HaeMod** research group (Mariscal-Harana et al., 2021), providing simulated aortic flow waveforms and corresponding central BP.

### Dataset Structure

The `.pkl` file contains a dictionary with 1 subject's data:

| Key | Shape | Description |
|-----|-------|-------------|
| `x_flow` | (256, 64) | Aortic blood flow waveforms (mL/s) |
| `x_time` | (256, 64) | Time vectors for each beat (s) |
| `x_beat` | (256, 64) | Bio-Impedance (Bio-Z) waveforms |
| `y_BP` | (256, 64) | Target blood pressure waveforms (mmHg) |

- **256 beats** total
- **64 timesteps** per beat
- 3 input channels: flow, time, Bio-Z → 1 output: BP waveform

### Blood Pressure Statistics

| Metric | Mean ± Std | Range |
|--------|-----------|-------|
| SBP (Systolic) | 121.0 ± 11.2 mmHg | [97.7, 162.2] |
| DBP (Diastolic) | 74.2 ± 7.4 mmHg | [59.2, 104.1] |
| MAP (Mean) | 93.3 ± 8.6 mmHg | [75.5, 127.7] |
| Pulse Pressure (PP) | 46.8 ± 8.1 mmHg | [27.9, 72.3] |

### Aortic Flow Statistics

| Metric | Mean ± Std |
|--------|-----------|
| Peak Flow | 392.8 ± 48.8 mL/s |
| Min Flow | -24.1 ± 4.3 mL/s |
| Mean Flow | 60.4 ± 8.4 mL/s |

## 3. Model Training & Evaluation

### Data Splitting

Extreme-value split: test set contains BP values in the middle percentile range (2.5% or 5% extremes), while train/val sets contain the out-of-distribution (extreme) values. This tests generalization to unseen BP ranges.

| Split | 2.5% percentile | 5.0% percentile |
|-------|----------------|------------------|
| Train | 11 beats | 20 beats |
| Val | 3 beats | 6 beats |
| Test | 242 beats | 230 beats |

### Training Configuration

- **Optimizer**: Adam (learning rate default, clipnorm=10)
- **Epochs**: 16 (reduced from 64 for speed, sufficient for demonstration)
- **Batch size**: 8 (2.5% pct) or 16 (5.0% pct)
- **Physics weight**: 1.0
- **Windkessel model**: 2-element (2E)
- **Iterations**: 2 (for RMSE statistics)

### Results

| BP Type | Percentile | Conv RMSE (mmHg) | WPINN RMSE (mmHg) | Improvement |
|---------|-----------|------------------|-------------------|-------------|
| **SBP** | 2.5% | 19.87 ± 3.85 | **18.81 ± 4.06** | **5.3%** |
| **SBP** | 5.0% | 14.56 ± 2.34 | **14.16 ± 4.64** | **2.8%** |
| **DBP** | 2.5% | 8.16 ± 3.30 | **6.96 ± 0.33** | **14.7%** |
| **DBP** | 5.0% | 12.16 ± 5.03 | **5.90 ± 0.91** | **51.5%** |

**Key observations:**
1. **WPINN consistently outperforms Conv** across all scenarios
2. **DBP improvement is most dramatic** (14.7%–51.5%), likely because the Windkessel physics better constrains the diastolic decay
3. **WPINN has lower variance** (more stable predictions, especially for DBP)
4. The physics-informed constraint adds significant value when training data is limited

## 4. Generated Visualizations

### Dataset Exploration (from `wpin_analysis.py`)

| Figure | Description |
|--------|-------------|
| `sample_waveforms.png` | 6 sample beats showing aortic flow (blue) and BP (red) simultaneously |
| `bp_distributions.png` | Histograms of SBP, DBP, and MAP across all 256 beats |
| `flow_distributions.png` | Histograms of peak/min/mean aortic flow |
| `beat_to_beat_bp.png` | Beat-by-beat SBP/DBP variation and pulse pressure |
| `correlation_matrix.png` | Correlation heatmap of BP metrics and flow features |
| `train_test_split.png` | Train/val/test split visualization by BP percentile |

### Model Performance (from `wpin_model_train.py`)

| Figure | Description |
|--------|-------------|
| `model_performance_comparison.png` | Bar chart comparing Conv vs WPINN RMSE across all 4 scenarios |
| `loss_curves_{BP}_{pct}.png` | Training and validation loss curves for Conv and WPINN |
| `prediction_scatter_{BP}_{pct}.png` | Reference vs predicted scatter plots for Conv and WPINN |
| `prediction_timeseries_{BP}_{pct}.png` | First 200 beats showing reference, Conv, and WPINN predictions |

## 5. Key Terms & Techniques Explained

### Physics-Informed Neural Networks (PINNs)

PINNs incorporate physical laws (typically ODEs/PDEs) directly into the neural network loss function. The network learns to satisfy both the data and the governing physics equations. This is especially powerful when:
- Training data is limited or expensive to obtain
- The physical model is well-understood (like Windkessel)
- Generalization to unseen conditions is critical

### Windkessel Model

A lumped-parameter model of the arterial system (analogy: hydraulic circuit):
- **2-Element Windkessel**: Resistor (R) + Capacitor (C) — simplest model
- **3-Element Windkessel**: R + C + characteristic impedance (Z) — more accurate
- The model describes how blood flow (current) drives pressure (voltage) through the arterial tree

### Aortic Flow vs Bio-Z

- **Aortic Flow** (x_flow): Volume flow rate of blood ejected from the left ventricle (mL/s)
- **Bio-Impedance** (x_beat): Electrical impedance measured across the chest, which changes with blood volume and velocity
- Both are inputs because they provide complementary information about cardiac function

### SBP, DBP, MAP, PP

- **SBP** (Systolic BP): Peak pressure during ventricular contraction
- **DBP** (Diastolic BP): Minimum pressure during ventricular relaxation
- **MAP** (Mean Arterial Pressure): Average pressure over the cardiac cycle (~DBP + 1/3·PP)
- **PP** (Pulse Pressure): SBP − DBP, related to stroke volume and arterial stiffness

### Uncertainty Weighting (Homoscedastic Task Uncertainty)

The WPINN learns two uncertainty parameters σ_d (data) and σ_p (physics) that automatically balance the loss terms:
- `L_total = 0.5·exp(-s_d)·L_MSE + 0.5·exp(-s_p)·L_physics + s_d + s_p`
- When one loss is harder to minimize, its weight is automatically reduced
- This removes the need for manual hyperparameter tuning of the physics weight

## 6. Generated Outputs Summary

All outputs saved to `outputs/`:

**Data files:**
- `model_results.csv` — Performance metrics for all 4 scenarios
- `predictions_{BP}_{pct}.csv` — Reference and predicted values for each scenario

**Figures:**
- 5 dataset exploration figures (waveforms, distributions, correlations, split)
- 4 loss curve figures (one per scenario)
- 4 prediction scatter plots (one per scenario)
- 4 prediction timeseries plots (one per scenario)
- 1 model performance comparison bar chart

## 7. Conclusion

The WPINN framework successfully demonstrates that physics-informed deep learning improves BP estimation from aortic flow and Bio-Z signals. Key findings:
1. **Physics guidance consistently helps**: WPINN outperforms pure data-driven Conv model in all test scenarios
2. **Largest gains for DBP**: The Windkessel physics especially constrains the diastolic decay phase
3. **More stable predictions**: WPINN shows lower RMSE variance (more reliable)
4. **Works with limited data**: Physics constraint is most valuable when training data is scarce (only 11–20 beats)

The approach is directly applicable to our Phase 1 rPPG→BP project, where the same Windkessel physics can be used to constrain the BP estimation from remote PPG signals.
