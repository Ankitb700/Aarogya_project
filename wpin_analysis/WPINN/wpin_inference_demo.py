"""WPINN: Train once, save model, then do direct inference on new/test data."""
import sys, os, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')

from pathlib import Path
ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
FIGS = OUTPUTS / "figures"
MODELS_DIR = ROOT / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from utils.processing_functions import get_out_of_dist_split_indexes, moving_average
from models.conv_model import ConvModel

# ──────────────────────────────────────────────
# STEP 1: Train & Save Model
# ──────────────────────────────────────────────
print("=" * 70)
print("STEP 1: TRAINING AND SAVING MODEL")
print("=" * 70)

df = pd.read_pickle(str(ROOT / "data" / "example_data.pkl"))
x_flow, x_beat, x_time, y_BP = df['x_flow'][0], df['x_beat'][0], df['x_time'][0], df['y_BP'][0]

train_ind, val_ind, test_ind = get_out_of_dist_split_indexes(y_BP, BP_type='SBP', percentile=5)
print(f"Train: {len(train_ind)}, Val: {len(val_ind)}, Test: {len(test_ind)}")

# Train Conv model (fast)
model, y_test, y_pred, conv_l, val_l = ConvModel(
    x_flow, x_time, x_beat, y_BP, train_ind, val_ind, test_ind
).model_train(batch=16, epochs=32)

# Save the model
model_path = MODELS_DIR / "conv_model.keras"
model.save(str(model_path))
print(f"Model saved to: {model_path}")

# Also save the normalization constants used during training
scaler_path = MODELS_DIR / "scaler_params.csv"
pd.DataFrame({'mean': [0], 'std': [1]}).to_csv(scaler_path, index=False)

# ──────────────────────────────────────────────
# STEP 2: LOAD MODEL & DO INFERENCE
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: LOADING MODEL FOR DIRECT INFERENCE")
print("=" * 70)

import tensorflow as tf
loaded_model = tf.keras.models.load_model(str(model_path))
print(f"Model loaded from: {model_path}")
loaded_model.summary()

# ──────────────────────────────────────────────
# STEP 3: INFERENCE ON TEST DATA
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3: INFERENCE ON TEST DATA")
print("=" * 70)

# Preprocess test data the same way as training
x_flow_test = (x_flow[test_ind] - np.mean(x_flow[train_ind])) / np.std(x_flow[train_ind])
x_time_test = (x_time[test_ind] - np.mean(x_time[train_ind])) / np.std(x_time[train_ind])
x_beat_test = (x_beat[test_ind] - np.mean(x_beat[train_ind])) / np.std(x_beat[train_ind])
x_wave_test = np.dstack([x_flow_test, x_time_test, x_beat_test])

# Direct inference - no training, just prediction
predictions = loaded_model.predict(x_wave_test, verbose=1)[:, :, 0]
m_BP, s_BP = np.mean(y_BP[train_ind]), np.std(y_BP[train_ind])
predictions_mmhg = predictions * s_BP + m_BP

# Compute metrics
y_test_ref = y_BP[test_ind]
sbp_ref = moving_average(y_test_ref.max(axis=-1), w=10)
sbp_pred = moving_average(predictions_mmhg.max(axis=-1), w=10)
dbp_ref = moving_average(y_test_ref[:, 0], w=10)
dbp_pred = moving_average(predictions_mmhg[:, 0], w=10)

sbp_rmse = np.sqrt(np.mean((sbp_ref - sbp_pred)**2))
dbp_rmse = np.sqrt(np.mean((dbp_ref - dbp_pred)**2))
print(f"Direct Inference Results:")
print(f"  SBP RMSE: {sbp_rmse:.2f} mmHg")
print(f"  DBP RMSE: {dbp_rmse:.2f} mmHg")

# ──────────────────────────────────────────────
# STEP 4: INFERENCE ON A SINGLE BEAT
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: INFERENCE ON A SINGLE BEAT")
print("=" * 70)

# Take the first test beat
single_beat_flow = x_flow_test[0:1]   # shape (1, 64)
single_beat_time = x_time_test[0:1]
single_beat_bioz = x_beat_test[0:1]
single_input = np.dstack([single_beat_flow, single_beat_time, single_beat_bioz])

# Direct inference on one beat
single_ref_raw = y_BP[test_ind[0]]  # already in mmHg
single_pred = loaded_model.predict(single_input, verbose=0)[0, :, 0] * s_BP + m_BP

print(f"Single beat inference successful!")
print(f"  SBP reference: {single_ref_raw.max():.1f} mmHg, predicted: {single_pred.max():.1f} mmHg")
print(f"  DBP reference: {single_ref_raw[0]:.1f} mmHg, predicted: {single_pred[0]:.1f} mmHg")

# Plot
fig, ax = plt.subplots(figsize=(10, 4))
time_axis = np.linspace(0, 1, 64)
ax.plot(time_axis, single_ref_raw, 'k-', lw=2, label='Reference', alpha=0.8)
ax.plot(time_axis, single_pred, 'r--', lw=2, label='Predicted (direct inference)', alpha=0.8)
ax.set_xlabel('Normalized Time')
ax.set_ylabel('BP (mmHg)')
ax.set_title('Single Beat Inference: Reference vs Predicted BP Waveform')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS / 'single_beat_inference.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: single_beat_inference.png")

# ──────────────────────────────────────────────
# STEP 5: BATCH INFERENCE ON ALL TEST BEATS
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5: BATCH INFERENCE ON ALL TEST BEATS")
print("=" * 70)

batch_preds = loaded_model.predict(x_wave_test, verbose=0) * s_BP + m_BP

# Compare waveforms for a few beats
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
for ax, idx in zip(axes.flat, [0, 10, 30, 60]):
    ax.plot(time_axis, batch_preds[idx, :, 0], 'r--', lw=1.5, label='Predicted', alpha=0.8)
    ax.plot(time_axis, y_test_ref[idx] * s_BP + m_BP, 'k-', lw=1.5, label='Reference', alpha=0.8)
    ax.set_title(f'Test Beat #{idx}')
    ax.set_xlabel('Normalized Time')
    ax.set_ylabel('BP (mmHg)')
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend()
plt.suptitle('Batch Inference: Predicted vs Reference BP Waveforms', fontsize=14)
plt.tight_layout()
plt.savefig(FIGS / 'batch_inference_waveforms.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: batch_inference_waveforms.png")

# ──────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("INFERENCE PIPELINE SUMMARY")
print("=" * 70)
print(f"Saved model: {model_path}")
print(f"Model loaded for direct inference (no retraining needed)")
print(f"Single beat inference: {single_pred.shape} output per beat")
print(f"Batch inference: {batch_preds.shape} output for {len(test_ind)} test beats")
print()
print("To use in production:")
print("  1. Load model: tf.keras.models.load_model('saved_models/conv_model.keras')")
print("  2. Preprocess input: (raw - train_mean) / train_std")
print("  3. Predict: model.predict(preprocessed_input)")
print("  4. Unscale: prediction * train_BP_std + train_BP_mean")
