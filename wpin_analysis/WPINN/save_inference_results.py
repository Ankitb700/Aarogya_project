"""Load saved model, run inference on test data, save CSV + plots."""
import sys, os, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from utils.processing_functions import get_out_of_dist_split_indexes, moving_average

OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
FIGS = OUTPUTS / "figures"

# Load data
df = pd.read_pickle(str(ROOT / "data" / "example_data.pkl"))
x_flow, x_beat, x_time, y_BP = df['x_flow'][0], df['x_beat'][0], df['x_time'][0], df['y_BP'][0]

# Get split indices
idx = np.argsort(y_BP.max(axis=-1))
n = len(y_BP)
train_ind = idx[:int(n * 0.05)]
val_ind = idx[int(n * 0.05):int(n * 0.08)]
test_ind = idx[int(n * 0.08):]

# Load saved model
model = tf.keras.models.load_model(str(ROOT / "saved_models" / "conv_model.keras"))

# Preprocess test data with training stats
x_flow_test = (x_flow[test_ind] - np.mean(x_flow[train_ind])) / np.std(x_flow[train_ind])
x_time_test = (x_time[test_ind] - np.mean(x_time[train_ind])) / np.std(x_time[train_ind])
x_beat_test = (x_beat[test_ind] - np.mean(x_beat[train_ind])) / np.std(x_beat[train_ind])
x_wave_test = np.dstack([x_flow_test, x_time_test, x_beat_test])

m_BP = np.mean(y_BP[train_ind])
s_BP = np.std(y_BP[train_ind])

# Direct inference
preds = model.predict(x_wave_test, verbose=0)[:, :, 0] * s_BP + m_BP
ref = y_BP[test_ind]

# Build results table
results = pd.DataFrame({
    'test_idx': test_ind,
    'sbp_ref': moving_average(ref.max(axis=-1), w=10),
    'sbp_pred': moving_average(preds.max(axis=-1), w=10),
    'dbp_ref': moving_average(ref[:, 0], w=10),
    'dbp_pred': moving_average(preds[:, 0], w=10),
})
results.to_csv(OUTPUTS / 'inference_results.csv', index=False)

# Metrics
sbp_rmse = np.sqrt(np.mean((results.sbp_ref - results.sbp_pred)**2))
dbp_rmse = np.sqrt(np.mean((results.dbp_ref - results.dbp_pred)**2))
sbp_mae = np.mean(np.abs(results.sbp_ref - results.sbp_pred))
dbp_mae = np.mean(np.abs(results.dbp_ref - results.dbp_pred))

summary = pd.DataFrame({
    'Metric': ['SBP RMSE', 'DBP RMSE', 'SBP MAE', 'DBP MAE'],
    'Value (mmHg)': [f'{sbp_rmse:.2f}', f'{dbp_rmse:.2f}', f'{sbp_mae:.2f}', f'{dbp_mae:.2f}']
})
summary.to_csv(OUTPUTS / 'inference_metrics.csv', index=False)

print("=== INFERENCE RESULTS ===")
print(results.describe())
print("\n=== METRICS ===")
print(summary.to_string(index=False))

# Scatter plot: SBP
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (ref_vals, pred_vals, label) in zip(axes,
    [(results.sbp_ref, results.sbp_pred, 'SBP'),
     (results.dbp_ref, results.dbp_pred, 'DBP')]):
    ax.scatter(ref_vals, pred_vals, alpha=0.5, s=15)
    lims = [min(ref_vals.min(), pred_vals.min()), max(ref_vals.max(), pred_vals.max())]
    ax.plot(lims, lims, 'r--', lw=1)
    ax.set_xlabel(f'{label} Reference (mmHg)')
    ax.set_ylabel(f'{label} Predicted (mmHg)')
    ax.set_title(f'{label}: Direct Inference vs Reference')
    rmse = np.sqrt(np.mean((ref_vals - pred_vals)**2))
    ax.text(0.05, 0.95, f'RMSE={rmse:.1f} mmHg', transform=ax.transAxes, va='top')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGS / 'inference_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nSaved: {OUTPUTS / 'inference_results.csv'}")
print(f"Saved: {OUTPUTS / 'inference_metrics.csv'}")
print(f"Saved: {FIGS / 'inference_scatter.png'}")
