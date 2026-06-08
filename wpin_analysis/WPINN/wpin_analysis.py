"""WPINN: Comprehensive dataset analysis, model training, and visualization."""
import sys, os, warnings, json
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')
sns.set_context('talk', font_scale=0.8)

from pathlib import Path
ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
FIGS = OUTPUTS / "figures"
FIGS.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# 1. DATASET EXPLORATION
# ──────────────────────────────────────────────
print("=" * 70)
print("1. DATASET EXPLORATION")
print("=" * 70)

df = pd.read_pickle(str(ROOT / "data" / "example_data.pkl"))
print(f"\nDataset type: {type(df)}")
if isinstance(df, dict):
    print(f"Dictionary keys: {list(df.keys())}")
    for k, v in df.items():
        if isinstance(v, (list, np.ndarray)):
            print(f"  {k}: type={type(v)}, len={len(v)}")
            for i, arr in enumerate(v):
                print(f"    [{i}]: shape={arr.shape}, dtype={arr.dtype}, "
                      f"range=[{arr.min():.2f}, {arr.max():.2f}], mean={arr.mean():.2f}")
        elif isinstance(v, pd.DataFrame):
            print(f"  {k}: DataFrame shape={v.shape}")
        else:
            print(f"  {k}: {type(v)}")
elif isinstance(df, pd.DataFrame):
    print(f"DataFrame shape={df.shape}")
    print(df.head())
    print(df.describe())

# Extract arrays
x_flow = df['x_flow'][0]
x_beat = df['x_beat'][0]
x_time = df['x_time'][0]
y_BP = df['y_BP'][0]

print(f"\nDataset Summary:")
print(f"  x_flow (aortic flow): {x_flow.shape} [{x_flow.min():.2f}, {x_flow.max():.2f}] mL/s")
print(f"  x_beat (Bio-Z):       {x_beat.shape} [{x_beat.min():.2f}, {x_beat.max():.2f}]")
print(f"  x_time (time):         {x_time.shape} [{x_time.min():.2f}, {x_time.max():.2f}] s")
print(f"  y_BP (blood pressure): {y_BP.shape} [{y_BP.min():.2f}, {y_BP.max():.2f}] mmHg")

n_sub = y_BP.shape[0]
n_timesteps = y_BP.shape[1]
print(f"  Subjects/beats: {n_sub}, Timesteps per beat: {n_timesteps}")

# ──────────────────────────────────────────────
# 2. VISUALIZATION - Sample waveforms
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("2. GENERATING WAVEFORM VISUALIZATIONS")
print("=" * 70)

# Plot sample beats
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
beat_ids = [0, 50, 100, 200, 300, 500]
for ax, bid in zip(axes.flat, beat_ids):
    if bid < n_sub:
        ax.plot(x_time[bid], x_flow[bid], 'b-', lw=1, label='Flow')
        ax_twin = ax.twinx()
        ax_twin.plot(x_time[bid], y_BP[bid], 'r-', lw=1, label='BP')
        ax.set_title(f'Beat #{bid}', fontsize=10)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Flow (mL/s)', color='b')
        ax_twin.set_ylabel('BP (mmHg)', color='r')
        if bid == beat_ids[0]:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_twin.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
plt.suptitle('Sample Aortic Flow & Blood Pressure Waveforms', fontsize=14)
plt.tight_layout()
plt.savefig(FIGS / 'sample_waveforms.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: sample_waveforms.png")

# BP distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sbp = y_BP.max(axis=-1)
dbp = y_BP[:, 0]
map_val = y_BP.mean(axis=-1)
pp = sbp - dbp
for ax, data, title, unit in zip(axes, [sbp, dbp, map_val],
                                  ['SBP (Systolic)', 'DBP (Diastolic)', 'MAP (Mean)'],
                                  ['mmHg', 'mmHg', 'mmHg']):
    ax.hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(data.mean(), color='red', ls='--', lw=2, label=f'Mean: {data.mean():.1f}')
    ax.set_title(f'{title} Distribution')
    ax.set_xlabel(unit)
    ax.set_ylabel('Count')
    ax.legend()
plt.suptitle('Blood Pressure Distributions Across All Beats', fontsize=14)
plt.tight_layout()
plt.savefig(FIGS / 'bp_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: bp_distributions.png")

# Flow statistics
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
peak_flow = x_flow.max(axis=-1)
min_flow = x_flow.min(axis=-1)
mean_flow = x_flow.mean(axis=-1)
for ax, data, title in zip(axes, [peak_flow, min_flow, mean_flow],
                            ['Peak Flow', 'Min Flow', 'Mean Flow']):
    ax.hist(data, bins=30, edgecolor='black', alpha=0.7, color='coral')
    ax.axvline(data.mean(), color='red', ls='--', lw=2, label=f'Mean: {data.mean():.1f}')
    ax.set_title(f'{title} Distribution')
    ax.set_xlabel('mL/s')
    ax.set_ylabel('Count')
    ax.legend()
plt.suptitle('Aortic Flow Distributions', fontsize=14)
plt.tight_layout()
plt.savefig(FIGS / 'flow_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: flow_distributions.png")

# Beat-to-beat variation
fig, axes = plt.subplots(2, 1, figsize=(14, 6))
axes[0].plot(sbp, 'b-', lw=0.8, alpha=0.7, label='SBP')
axes[0].plot(dbp, 'g-', lw=0.8, alpha=0.7, label='DBP')
axes[0].set_title('Beat-to-Beat BP Variation', fontsize=12)
axes[0].set_xlabel('Beat Index'), axes[0].set_ylabel('BP (mmHg)')
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(pp, 'r-', lw=0.8, alpha=0.7)
axes[1].set_title('Pulse Pressure (SBP - DBP)', fontsize=12)
axes[1].set_xlabel('Beat Index'), axes[1].set_ylabel('Pulse Pressure (mmHg)')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS / 'beat_to_beat_bp.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: beat_to_beat_bp.png")

# Correlation analysis
corr_data = pd.DataFrame({
    'SBP': sbp, 'DBP': dbp, 'MAP': map_val,
    'PP': pp, 'Peak_Flow': peak_flow,
    'Min_Flow': min_flow, 'Mean_Flow': mean_flow
})
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_data.corr(), annot=True, cmap='RdBu_r', center=0,
            square=True, fmt='.2f', ax=ax)
ax.set_title('Feature Correlation Matrix', fontsize=14)
plt.tight_layout()
plt.savefig(FIGS / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: correlation_matrix.png")

print("\nDataset Statistics:")
stats = {
    'SBP (mmHg)': {'Mean': float(sbp.mean()), 'Std': float(sbp.std()),
                   'Min': float(sbp.min()), 'Max': float(sbp.max())},
    'DBP (mmHg)': {'Mean': float(dbp.mean()), 'Std': float(dbp.std()),
                   'Min': float(dbp.min()), 'Max': float(dbp.max())},
    'MAP (mmHg)': {'Mean': float(map_val.mean()), 'Std': float(map_val.std()),
                   'Min': float(map_val.min()), 'Max': float(map_val.max())},
    'PP (mmHg)': {'Mean': float(pp.mean()), 'Std': float(pp.std()),
                   'Min': float(pp.min()), 'Max': float(pp.max())},
    'Peak Flow (mL/s)': {'Mean': float(peak_flow.mean()), 'Std': float(peak_flow.std()),
                         'Min': float(peak_flow.min()), 'Max': float(peak_flow.max())},
}
stats_df = pd.DataFrame(stats).T
print(stats_df.round(2).to_string())

# ──────────────────────────────────────────────
# 3. MODEL TRAINING & EVALUATION
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("3. MODEL TRAINING & EVALUATION")
print("=" * 70)

sys.path.insert(0, str(ROOT))
from utils.processing_functions import get_out_of_dist_split_indexes, moving_average
from models.conv_model import ConvModel
from models.pdp_model import PDPModel

Ew_model = 2
num_epochs = 32  # Reduced for speed, sufficient for demonstration
w = 10
N_iter = 3  # Reduced iterations for speed

results_summary = []

for BP_type in ['SBP', 'DBP']:
    for mult in [1, 2]:
        rmse_conv_arr, rmse_wpinn_arr = [], []
        conv_losses, wpinn_losses = [], []
        
        for it in range(N_iter):
            train_ind, val_ind, test_ind = get_out_of_dist_split_indexes(
                y_BP, BP_type=BP_type, percentile=2.5 * mult
            )
            print(f"  {BP_type}, pct={2.5*mult}%, iter={it+1}/{N_iter}: "
                  f"train={len(train_ind)}, val={len(val_ind)}, test={len(test_ind)}")

            # Conv model
            _, y_test_conv, y_conv, conv_conv_loss, conv_val_loss = ConvModel(
                x_flow, x_time, x_beat, y_BP, train_ind, val_ind, test_ind
            ).model_train(batch=8 * mult, epochs=num_epochs)
            conv_losses.append({'conv_loss': conv_conv_loss.tolist(),
                                'val_loss': conv_val_loss.tolist()})

            # WPINN model
            _, _, y_wpinn, wpinn_conv_loss, wpinn_phys_loss, wpinn_val_loss = PDPModel(
                x_flow, x_time, x_beat, y_BP, train_ind, val_ind, test_ind
            ).model_train(Ew_model=Ew_model, physics_weight=1,
                          batch=8 * mult, epochs=num_epochs)
            wpinn_losses.append({'conv_loss': wpinn_conv_loss.tolist(),
                                 'phys_loss': wpinn_phys_loss.tolist(),
                                 'val_loss': wpinn_val_loss.tolist()})

            # Evaluate
            if BP_type == 'DBP':
                ref = moving_average(y_test_conv[:, 0], w=w)
                conv_pred = moving_average(y_conv[:, 0], w=w)
                wpinn_pred = moving_average(y_wpinn[:, 0], w=w)
            else:
                ref = moving_average(y_test_conv.max(axis=-1), w=w)
                conv_pred = moving_average(y_conv.max(axis=-1), w=w)
                wpinn_pred = moving_average(y_wpinn.max(axis=-1), w=w)

            rmse_conv = np.sqrt(np.mean(np.square(ref - conv_pred)))
            rmse_wpinn = np.sqrt(np.mean(np.square(ref - wpinn_pred)))
            rmse_conv_arr.append(rmse_conv)
            rmse_wpinn_arr.append(rmse_wpinn)

            # Save predictions for first iteration for plotting
            if it == 0 and mult == 1:
                pred_data = {'ref': ref, 'conv': conv_pred, 'wpinn': wpinn_pred}
                pred_df = pd.DataFrame(pred_data)
                pred_df.to_csv(OUTPUTS / f'predictions_{BP_type}.csv', index=False)

        rmse_conv_arr = np.array(rmse_conv_arr)
        rmse_wpinn_arr = np.array(rmse_wpinn_arr)
        
        print(f"  >> Conv RMSE: {rmse_conv_arr.mean():.2f} ± {rmse_conv_arr.std():.2f} mmHg")
        print(f"  >> WPINN RMSE: {rmse_wpinn_arr.mean():.2f} ± {rmse_wpinn_arr.std():.2f} mmHg")
        
        results_summary.append({
            'BP_Type': BP_type, 'Percentile': f'{2.5*mult}%',
            'Conv_RMSE_mean': float(rmse_conv_arr.mean()),
            'Conv_RMSE_std': float(rmse_conv_arr.std()),
            'WPINN_RMSE_mean': float(rmse_wpinn_arr.mean()),
            'WPINN_RMSE_std': float(rmse_wpinn_arr.std()),
            'Improvement_pct': float((rmse_conv_arr.mean() - rmse_wpinn_arr.mean()) / rmse_conv_arr.mean() * 100)
        })

results_df = pd.DataFrame(results_summary)
print("\n\nModel Performance Summary:")
print(results_df.round(2).to_string(index=False))
results_df.to_csv(OUTPUTS / 'model_results.csv', index=False)

# ──────────────────────────────────────────────
# 4. PERFORMANCE VISUALIZATIONS
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("4. PERFORMANCE VISUALIZATIONS")
print("=" * 70)

# RMSE comparison bar chart
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(results_df))
width = 0.35
ax.bar(x - width/2, results_df['Conv_RMSE_mean'], width, yerr=results_df['Conv_RMSE_std'],
       label='Conv (Data-Driven)', capsize=3, color='steelblue')
ax.bar(x + width/2, results_df['WPINN_RMSE_mean'], width, yerr=results_df['WPINN_RMSE_std'],
       label='WPINN (Physics-Informed)', capsize=3, color='coral')
ax.set_xticks(x)
ax.set_xticklabels([f"{r['BP_Type']}\n{pct}" for r, pct in
                    zip(results_summary, results_df['Percentile'])])
ax.set_ylabel('RMSE (mmHg)')
ax.set_title('Conv vs WPINN Model Performance Comparison')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(FIGS / 'model_performance_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: model_performance_comparison.png")

# Prediction scatter plots
for bp_type in ['SBP', 'DBP']:
    pred_path = OUTPUTS / f'predictions_{bp_type}.csv'
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, (pred_name, color) in zip(axes, [('conv', 'steelblue'), ('wpinn', 'coral')]):
            ax.scatter(pred_df['ref'], pred_df[pred_name], alpha=0.5, s=10, color=color)
            lims = [min(pred_df['ref'].min(), pred_df[pred_name].min()),
                    max(pred_df['ref'].max(), pred_df[pred_name].max())]
            ax.plot(lims, lims, 'k--', lw=1, alpha=0.5)
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_xlabel(f'Reference {bp_type} (mmHg)')
            ax.set_ylabel(f'Predicted {bp_type} (mmHg)')
            rmse = np.sqrt(np.mean((pred_df['ref'] - pred_df[pred_name])**2))
            ax.set_title(f'{"Conv" if pred_name == "conv" else "WPINN"} — RMSE: {rmse:.2f} mmHg')
            ax.grid(True, alpha=0.3)
        plt.suptitle(f'{bp_type} Prediction: Reference vs Predicted', fontsize=14)
        plt.tight_layout()
        plt.savefig(FIGS / f'prediction_scatter_{bp_type}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: prediction_scatter_{bp_type}.png")

# Time-series prediction example
for bp_type in ['SBP', 'DBP']:
    pred_path = OUTPUTS / f'predictions_{bp_type}.csv'
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path)
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(pred_df['ref'][:200], 'k-', lw=1.5, label='Reference', alpha=0.8)
        ax.plot(pred_df['conv'][:200], 'b-', lw=1, label='Conv', alpha=0.7)
        ax.plot(pred_df['wpinn'][:200], 'r-', lw=1, label='WPINN', alpha=0.7)
        ax.set_xlabel('Beat Index')
        ax.set_ylabel(f'{bp_type} (mmHg)')
        ax.set_title(f'{bp_type} Prediction: Reference vs Models (First 200 beats)')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGS / f'prediction_timeseries_{bp_type}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: prediction_timeseries_{bp_type}.png")

# Loss curves from first iteration
if conv_losses:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, losses, title, color in zip(axes,
                                        [conv_losses[0], wpinn_losses[0]],
                                        ['Conv Model Loss', 'WPINN Model Loss'],
                                        ['steelblue', 'coral']):
        for key, ls in [('conv_loss', '-'), ('val_loss', '--')]:
            if key in losses:
                ax.plot(losses[key], ls=ls, lw=1.5, label=key.replace('_', ' ').title())
        if 'phys_loss' in losses:
            ax.plot(losses['phys_loss'], ls=':', lw=1.5, label='Physics Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.suptitle('Training Loss Curves', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGS / 'training_loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: training_loss_curves.png")

# ──────────────────────────────────────────────
# 5. SAVE RESULTS
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("5. SAVING RESULTS")
print("=" * 70)

# BP percentile split visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, bp_type, color in zip(axes, ['SBP', 'DBP'], ['steelblue', 'green']):
    train_ind, val_ind, test_ind = get_out_of_dist_split_indexes(
        y_BP, BP_type=bp_type, percentile=5)
    train_bp = y_BP.max(axis=-1) if bp_type == 'SBP' else y_BP[:, 0]
    ax.hist(train_bp[train_ind], bins=30, alpha=0.7, label=f'Train (n={len(train_ind)})', color=color)
    ax.hist(train_bp[val_ind], bins=15, alpha=0.6, label=f'Val (n={len(val_ind)})', color='orange')
    ax.hist(train_bp[test_ind], bins=15, alpha=0.6, label=f'Test (n={len(test_ind)})', color='red')
    ax.set_title(f'{bp_type}: Train/Val/Test Split (Percentile=5%)')
    ax.set_xlabel(f'{bp_type} (mmHg)'), ax.set_ylabel('Count')
    ax.legend()
plt.tight_layout()
plt.savefig(FIGS / 'train_test_split.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: train_test_split.png")

# Save comprehensive results JSON
all_results = {
    'dataset': {
        'n_beats': n_sub,
        'n_timesteps': n_timesteps,
        'features': ['x_flow (aortic flow)', 'x_time', 'x_beat (Bio-Z)'],
        'target': 'y_BP (blood pressure)',
    },
    'statistics': {
        'SBP': {'mean': float(sbp.mean()), 'std': float(sbp.std()),
                'min': float(sbp.min()), 'max': float(sbp.max())},
        'DBP': {'mean': float(dbp.mean()), 'std': float(dbp.std()),
                'min': float(dbp.min()), 'max': float(dbp.max())},
        'MAP': {'mean': float(map_val.mean()), 'std': float(map_val.std()),
                'min': float(map_val.min()), 'max': float(map_val.max())},
        'PP': {'mean': float(pp.mean()), 'std': float(pp.std()),
               'min': float(pp.min()), 'max': float(pp.max())},
    },
    'model_performance': results_summary,
}

with open(OUTPUTS / 'analysis_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print("Saved: analysis_results.json")
print(f"\nAll outputs saved to: {OUTPUTS}")
print(f"Figures saved to: {FIGS}")
print("\n" + "=" * 70)
print("WPINN ANALYSIS COMPLETE")
print("=" * 70)
