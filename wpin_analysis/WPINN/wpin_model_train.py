"""WPINN Model Training (lightweight version)."""
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
sns.set_context('talk', font_scale=0.8)

from pathlib import Path
ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
FIGS = OUTPUTS / "figures"
OUTPUTS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
from utils.processing_functions import get_out_of_dist_split_indexes, moving_average
from models.conv_model import ConvModel
from models.pdp_model import PDPModel

df = pd.read_pickle(str(ROOT / "data" / "example_data.pkl"))
x_flow, x_beat, x_time, y_BP = df['x_flow'][0], df['x_beat'][0], df['x_time'][0], df['y_BP'][0]

Ew_model = 2
num_epochs = 16
w = 10
N_iter = 2

results_summary = []

for BP_type in ['SBP', 'DBP']:
    for mult in [1, 2]:
        rmse_conv_arr, rmse_wpinn_arr = [], []
        all_preds = []
        for it in range(N_iter):
            train_ind, val_ind, test_ind = get_out_of_dist_split_indexes(
                y_BP, BP_type=BP_type, percentile=2.5 * mult)
            print(f"[{BP_type}, pct={2.5*mult}%] iter {it+1}/{N_iter}: "
                  f"train={len(train_ind)}, val={len(val_ind)}, test={len(test_ind)}")

            # Conv model
            res = ConvModel(x_flow, x_time, x_beat, y_BP, train_ind, val_ind, test_ind
                          ).model_train(batch=8 * mult, epochs=num_epochs)
            _, y_test_c, y_conv, conv_conv_l, conv_val_l = res

            # WPINN model
            res2 = PDPModel(x_flow, x_time, x_beat, y_BP, train_ind, val_ind, test_ind
                          ).model_train(Ew_model=Ew_model, physics_weight=1,
                                        batch=8 * mult, epochs=num_epochs)
            _, _, y_wpinn, wp_conv_l, wp_phys_l, wp_val_l = res2

            # Evaluate
            if BP_type == 'DBP':
                ref = moving_average(y_test_c[:, 0], w=w)
                conv_pred = moving_average(y_conv[:, 0], w=w)
                wpinn_pred = moving_average(y_wpinn[:, 0], w=w)
            else:
                ref = moving_average(y_test_c.max(axis=-1), w=w)
                conv_pred = moving_average(y_conv.max(axis=-1), w=w)
                wpinn_pred = moving_average(y_wpinn.max(axis=-1), w=w)

            rmse_conv = np.sqrt(np.mean(np.square(ref - conv_pred)))
            rmse_wpinn = np.sqrt(np.mean(np.square(ref - wpinn_pred)))
            rmse_conv_arr.append(rmse_conv)
            rmse_wpinn_arr.append(rmse_wpinn)

            if it == 0:
                all_preds = {'ref': ref, 'conv': conv_pred, 'wpinn': wpinn_pred,
                             'conv_loss': conv_conv_l, 'val_loss': conv_val_l,
                             'wp_conv_loss': wp_conv_l, 'wp_phys_loss': wp_phys_l,
                             'wp_val_loss': wp_val_l}

        rmse_c = np.array(rmse_conv_arr)
        rmse_w = np.array(rmse_wpinn_arr)
        pct_imp = (rmse_c.mean() - rmse_w.mean()) / rmse_c.mean() * 100
        print(f"  >> Conv: {rmse_c.mean():.2f}±{rmse_c.std():.2f} mmHg")
        print(f"  >> WPINN: {rmse_w.mean():.2f}±{rmse_w.std():.2f} mmHg")
        print(f"  >> Improvement: {pct_imp:.1f}%")
        
        results_summary.append({
            'BP_Type': BP_type, 'Percentile': f'{2.5*mult}%',
            'Conv_RMSE_mean': float(rmse_c.mean()),
            'Conv_RMSE_std': float(rmse_c.std()),
            'WPINN_RMSE_mean': float(rmse_w.mean()),
            'WPINN_RMSE_std': float(rmse_w.std()),
            'Improvement_pct': float(pct_imp)
        })

        # Save predictions & loss curves for plotting
        if all_preds:
            pd.DataFrame({'ref': all_preds['ref'], 'conv': all_preds['conv'],
                         'wpinn': all_preds['wpinn']}).to_csv(
                OUTPUTS / f'predictions_{BP_type}_{int(2.5*mult)}pct.csv', index=False)

            # Loss curves
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            axes[0].plot(all_preds['conv_loss'], 'b-', lw=1.5, label='Train MSE')
            axes[0].plot(all_preds['val_loss'], 'b--', lw=1.5, label='Val MSE')
            axes[0].set_title('Conv Model Loss'); axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
            axes[1].plot(all_preds['wp_conv_loss'], 'r-', lw=1.5, label='Train MSE')
            axes[1].plot(all_preds['wp_phys_loss'], 'r:', lw=1.5, label='Physics Loss')
            axes[1].plot(all_preds['wp_val_loss'], 'r--', lw=1.5, label='Val MSE')
            axes[1].set_title('WPINN Model Loss'); axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
            plt.suptitle(f'Training Loss - {BP_type} ({2.5*mult}% percentile)')
            plt.tight_layout()
            plt.savefig(FIGS / f'loss_curves_{BP_type}_{int(2.5*mult)}pct.png', dpi=150)
            plt.close()

results_df = pd.DataFrame(results_summary)
print("\n\n====== FINAL RESULTS ======")
print(results_df.round(2).to_string(index=False))
results_df.to_csv(OUTPUTS / 'model_results.csv', index=False)

# RMSE comparison chart
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(results_df))
width = 0.35
ax.bar(x - width/2, results_df['Conv_RMSE_mean'], width, yerr=results_df['Conv_RMSE_std'],
       label='Conv (Data-Driven)', capsize=3, color='steelblue')
ax.bar(x + width/2, results_df['WPINN_RMSE_mean'], width, yerr=results_df['WPINN_RMSE_std'],
       label='WPINN (Physics-Informed)', capsize=3, color='coral')
labels = [f"{r['BP_Type']}\n{pct}" for _, r in results_df.iterrows() 
          for pct in [r['Percentile']]]
ax.set_xticks(x)
ax.set_xticklabels([f"{r['BP_Type']}\n{r['Percentile']}" for _, r in results_df.iterrows()])
ax.set_ylabel('RMSE (mmHg)')
ax.set_title('Conv vs WPINN Model Performance')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(FIGS / 'model_performance_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: model_performance_comparison.png")

# Prediction scatter plots
for _, row in results_df.iterrows():
    bp_type = row['BP_Type']
    pct = row['Percentile'].replace('%', 'pct')
    csv_file = OUTPUTS / f'predictions_{bp_type}_{pct}.csv'
    if not csv_file.exists():
        continue
    pred_df = pd.read_csv(csv_file)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (pname, color) in zip(axes, [('conv', 'steelblue'), ('wpinn', 'coral')]):
        ax.scatter(pred_df['ref'], pred_df[pname], alpha=0.5, s=10, color=color)
        lims = [min(pred_df['ref'].min(), pred_df[pname].min()),
                max(pred_df['ref'].max(), pred_df[pname].max())]
        ax.plot(lims, lims, 'k--', lw=1, alpha=0.5)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(f'Reference {bp_type} (mmHg)')
        ax.set_ylabel(f'Predicted {bp_type} (mmHg)')
        rmse = np.sqrt(np.mean((pred_df['ref'] - pred_df[pname])**2))
        ax.set_title(f'{"Conv" if pname == "conv" else "WPINN"} — RMSE: {rmse:.2f} mmHg')
        ax.grid(True, alpha=0.3)
    plt.suptitle(f'{bp_type} ({row["Percentile"]}): Reference vs Predicted')
    plt.tight_layout()
    plt.savefig(FIGS / f'prediction_scatter_{bp_type}_{pct}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: prediction_scatter_{bp_type}_{pct}.png")

# Time-series prediction
for _, row in results_df.iterrows():
    bp_type = row['BP_Type']
    pct = row['Percentile'].replace('%', 'pct')
    csv_file = OUTPUTS / f'predictions_{bp_type}_{pct}.csv'
    if not csv_file.exists():
        continue
    pred_df = pd.read_csv(csv_file)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(pred_df['ref'][:200], 'k-', lw=1.5, label='Reference', alpha=0.8)
    ax.plot(pred_df['conv'][:200], 'b-', lw=1, label='Conv', alpha=0.7)
    ax.plot(pred_df['wpinn'][:200], 'r-', lw=1, label='WPINN', alpha=0.7)
    ax.set_xlabel('Beat Index'), ax.set_ylabel(f'{bp_type} (mmHg)')
    ax.set_title(f'{bp_type} ({row["Percentile"]}): Reference vs Models (First 200 beats)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / f'prediction_timeseries_{bp_type}_{pct}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: prediction_timeseries_{bp_type}_{pct}.png")

print("\nWPINN Model Training Complete!")
