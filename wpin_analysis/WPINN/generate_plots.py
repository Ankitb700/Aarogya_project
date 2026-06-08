"""Generate scatter and timeseries plots from model predictions."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIGS = ROOT / 'outputs' / 'figures'
FIGS.mkdir(parents=True, exist_ok=True)

for bp_type in ['SBP', 'DBP']:
    for pct_tag, pct_label in [('2', '2.5%'), ('5', '5.0%')]:
        csv_name = f'predictions_{bp_type}_{pct_tag}pct.csv'
        csv_p = ROOT / 'outputs' / csv_name
        if not csv_p.exists():
            print(f"Skipping: {csv_name} not found")
            continue
        pred_df = pd.read_csv(csv_p)

        # Scatter plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, (pname, color, label) in zip(axes, [
            ('conv', 'steelblue', 'Conv'),
            ('wpinn', 'coral', 'WPINN')
        ]):
            ax.scatter(pred_df['ref'], pred_df[pname], alpha=0.5, s=10, color=color)
            lims = [
                min(pred_df['ref'].min(), pred_df[pname].min()),
                max(pred_df['ref'].max(), pred_df[pname].max())
            ]
            ax.plot(lims, lims, 'k--', lw=1, alpha=0.5)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel(f'Reference {bp_type} (mmHg)')
            ax.set_ylabel(f'Predicted {bp_type} (mmHg)')
            rmse = np.sqrt(np.mean((pred_df['ref'] - pred_df[pname])**2))
            ax.set_title(f'{label} — RMSE: {rmse:.2f} mmHg')
            ax.grid(True, alpha=0.3)
        plt.suptitle(f'{bp_type} ({pct_label}): Reference vs Predicted')
        plt.tight_layout()
        plt.savefig(
            FIGS / f'prediction_scatter_{bp_type}_{pct_tag}pct.png',
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        print(f'Saved: prediction_scatter_{bp_type}_{pct_tag}pct.png')

        # Timeseries plot
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(pred_df['ref'][:200], 'k-', lw=1.5, label='Reference', alpha=0.8)
        ax.plot(pred_df['conv'][:200], 'b-', lw=1, label='Conv', alpha=0.7)
        ax.plot(pred_df['wpinn'][:200], 'r-', lw=1, label='WPINN', alpha=0.7)
        ax.set_xlabel('Beat Index')
        ax.set_ylabel(f'{bp_type} (mmHg)')
        ax.set_title(f'{bp_type} ({pct_label}): Prediction Comparison (first 200 beats)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            FIGS / f'prediction_timeseries_{bp_type}_{pct_tag}pct.png',
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        print(f'Saved: prediction_timeseries_{bp_type}_{pct_tag}pct.png')

print('All plots generated successfully!')
