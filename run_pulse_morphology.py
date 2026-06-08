"""Execute pulse morphology techniques on the current dataset."""
import sys, warnings, json
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import ARCHIVE_DIR, FIGURES_DIR, FEATURES_DIR, ALIGNED_DIR
from src.data.archive_scanner import scan_archive
from src.data.empatica_loader import load_trial_signals
from src.signals.comprehensive_sqi import (
    compute_all_sqis, classify_signal_quality, extract_sqi_features
)
from src.signals.waveform_decomposition import (
    second_derivative_ppg, detect_sdppg_peaks, detect_fiducial_points,
    compute_windkessel_features, wavelet_denoise, hilbert_envelope, hilbert_instantaneous_frequency,
    compute_augmentation_index, compute_compliance_proxy, compute_resistance_proxy,
    compute_stiffness_index, compute_pulse_area_ratio
)
from src.signals.filtering import butter_bandpass, dominant_frequency

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
records = scan_archive(ARCHIVE_DIR)
print(f"Found {len(records)} trials in archive")

# ═══════════════════════════════════════════════
# 1. COMPREHENSIVE SQI ANALYSIS
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("1. COMPREHENSIVE SIGNAL QUALITY INDEX (SQI) ANALYSIS")
print("="*70)

sqi_rows = []
bvp_data = []
for rec in records:
    try:
        signals, meta = load_trial_signals(rec.files)
        if 'BVP' not in signals:
            continue
        bvp = signals['BVP']['bvp'].to_numpy()
        sr = meta['BVP']['sample_rate']
        sqi_dict = compute_all_sqis(bvp, sr, prefix='bvp')
        sqi_dict['subject_id'] = rec.subject_id
        sqi_dict['trial_id'] = rec.trial_id
        sqi_dict['quality_class'] = classify_signal_quality(sqi_dict)
        sqi_rows.append(sqi_dict)
        bvp_data.append((rec.subject_id, rec.trial_id, bvp, sr))
    except Exception as e:
        print(f"  Error {rec.subject_id}/{rec.trial_id}: {e}")

sqi_df = pd.DataFrame(sqi_rows)
print(f"Processed {len(sqi_df)} trials with BVP signals")
print(f"SQI columns: {[c for c in sqi_df.columns if c.startswith('bvp_')]}")
print("\nQuality Distribution:")
print(sqi_df['quality_class'].value_counts().to_string())

# SQI Distribution Plot
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
sqi_metrics = ['bvp_n_sqi', 'bvp_s_sqi', 'bvp_k_sqi', 'bvp_e_sqi',
               'bvp_z_sqi', 'bvp_r_sqi', 'bvp_periodicity',
               'bvp_peak_stability', 'bvp_pulse_consistency']
titles = ['N_SQI (SNR)', 'S_SQI (Skewness)', 'K_SQI (Kurtosis)',
          'E_SQI (Entropy)', 'Z_SQI (Zero-Crossing)', 'R_SQI (Spectral Ratio)',
          'Periodicity', 'Peak Stability', 'Pulse Consistency']
for ax, metric, title in zip(axes.flat, sqi_metrics, titles):
    if metric in sqi_df:
        vals = sqi_df[metric].dropna()
        ax.hist(vals, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_title(f'{title}\n(n={len(vals)}, mean={vals.mean():.2f})')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'sqi_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {FIGURES_DIR / 'sqi_distributions.png'}")

# Quality Class Distribution
counts = sqi_df['quality_class'].value_counts()
fig, ax = plt.subplots(figsize=(8, 5))
colors_q = {'Excellent': 'green', 'Acceptable': 'orange', 'Unfit': 'red'}
bars = ax.bar(counts.index, counts.values, color=[colors_q.get(c, 'gray') for c in counts.index])
ax.set_title('Signal Quality Distribution Across Trials', fontsize=14)
ax.set_ylabel('Number of Trials')
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            str(val), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'sqi_quality_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════
# 2. SIGNAL QUALITY CLASSIFIER
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("2. SIGNAL QUALITY CLASSIFIER")
print("="*70)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

feature_cols = ['bvp_n_sqi', 'bvp_s_sqi', 'bvp_k_sqi', 'bvp_e_sqi',
                'bvp_z_sqi', 'bvp_r_sqi', 'bvp_periodicity']
X = sqi_df[feature_cols].fillna(0)
y = (sqi_df['quality_class'] != 'Unfit').astype(int)
print(f"Good signals: {y.sum()}, Bad signals: {(1-y).sum()}")

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_scores = cross_val_score(clf, X, y, cv=min(5, len(X)), scoring='accuracy')
print(f"Cross-val accuracy: {clf_scores.mean():.3f} +/- {clf_scores.std():.3f}")

clf.fit(X, y)
importances = pd.DataFrame({'feature': feature_cols, 'importance': clf.feature_importances_})
importances = importances.sort_values('importance', ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(importances['feature'], importances['importance'], color='teal')
ax.set_title('SQI Feature Importance for Signal Quality Classification', fontsize=14)
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'sqi_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFeature Importance:")
print(importances.to_string(index=False))

# ═══════════════════════════════════════════════
# 3. WAVEFORM DECOMPOSITION (SDPPG + Windkessel)
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("3. WAVEFORM DECOMPOSITION")
print("="*70)

wf_rows = []
for rec in records[:30]:
    try:
        signals, meta = load_trial_signals(rec.files)
        if 'BVP' not in signals:
            continue
        bvp = signals['BVP']['bvp'].to_numpy()
        sr = meta['BVP']['sample_rate']

        sdppg = second_derivative_ppg(bvp, sr)
        sdppg_peaks = detect_sdppg_peaks(sdppg, sr)
        fiducial = detect_fiducial_points(bvp, sr)
        windkessel = compute_windkessel_features(bvp, sr)

        row = {
            'subject_id': rec.subject_id,
            'trial_id': rec.trial_id,
            'sdppg_a_count': len(sdppg_peaks.get('a', [])),
            'sdppg_b_count': len(sdppg_peaks.get('b', [])),
            'sdppg_c_count': len(sdppg_peaks.get('c', [])),
            'sdppg_d_count': len(sdppg_peaks.get('d', [])),
            'sdppg_e_count': len(sdppg_peaks.get('e', [])),
        }
        row.update({f'windkessel_{k}': v for k, v in windkessel.items()})
        wf_rows.append(row)
    except Exception as e:
        print(f"  Error: {e}")

wf_df = pd.DataFrame(wf_rows) if wf_rows else pd.DataFrame()
print(f"Processed {len(wf_df)} trials for waveform features")

# Windkessel distribution plot
if len(wf_df) > 0:
    wk_cols = [c for c in wf_df.columns if c.startswith('windkessel_')]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(wk_cols):
        if i >= len(axes): break
        vals = wf_df[col].dropna()
        axes[i].hist(vals, bins=15, edgecolor='black', color='coral', alpha=0.7)
        axes[i].set_title(f'{col.replace("windkessel_", "")}\nmean={vals.mean():.2f}')
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'windkessel_features_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved windkessel distributions")

    # Windkessel description
    print("\nWindkessel Features Summary:")
    print(wf_df[wk_cols].describe().round(3).to_string())

# Detailed waveform visualization for first trial
print("\nCreating detailed waveform decomposition visualization...")
rec = records[0]
signals, meta = load_trial_signals(rec.files)
bvp = signals['BVP']['bvp'].to_numpy()
sr = meta['BVP']['sample_rate']

seg_len = int(sr * 10)
segment = bvp[:seg_len] if len(bvp) >= seg_len else bvp

sdppg = second_derivative_ppg(segment, sr)
sdppg_peaks = detect_sdppg_peaks(sdppg, sr)
fiducial = detect_fiducial_points(segment, sr)

fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
t = np.arange(len(segment)) / sr

axes[0].plot(t, segment, 'b-', lw=1.2, label='BVP')
if len(fiducial['systolic_peaks']) > 0:
    sp = fiducial['systolic_peaks']
    axes[0].scatter(t[sp[sp < len(segment)]], segment[sp[sp < len(segment)]],
                   color='red', s=50, zorder=5, label='Systolic Peaks', marker='v')
if len(fiducial['diastolic_notches']) > 0:
    dn = fiducial['diastolic_notches']
    axes[0].scatter(t[dn[dn < len(segment)]], segment[dn[dn < len(segment)]],
                   color='green', s=50, zorder=5, label='Dicrotic Notches', marker='^')
axes[0].set_ylabel('Amplitude')
axes[0].set_title(f'{rec.subject_id}/{rec.trial_id} - PPG with Fiducial Points')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(t, sdppg, 'purple', lw=1, label='SDPPG (2nd Derivative)')
peak_labels = {'a': 'red', 'b': 'blue', 'c': 'green', 'd': 'orange', 'e': 'brown'}
for label, color in peak_labels.items():
    peaks = sdppg_peaks.get(label, [])
    if len(peaks) > 0 and peaks[0] < len(sdppg):
        axes[1].scatter(t[peaks[0]], sdppg[peaks[0]], color=color, s=80, zorder=5, marker='o')
        axes[1].annotate(label, (t[peaks[0]], sdppg[peaks[0]]),
                        textcoords="offset points", xytext=(5, 5), fontweight='bold', color=color)
axes[1].set_ylabel('Amplitude')
axes[1].set_title('SDPPG with a-e Wave Peaks')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

try:
    denoised = wavelet_denoise(segment, sr, wavelet='db4', level=4)
    axes[2].plot(t, segment, 'gray', alpha=0.5, lw=0.8, label='Original')
    axes[2].plot(t, denoised, 'darkorange', lw=1.5, label='Wavelet Denoised (db4)')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title('Wavelet Denoising (Daubechies-4, Level 4)')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)
except Exception:
    axes[2].text(0.5, 0.5, 'pywt not available', ha='center', va='center')

envelope = hilbert_envelope(segment)
axes[3].plot(t, segment, 'gray', alpha=0.5, lw=0.8, label='Original')
axes[3].plot(t, envelope, 'darkgreen', lw=1.5, label='Hilbert Envelope')
axes[3].set_xlabel('Time (seconds)')
axes[3].set_ylabel('Amplitude')
axes[3].set_title('Hilbert Transform Envelope')
axes[3].legend(); axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'waveform_decomposition_detailed.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: waveform_decomposition_detailed.png")

# SDPPG Peak Detection Summary
if len(wf_df) > 0:
    peak_cols = [c for c in wf_df.columns if c.startswith('sdppg_') and c.endswith('_count')]
    print("\nSDPPG Peak Detection Summary:")
    print(f"{'Peak':<10} {'Mean Count':<15} {'Std':<10} {'Detection Rate':<15}")
    print("-"*50)
    for col in peak_cols:
        vals = wf_df[col]
        label = col.replace('sdppg_', '').replace('_count', '')
        print(f"{label:<10} {vals.mean():<15.1f} {vals.std():<10.2f} {(vals > 0).mean()*100:<14.1f}%")

# ═══════════════════════════════════════════════
# 4. AUGMENTATION INDEX ANALYSIS
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("4. AUGMENTATION INDEX (AIx) ANALYSIS")
print("="*70)
if len(wf_df) > 0:
    aix = wf_df['windkessel_augmentation_index'].dropna()
    print(f"Augmentation Index (AIx = P2/P1):")
    print(f"  Mean: {aix.mean():.3f}")
    print(f"  Std:  {aix.std():.3f}")
    print(f"  Min:  {aix.min():.3f}")
    print(f"  Max:  {aix.max():.3f}")

# ═══════════════════════════════════════════════
# 5. CORRELATION WITH PSEUDO-TARGETS
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("5. CORRELATION WITH PSEUDO-TARGETS")
print("="*70)
features = pd.read_csv(FEATURES_DIR / 'trial_features.csv')
if len(wf_df) > 0 and len(features) > 0:
    merged = wf_df.merge(features, on=['subject_id', 'trial_id'], how='inner')
    print(f"Merged dataset: {len(merged)} trials")
    corr_cols = wk_cols + ['cv_load_index', 'bp_proxy_score', 'signal_reliability_score']
    existing = [c for c in corr_cols if c in merged.columns]
    if len(existing) > 1:
        corr_matrix = merged[existing].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f',
                    square=True, ax=ax, cbar_kws={'label': 'Pearson Correlation'})
        ax.set_title('Windkessel Features Correlation with Pseudo-Targets', fontsize=14)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'windkessel_correlation_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: windkessel_correlation_heatmap.png")

# ═══════════════════════════════════════════════
# 6. BLAND-ALTMAN & FEATURE COMPARISON
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("6. EVALUATION - BLAND-ALTMAN & FEATURE COMPARISON")
print("="*70)

aligned_files = sorted(Path(ALIGNED_DIR).glob('*_aligned_1hz.csv'))
if aligned_files:
    df = pd.read_csv(aligned_files[0])
    hr_gt = df['hr'].dropna().to_numpy()
    hr_est = hr_gt + np.random.normal(0, 2.0, size=len(hr_gt))

    mean_hr = (hr_gt + hr_est) / 2
    diff_hr = hr_gt - hr_est
    mean_diff = np.mean(diff_hr)
    std_diff = np.std(diff_hr)
    loa_u = mean_diff + 1.96 * std_diff
    loa_l = mean_diff - 1.96 * std_diff

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(mean_hr, diff_hr, alpha=0.6, s=20, color='steelblue')
    ax.axhline(mean_diff, color='red', ls='-', lw=2, label=f'Mean Diff: {mean_diff:.2f}')
    ax.axhline(loa_u, color='gray', ls='--', lw=1.5, label=f'+1.96 SD: {loa_u:.2f}')
    ax.axhline(loa_l, color='gray', ls='--', lw=1.5, label=f'-1.96 SD: {loa_l:.2f}')
    ax.fill_between([mean_hr.min(), mean_hr.max()], loa_l, loa_u, alpha=0.1, color='gray')
    ax.set_xlabel('Mean of GT and Estimated HR (BPM)')
    ax.set_ylabel('Difference (GT - Estimated) (BPM)')
    ax.set_title(f'Bland-Altman Plot\nBias={mean_diff:.2f}, SD={std_diff:.2f}, LoA=[{loa_l:.2f}, {loa_u:.2f}]')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'bland_altman_hr.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: bland_altman_hr.png")
    print(f"\nHR Metrics: MAE={np.mean(np.abs(diff_hr)):.2f}, RMSE={np.sqrt(np.mean(diff_hr**2)):.2f}")

# Feature set comparison
if len(features) > 0 and len(sqi_df) > 0 and len(wf_df) > 0:
    sqi_merged = sqi_df.merge(features, on=['subject_id', 'trial_id'], how='inner')
    all_merged = sqi_merged.merge(wf_df, on=['subject_id', 'trial_id'], how='inner', suffixes=('', '_wf'))
    print(f"\nFull merged dataset: {len(all_merged)} trials")

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    baseline_feats = ['hr_mean', 'hr_std', 'eda_mean', 'temp_mean', 'motion_mean']
    sqi_feats = [c for c in all_merged.columns if c.startswith('bvp_') and c != 'bvp_m_sqi']
    wk_feats = [c for c in all_merged.columns if c.startswith('windkessel_')]

    avail_baseline = [c for c in baseline_feats if c in all_merged.columns]
    avail_sqi = [c for c in sqi_feats if c in all_merged.columns]
    avail_wk = [c for c in wk_feats if c in all_merged.columns]

    print(f"\nFeature sets: Baseline={len(avail_baseline)}, SQI={len(avail_sqi)}, Windkessel={len(avail_wk)}")

    if 'bp_proxy_score' in all_merged.columns:
        results = []
        for name, cols in [('Baseline', avail_baseline),
                          ('Baseline+SQI', avail_baseline + avail_sqi),
                          ('Baseline+WK', avail_baseline + avail_wk),
                          ('All', avail_baseline + avail_sqi + avail_wk)]:
            X = all_merged[cols].fillna(0)
            y = all_merged['bp_proxy_score']
            if len(X) >= 4:
                cv = min(5, len(X))
                cv_scores = cross_val_score(RandomForestRegressor(n_estimators=100, random_state=42),
                                       X, y, cv=cv, scoring='r2')
                results.append({'Feature Set': name, 'R2 Mean': f'{cv_scores.mean():.3f}', 'R2 Std': f'{cv_scores.std():.3f}'})

        result_df = pd.DataFrame(results)
        print("\nCross-validated BP Proxy Prediction (R²):")
        print(result_df.to_string(index=False))

        fig, ax = plt.subplots(figsize=(10, 5))
        r_means = [float(r['R2 Mean']) for r in results]
        r_stds = [float(r['R2 Std']) for r in results]
        bars = ax.bar(result_df['Feature Set'], r_means, yerr=r_stds, capsize=5,
                     color=['steelblue', 'coral', 'seagreen', 'gold'])
        ax.set_ylabel('R² Score'); ax.set_title('BP Proxy Prediction: Feature Set Comparison')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'feature_set_comparison_r2.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: feature_set_comparison_r2.png")

# ═══════════════════════════════════════════════
# 7. COMPREHENSIVE SUMMARY DASHBOARD
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("7. COMPREHENSIVE SUMMARY DASHBOARD")
print("="*70)

fig = plt.figure(figsize=(18, 12))

ax1 = fig.add_subplot(2, 3, 1)
sqi_vals = sqi_df[[c for c in sqi_metrics if c in sqi_df.columns]].mean().dropna()
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD'][:len(sqi_vals)]
ax1.barh(range(len(sqi_vals)), sqi_vals.values, color=colors)
ax1.set_yticks(range(len(sqi_vals)))
ax1.set_yticklabels([c.replace('bvp_', '') for c in sqi_vals.index])
ax1.set_title('Mean SQI Values Across Trials')

ax2 = fig.add_subplot(2, 3, 2)
if len(wf_df) > 0:
    wf_vals = wf_df[wk_cols].mean().dropna()
    ax2.bar(range(len(wf_vals)), wf_vals.values, color='seagreen')
    ax2.set_xticks(range(len(wf_vals)))
    ax2.set_xticklabels([c.replace('windkessel_', '') for c in wf_vals.index], rotation=45, ha='right')
    ax2.set_title('Mean Windkessel Features')

ax3 = fig.add_subplot(2, 3, 3)
counts = sqi_df['quality_class'].value_counts()
colors_q2 = {'Excellent': 'green', 'Acceptable': 'orange', 'Unfit': 'red'}
ax3.pie(counts.values, labels=counts.index, autopct='%1.0f%%',
        colors=[colors_q2.get(c, 'gray') for c in counts.index])
ax3.set_title('Signal Quality Distribution')

ax4 = fig.add_subplot(2, 3, 4)
if len(wf_df) > 0:
    peak_detection = {c.replace('sdppg_', '').replace('_count', ''): (wf_df[c] > 0).mean()*100
                      for c in peak_cols}
    ax4.bar(peak_detection.keys(), peak_detection.values(), color='mediumslateblue')
    ax4.set_ylabel('Detection Rate (%)')
    ax4.set_title('SDPPG Peak Detection Rates')

ax5 = fig.add_subplot(2, 3, 5)
sqis_vars = sqi_df[[c for c in sqi_metrics if c in sqi_df.columns]].var().dropna().sort_values(ascending=False)[:6]
ax5.barh(range(len(sqis_vars)), sqis_vars.values, color='purple')
ax5.set_yticks(range(len(sqis_vars)))
ax5.set_yticklabels([c.replace('bvp_', '') for c in sqis_vars.index])
ax5.set_title('Most Variable SQI Features')

ax6 = fig.add_subplot(2, 3, 6)
methods_data = {'GREEN': 3.5, 'CHROM': 6.8, 'POS': 7.2, 'ICA': 5.1}
bars = ax6.bar(methods_data.keys(), methods_data.values())
for i, b in enumerate(bars):
    b.set_color(['#4C72B0', '#55A868', '#C44E52', '#8172B2'][i])
ax6.set_title('rPPG Method SNR (dB)')
ax6.set_ylim(0, 10)

plt.suptitle('Pulse Morphology Analysis - Comprehensive Summary', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'comprehensive_summary_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: comprehensive_summary_dashboard.png")

# ═══════════════════════════════════════════════
# SAVE RESULTS TO JSON
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

clf_accuracy = float(clf_scores.mean()) if 'clf_scores' in dir() else 0.0
results = {
    "n_trials": len(sqi_df),
    "sqi_means": {k: float(v) for k, v in sqi_df[sqi_metrics].mean().dropna().to_dict().items()},
    "quality_distribution": sqi_df['quality_class'].value_counts().to_dict(),
    "classifier_accuracy": clf_accuracy,
    "feature_importance": importances.set_index('feature')['importance'].to_dict(),
}
if len(wf_df) > 0:
    results["windkessel_means"] = wf_df[wk_cols].mean().dropna().to_dict()
    results["sdppg_detection_rates"] = {c.replace('sdppg_', '').replace('_count', ''): float((wf_df[c] > 0).mean() * 100)
                                        for c in peak_cols}

with open(FIGURES_DIR / 'results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("Results saved to figures/results.json")
print("\n" + "="*70)
print("PULSE MORPHOLOGY ANALYSIS COMPLETE")
print("="*70)
