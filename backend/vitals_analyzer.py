import argparse
import json
import os
import time
import webbrowser
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rppg
from scipy.signal import butter, filtfilt, find_peaks, welch


def extract_vitals(signal, timestamps):
    signal = np.asarray(signal)
    timestamps = np.asarray(timestamps)

    if len(signal) < 300:
        return None

    try:
        duration = timestamps[-1] - timestamps[0]
        if duration <= 0:
            return None

        fps = len(signal) / duration
        nyquist = fps / 2

        b, a = butter(
            3,
            [0.7 / nyquist, 4.0 / nyquist],
            btype="band"
        )

        filtered = filtfilt(b, a, signal)

        peaks, _ = find_peaks(
            filtered,
            distance=int(fps * 0.45),
            prominence=np.std(filtered) * 0.3
        )

        if len(peaks) < 10:
            return None

        peak_times = timestamps[peaks]

        rr_intervals = np.diff(peak_times) * 1000
        rr_intervals = rr_intervals[
            (rr_intervals > 300) &
            (rr_intervals < 2000)
        ]

        if len(rr_intervals) < 5:
            return None

        hr = 60000 / np.mean(rr_intervals)

        sdnn = np.std(rr_intervals, ddof=1)

        rmssd = np.sqrt(
            np.mean(np.square(np.diff(rr_intervals)))
        )

        freqs, psd = welch(
            filtered,
            fs=fps,
            nperseg=min(256, len(filtered))
        )

        resp_mask = (freqs >= 0.1) & (freqs <= 0.5)

        if np.any(resp_mask):
            resp_freq = freqs[resp_mask][np.argmax(psd[resp_mask])]
            rr_bpm = resp_freq * 60
        else:
            rr_bpm = 0

        stress_index = (
            (hr / 100.0) * 40 +
            (100.0 / max(rmssd, 1)) * 30 +
            (100.0 / max(sdnn, 1)) * 30
        )

        stress_index = np.clip(stress_index, 0, 100)

        return {
            "hr": round(float(hr), 2),
            "hrv_rmssd": round(float(rmssd), 2),
            "hrv_sdnn": round(float(sdnn), 2),
            "rr": round(float(rr_bpm), 2),
            "stress_index": round(float(stress_index), 2),
            "duration": round(float(duration), 2),
            "beats": int(len(peaks))
        }

    except Exception as e:
        print("Vital Extraction Error:", e)
        return None


def generate_report(results):
    if len(results) == 0:
        print("No valid measurements collected.")
        return None

    df = pd.DataFrame(results)

    report = {
        "timestamp": str(datetime.now()),
        "recording_duration_sec": float(df["duration"].max()),
        "samples": len(df),
        "avg_hr": round(df["hr"].mean(), 2),
        "min_hr": round(df["hr"].min(), 2),
        "max_hr": round(df["hr"].max(), 2),
        "avg_rmssd": round(df["hrv_rmssd"].mean(), 2),
        "avg_sdnn": round(df["hrv_sdnn"].mean(), 2),
        "avg_rr": round(df["rr"].mean(), 2),
        "avg_stress": round(df["stress_index"].mean(), 2),
        "avg_beats": round(df["beats"].mean(), 2),
    }

    print("\n" + "=" * 60)
    print("FINAL VITALS REPORT")
    print("=" * 60)

    for k, v in report.items():
        print(f"{k:<25}: {v}")

    print("=" * 60)

    df.to_csv("vitals_timeseries.csv", index=False)

    with open("vitals_summary.json", "w") as f:
        json.dump(report, f, indent=4)

    return report


def generate_visual_report(csv_path="vitals_timeseries.csv",
                           json_path="vitals_summary.json",
                           output_dir="vitals_report"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    report = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            report = json.load(f)

    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    # 1. HR trend
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(range(len(df)), df['hr'], 'o-', color='crimson', lw=1.5, ms=4)
    ax.axhline(report.get('avg_hr', df['hr'].mean()), color='gray', ls='--', lw=1, label='Avg')
    ax.set_ylabel('Heart Rate (BPM)')
    ax.set_xlabel('Sample Window')
    ax.set_title('Heart Rate Trend')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out / 'hr_trend.png', dpi=150)
    plt.close(fig)

    # 2. HRV
    fig, ax = plt.subplots(figsize=(10, 3.5))
    x = range(len(df))
    ax.plot(x, df['hrv_rmssd'], 's-', color='royalblue', lw=1.5, ms=4, label='RMSSD (ms)')
    ax.plot(x, df['hrv_sdnn'],  '^-', color='darkorange', lw=1.5, ms=4, label='SDNN (ms)')
    ax.set_ylabel('HRV (ms)')
    ax.set_xlabel('Sample Window')
    ax.set_title('Heart Rate Variability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out / 'hrv_trend.png', dpi=150)
    plt.close(fig)

    # 3. Stress
    fig, ax = plt.subplots(figsize=(10, 3.5))
    colors = ['#2ecc71' if v < 40 else '#f1c40f' if v < 70 else '#e74c3c' for v in df['stress_index']]
    ax.bar(range(len(df)), df['stress_index'], color=colors, width=0.5, edgecolor='none')
    ax.axhline(report.get('avg_stress', df['stress_index'].mean()), color='gray', ls='--', lw=1, label='Avg')
    ax.set_ylabel('Stress Index (0-100)')
    ax.set_xlabel('Sample Window')
    ax.set_title('Stress Level')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', label='Low'),
                       Patch(facecolor='#f1c40f', label='Moderate'),
                       Patch(facecolor='#e74c3c', label='High')]
    ax.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    fig.savefig(out / 'stress_trend.png', dpi=150)
    plt.close(fig)

    # 4. RR
    if 'rr' in df.columns and df['rr'].sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(range(len(df)), df['rr'], 'D-', color='seagreen', lw=1.5, ms=4)
        ax.axhline(report.get('avg_rr', df['rr'].mean()), color='gray', ls='--', lw=1, label='Avg')
        ax.set_ylabel('Respiratory Rate (breaths/min)')
        ax.set_xlabel('Sample Window')
        ax.set_title('Respiratory Rate Trend')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out / 'rr_trend.png', dpi=150)
        plt.close(fig)

    # 5. Dashboard
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
    metrics = [
        ('HR (BPM)', report.get('avg_hr', '—'), f"Min: {report.get('min_hr', '—')}  Max: {report.get('max_hr', '—')}", '#e74c3c'),
        ('HRV RMSSD (ms)', report.get('avg_rmssd', '—'), '', '#3498db'),
        ('HRV SDNN (ms)', report.get('avg_sdnn', '—'), '', '#2980b9'),
        ('Stress Index', report.get('avg_stress', '—'), f"0-100 scale", '#9b59b6'),
        ('Resp. Rate', report.get('avg_rr', '—'), 'breaths/min', '#27ae60'),
        ('Samples', report.get('samples', '—'), '', '#34495e'),
        ('Duration', f"{report.get('recording_duration_sec', 0):.0f}s", '', '#34495e'),
        ('Beats/Window', report.get('avg_beats', '—'), '', '#34495e'),
    ]
    for i, (title, value, sub, color) in enumerate(metrics):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('off')
        ax.add_patch(FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                                      boxstyle="round,pad=0.1",
                                      facecolor=color + '15',
                                      edgecolor=color, lw=2))
        ax.text(0.5, 0.65, str(value), ha='center', va='center',
                fontsize=22, fontweight='bold', color=color, transform=ax.transAxes)
        ax.text(0.5, 0.25, title, ha='center', va='center',
                fontsize=9, color='gray', transform=ax.transAxes)
        if sub:
            ax.text(0.5, 0.08, sub, ha='center', va='center',
                    fontsize=7, color='gray', transform=ax.transAxes)
    fig.suptitle('Vitals Summary Dashboard', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(out / 'dashboard.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 6. HTML report
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vitals Report</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f5f6fa; color: #2c3e50; padding: 20px; }}
.container {{ max-width: 1100px; margin: 0 auto; }}
h1 {{ text-align: center; color: #2c3e50; margin-bottom: 5px; }}
.subtitle {{ text-align: center; color: #7f8c8d; font-size: 14px; margin-bottom: 25px; }}
.summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
           gap: 12px; margin-bottom: 25px; }}
.card {{ background: white; border-radius: 10px; padding: 15px 10px; text-align: center;
         box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.card .value {{ font-size: 24px; font-weight: bold; }}
.card .label {{ font-size: 11px; color: #7f8c8d; margin-top: 4px; }}
footer {{ text-align: center; color: #bdc3c7; font-size: 12px; margin-top: 20px; }}
</style>
</head>
<body>
<div class="container">
<h1>Vitals Analysis Report</h1>
<p class="subtitle">{report.get('timestamp', '')}</p>
<div class="summary">
"""

    card_map = [
        ('avg_hr', 'Avg HR', 'BPM'),
        ('avg_rmssd', 'RMSSD', 'ms'),
        ('avg_sdnn', 'SDNN', 'ms'),
        ('avg_stress', 'Stress', ''),
        ('avg_rr', 'Resp Rate', '/min'),
        ('samples', 'Samples', ''),
        ('recording_duration_sec', 'Duration', 's'),
        ('avg_beats', 'Beats/Win', ''),
    ]
    for key, label, unit in card_map:
        v = report.get(key, '—')
        if isinstance(v, float):
            v = f"{v:.1f}"
        html += f'<div class="card"><div class="value">{v} <span style="font-size:14px;color:#95a5a6">{unit}</span></div><div class="label">{label}</div></div>\n'

    html += f'    <footer>Generated by Vitals Analyzer &mdash; {datetime.now().strftime("%Y-%m-%d %H:%M")}</footer>\n'
    html += '</div>\n</body>\n</html>'

    html_path = out / 'report.html'
    with open(html_path, 'w') as f:
        f.write(html)
    print(f"\nReport saved: {html_path.resolve()}")
    webbrowser.open(f'file://{html_path.resolve()}')
    return html_path


def _has_display():
    return os.name == 'nt' or bool(os.environ.get('DISPLAY'))

def analyze_vitals(video_path=None, duration_sec=30):
    model = rppg.Model()
    results = []

    if video_path:
        # ── Fast path: file-based, no preview window ──
        print(f"Processing: {video_path}")
        hr_result = model.process_video(video_path)
        if hr_result and hr_result.get('hr'):
            signal, timestamps = model.bvp()
            vitals = extract_vitals(signal, timestamps)
            if vitals:
                results.append(vitals)
                print(f"HR: {vitals['hr']} BPM | Stress: {vitals['stress_index']}")
        else:
            print("No valid signal extracted.")
    else:
        # ── Webcam path: warm up then real-time preview ──
        print("Initializing camera (warmup)...")
        dummy = np.random.randint(0, 256, (30, 256, 256, 3), dtype='uint8')
        model.process_video_tensor(dummy, fps=30)
        print("Ready. Opening camera...")

        with model.video_capture(0):
            print(f"Collecting vitals for {duration_sec} seconds...")
            start_time = time.time()
            last_analysis = 0

            for frame, box in model.preview:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                elapsed = time.time() - start_time
                if elapsed >= duration_sec:
                    break
                if time.time() - last_analysis > 5:
                    try:
                        signal, timestamps = model.bvp()
                        vitals = extract_vitals(signal, timestamps)
                        if vitals:
                            results.append(vitals)
                            print(f"[{elapsed:.1f}s] {vitals}")
                    except Exception as e:
                        print(e)
                    last_analysis = time.time()
                if _has_display():
                    if box is not None:
                        y1, y2 = box[0]
                        x1, x2 = box[1]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Elapsed: {elapsed:.1f}s", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Vitals Analyzer", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        if _has_display():
            cv2.destroyAllWindows()

    report = generate_report(results)
    generate_visual_report()
    return report


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Duration in seconds"
    )

    args = parser.parse_args()

    analyze_vitals(
        video_path=args.video,
        duration_sec=args.duration
    )
