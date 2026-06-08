from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
DIRS = {
    "executive": REPORTS / "executive_summary",
    "comparisons": REPORTS / "model_comparisons",
    "success": REPORTS / "success_cases",
    "failure": REPORTS / "failure_cases",
    "workflow": REPORTS / "workflow_visuals",
    "presentation": REPORTS / "presentation_assets",
    "dashboard": REPORTS / "dashboard_visuals",
    "pretrained": REPORTS / "pretrained_analysis",
    "custom": REPORTS / "custom_model_results",
}

COLORS = {
    "ink": "#1f2933",
    "muted": "#5b6777",
    "blue": "#2f80ed",
    "green": "#27ae60",
    "orange": "#f2994a",
    "red": "#d64545",
    "teal": "#11999e",
    "lavender": "#7b61ff",
    "paper": "#f7f9fc",
    "line": "#d6dde8",
}


def ensure_dirs() -> None:
    for path in DIRS.values():
        path.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def metric_summary() -> dict[str, float | str]:
    ml = pd.read_csv(ROOT / "outputs/predictions/ml_random_sample_predictions.csv")
    dl = pd.read_csv(ROOT / "outputs/predictions/dl_random_sample_predictions.csv")
    wpin = pd.read_csv(ROOT / "wpin_analysis/WPINN/outputs/model_results.csv")
    with open(ROOT / "outputs/figures/results.json", "r", encoding="utf-8") as handle:
        pulse = json.load(handle)
    with open(ROOT / "rppg_inference_vbpe/research_output/final_summary.json", "r", encoding="utf-8") as handle:
        roi_classifier = json.load(handle)

    ml_hr_mae = float((ml["actual_hr"] - ml["predicted_hr"]).abs().mean())
    dl_hr_mae = float((dl["actual_hr"] - dl["predicted_hr"]).abs().mean())
    ml_bp_mae = float((ml["actual_bp_proxy"] - ml["predicted_bp_proxy"]).abs().mean())
    dl_bp_mae = float((dl["actual_bp_proxy"] - dl["predicted_bp_proxy"]).abs().mean())
    cv_load_acc = float((dl["actual_cv_load_class"] == dl["predicted_cv_load_class"]).mean())
    best_wpin = wpin.loc[wpin["Improvement_pct"].idxmax()]

    return {
        "n_trials": int(pulse["n_trials"]),
        "quality": ", ".join(f"{k}: {v}" for k, v in pulse["quality_distribution"].items()),
        "dl_hr_mae": dl_hr_mae,
        "ml_hr_mae": ml_hr_mae,
        "dl_bp_mae": dl_bp_mae,
        "ml_bp_mae": ml_bp_mae,
        "cv_load_acc": cv_load_acc,
        "real_roi_acc": float(roi_classifier["real_RF_Acc"]),
        "best_wpin_label": f"{best_wpin['BP_Type']} {best_wpin['Percentile']}",
        "best_wpin_improvement": float(best_wpin["Improvement_pct"]),
        "sdppg_rate": float(np.mean(list(pulse["sdppg_detection_rates"].values()))),
    }


def save_scorecard(summary: dict[str, float | str]) -> Path:
    out = DIRS["executive"] / "executive_scorecard.png"
    fig, ax = plt.subplots(figsize=(14, 7), dpi=160)
    ax.set_axis_off()
    fig.patch.set_facecolor(COLORS["paper"])
    ax.text(0.04, 0.91, "Executive Summary", fontsize=28, weight="bold", color=COLORS["ink"])
    ax.text(
        0.04,
        0.85,
        "Non-contact physiology analytics from existing videos, wearable signals, and model outputs",
        fontsize=13,
        color=COLORS["muted"],
    )

    cards = [
        ("Best Overall Signal Quality", str(summary["quality"]), "All analyzed trials are usable for reporting", COLORS["green"]),
        ("Heart-Rate Error", f"{summary['dl_hr_mae']:.1f} bpm", "Deep model held-out sample MAE", COLORS["blue"]),
        ("ROI Reliability", f"{summary['real_roi_acc']:.0%}", "Face/hand region quality classifier accuracy", COLORS["teal"]),
        ("Largest BP Gain", f"{summary['best_wpin_improvement']:.1f}%", f"WPINN improvement on {summary['best_wpin_label']}", COLORS["orange"]),
        ("Pulse Markers Found", f"{summary['sdppg_rate']:.0f}%", "A-E markers found in all trials", COLORS["lavender"]),
        ("Recommended Direction", "WPINN + ROI QA", "Use quality checks before BP/HR reporting", COLORS["red"]),
    ]
    positions = [(0.04, 0.54), (0.36, 0.54), (0.68, 0.54), (0.04, 0.22), (0.36, 0.22), (0.68, 0.22)]
    for (title, value, subtitle, color), (x, y) in zip(cards, positions):
        rect = plt.Rectangle((x, y), 0.28, 0.22, transform=ax.transAxes, facecolor="white", edgecolor=COLORS["line"], linewidth=1.2)
        ax.add_patch(rect)
        ax.add_patch(plt.Rectangle((x, y + 0.205), 0.28, 0.015, transform=ax.transAxes, facecolor=color, edgecolor=color))
        ax.text(x + 0.02, y + 0.155, title, transform=ax.transAxes, fontsize=12, color=COLORS["muted"], weight="bold")
        value_size = 16 if len(str(value)) >= 14 else 24
        ax.text(x + 0.02, y + 0.09, value, transform=ax.transAxes, fontsize=value_size, color=COLORS["ink"], weight="bold")
        ax.text(x + 0.02, y + 0.035, subtitle, transform=ax.transAxes, fontsize=10, color=COLORS["muted"], wrap=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_model_comparison(summary: dict[str, float | str]) -> list[Path]:
    out1 = DIRS["comparisons"] / "model_comparison_overview.png"
    out2 = DIRS["comparisons"] / "wpinn_rmse_improvement.png"
    ml_dl = pd.DataFrame(
        {
            "Model": ["Classic ML", "Deep Multimodal"],
            "Heart-rate error (bpm)": [summary["ml_hr_mae"], summary["dl_hr_mae"]],
            "BP proxy error": [summary["ml_bp_mae"], summary["dl_bp_mae"]],
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=160)
    for ax, metric, color in zip(axes, ["Heart-rate error (bpm)", "BP proxy error"], [COLORS["blue"], COLORS["orange"]]):
        bars = ax.bar(ml_dl["Model"], ml_dl[metric], color=[color, COLORS["green"]])
        ax.set_title(metric, fontsize=15, weight="bold")
        ax.set_ylabel("Lower is better")
        ax.grid(axis="y", alpha=0.25)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=11)
    fig.suptitle("Held-Out Prediction Comparison", fontsize=20, weight="bold")
    fig.tight_layout()
    fig.savefig(out1, bbox_inches="tight")
    plt.close(fig)

    wpin = pd.read_csv(ROOT / "wpin_analysis/WPINN/outputs/model_results.csv")
    labels = wpin["BP_Type"] + " " + wpin["Percentile"].astype(str)
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    ax.bar(x - width / 2, wpin["Conv_RMSE_mean"], width, label="Baseline Conv Model", color=COLORS["muted"])
    ax.bar(x + width / 2, wpin["WPINN_RMSE_mean"], width, label="Custom WPINN", color=COLORS["green"])
    for idx, gain in enumerate(wpin["Improvement_pct"]):
        ax.text(idx, max(wpin.loc[idx, "Conv_RMSE_mean"], wpin.loc[idx, "WPINN_RMSE_mean"]) + 0.7, f"{gain:.1f}% better", ha="center", fontsize=10, color=COLORS["ink"])
    ax.set_xticks(x, labels)
    ax.set_ylabel("RMSE in mmHg, lower is better")
    ax.set_title("Custom BP Model Improves Error Across Test Settings", fontsize=18, weight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out2, bbox_inches="tight")
    plt.close(fig)
    return [out1, out2]


def save_roi_quality() -> Path:
    out = DIRS["dashboard"] / "roi_signal_quality.png"
    roi = pd.read_csv(ROOT / "rppg_inference_vbpe/research_output/roi_summary_table.csv")
    roi = roi.sort_values("SNR_time", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    colors = [COLORS["red"] if v < 20 else COLORS["orange"] if v < 30 else COLORS["green"] for v in roi["SNR_time"]]
    ax.barh(roi["ROI"].str.replace("_", " ").str.title(), roi["SNR_time"], color=colors)
    ax.axvline(20, color=COLORS["line"], linestyle="--")
    ax.set_xlabel("Signal clarity score, higher is better")
    ax.set_title("Best Face Regions For Reliable Video Measurement", fontsize=18, weight="bold")
    ax.grid(axis="x", alpha=0.25)
    for y, value in enumerate(roi["SNR_time"]):
        ax.text(value + 1, y, f"{value:.1f}", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def read_video_frame(path: Path, frame_number: int | None = None) -> Image.Image | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = frame_number if frame_number is not None else max(count // 3, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def cover_image(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    sw, sh = size
    scale = max(sw / w, sh / h)
    resized = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    left = (resized.width - sw) // 2
    top = (resized.height - sh) // 2
    return resized.crop((left, top, left + sw, top + sh))


def fit_image(img: Image.Image, size: tuple[int, int], fill: str = "white") -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    sw, sh = size
    scale = min(sw / w, sh / h)
    resized = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    canvas = Image.new("RGB", size, fill)
    canvas.paste(resized, ((sw - resized.width) // 2, (sh - resized.height) // 2))
    return canvas


def label_tile(base: Image.Image, title: str, subtitle: str, accent: str) -> Image.Image:
    draw = ImageDraw.Draw(base, "RGBA")
    draw.rectangle((0, base.height - 95, base.width, base.height), fill=(0, 0, 0, 160))
    draw.rectangle((0, base.height - 95, 12, base.height), fill=accent)
    draw.text((24, base.height - 82), title, fill="white", font=load_font(22, True))
    draw.text((24, base.height - 48), subtitle, fill=(235, 240, 245), font=load_font(15))
    return base


def save_visual_examples() -> list[Path]:
    out_success = DIRS["success"] / "video_measurement_examples.png"
    out_failure = DIRS["failure"] / "client_friendly_challenge_gallery.png"
    out_frame = DIRS["presentation"] / "representative_video_frame.png"

    items = [
        (ROOT / "rppg_inference_vbpe/Input_Photos/SL428.jpg", "Input Photo", "Original capture used by the pipeline", COLORS["blue"]),
        (ROOT / "rppg_inference_vbpe/Results/MP_Videos_MP4/Face/SL428_face_output.mp4", "Face Tracking", "Landmarks locate measurement regions", COLORS["green"]),
        (ROOT / "rppg_inference_vbpe/Results/MP_Videos_MP4/Hand/SL428_hand_output.mp4", "Hand Tracking", "Hand signal provides an alternate source", COLORS["teal"]),
    ]
    canvas = Image.new("RGB", (1500, 560), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 26), "Visual Measurement Examples", fill=COLORS["ink"], font=load_font(34, True))
    for idx, (path, title, subtitle, accent) in enumerate(items):
        if path.suffix.lower() in {".mp4", ".avi", ".mov"}:
            img = read_video_frame(path)
        else:
            img = Image.open(path)
        tile = cover_image(img, (440, 380))
        tile = label_tile(tile, title, subtitle, accent)
        canvas.paste(tile, (40 + idx * 485, 110))
    canvas.save(out_success)

    frame = read_video_frame(ROOT / "rppg_inference_vbpe/Input_Videos/SL428.mp4")
    if frame is not None:
        cover_image(frame, (1000, 600)).save(out_frame)

    roi = pd.read_csv(ROOT / "rppg_inference_vbpe/research_output/per_roi_sqi_all.csv")
    low = roi.sort_values(["Tmpl_corr", "SNR_freq"], ascending=True).head(3)
    labels = [
        ("Lower pattern match", "Use quality checks before reporting results"),
        ("Weak frequency signal", "Prefer forehead or nose bridge when available"),
        ("High movement risk", "Ask for steadier framing when possible"),
    ]
    frame_sources = {
        "SL413.mp4": ROOT / "rppg_inference_vbpe/Input_Videos/SL413.mp4",
        "SL421.mp4": ROOT / "rppg_inference_vbpe/Input_Videos/SL421.mp4",
        "SL426.mp4": ROOT / "rppg_inference_vbpe/Input_Videos/SL426.mp4",
        "SL428.mp4": ROOT / "rppg_inference_vbpe/Input_Videos/SL428.mp4",
        "SL377.mp4": ROOT / "rppg_inference_vbpe/Input_Videos/SL377.mp4",
    }
    canvas = Image.new("RGB", (1500, 610), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 28), "Client-Friendly Challenge Gallery", fill=COLORS["ink"], font=load_font(34, True))
    draw.text((42, 72), "These are not failed experiments; they are situations where automated quality checks protect the final result.", fill=COLORS["muted"], font=load_font(18))
    for idx, (_, row) in enumerate(low.iterrows()):
        frame = read_video_frame(frame_sources.get(row["Video"], next(iter(frame_sources.values()))))
        tile = cover_image(frame, (440, 350))
        title, subtitle = labels[idx]
        detail = f"{str(row['ROI']).replace('_', ' ').title()} on {row['Video']}"
        tile = label_tile(tile, title, detail, COLORS["red"] if idx == 0 else COLORS["orange"])
        canvas.paste(tile, (40 + idx * 485, 130))
        draw.text((55 + idx * 485, 500), subtitle, fill=COLORS["ink"], font=load_font(17, True))
    canvas.save(out_failure)
    return [out_success, out_failure, out_frame]


def save_workflow() -> Path:
    out = DIRS["workflow"] / "operational_workflow.png"
    steps = [
        ("Video + Wearable Data", "Raw recordings and sensor streams"),
        ("Quality Check", "Choose reliable face or hand regions"),
        ("Signal Extraction", "Turn video color changes into pulse signals"),
        ("Model Prediction", "Estimate heart rate and BP-related scores"),
        ("Client Output", "Dashboard, findings, and recommendation"),
    ]
    fig, ax = plt.subplots(figsize=(15, 4.8), dpi=160)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    ax.text(0.03, 0.88, "Operational Workflow", transform=ax.transAxes, fontsize=25, weight="bold", color=COLORS["ink"])
    for idx, (title, subtitle) in enumerate(steps):
        x = 0.04 + idx * 0.19
        ax.add_patch(plt.Rectangle((x, 0.38), 0.15, 0.26, transform=ax.transAxes, facecolor=COLORS["paper"], edgecolor=COLORS["line"], linewidth=1.2))
        ax.text(x + 0.075, 0.55, str(idx + 1), transform=ax.transAxes, ha="center", fontsize=18, weight="bold", color=COLORS["blue"])
        ax.text(x + 0.075, 0.47, title, transform=ax.transAxes, ha="center", fontsize=11, weight="bold", color=COLORS["ink"])
        ax.text(x + 0.075, 0.41, subtitle, transform=ax.transAxes, ha="center", fontsize=8.5, color=COLORS["muted"], wrap=True)
        if idx < len(steps) - 1:
            ax.annotate("", xy=(x + 0.18, 0.51), xytext=(x + 0.15, 0.51), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", lw=2, color=COLORS["muted"]))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_rppg_bp_results() -> Path:
    out = DIRS["pretrained"] / "rppg_vbpe_bp_summary.png"
    subjects = ["SL377", "SL413", "SL421", "SL426", "SL428"]
    sbp = pd.read_csv(ROOT / "rppg_inference_vbpe/SBP_new.csv", comment="#", header=None).iloc[:, 0].astype(float)
    dbp = pd.read_csv(ROOT / "rppg_inference_vbpe/DBP_new.csv", comment="#", header=None).iloc[:, 0].astype(float)
    x = np.arange(len(subjects))
    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    ax.plot(x, sbp, marker="o", linewidth=3, color=COLORS["red"], label="Systolic BP")
    ax.plot(x, dbp, marker="o", linewidth=3, color=COLORS["blue"], label="Diastolic BP")
    ax.fill_between(x, dbp, sbp, color=COLORS["orange"], alpha=0.12, label="Pulse pressure gap")
    for idx, (s_val, d_val) in enumerate(zip(sbp, dbp)):
        ax.text(idx, s_val + 2, f"{s_val:.0f}", ha="center", fontsize=10, color=COLORS["red"], weight="bold")
        ax.text(idx, d_val - 5, f"{d_val:.0f}", ha="center", fontsize=10, color=COLORS["blue"], weight="bold")
    ax.set_xticks(x, subjects)
    ax.set_ylabel("Estimated BP in mmHg")
    ax.set_title("Pretrained rPPG/V-BPE Blood Pressure Estimates", fontsize=18, weight="bold")
    ax.text(0.01, 0.94, "Dataset used: 5 local videos, 5 full-body photos, demographic CSV, pretrained CAN weights", transform=ax.transAxes, fontsize=10, color=COLORS["muted"])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_rppg_dataset_story() -> Path:
    out = DIRS["pretrained"] / "rppg_dataset_and_outputs_map.png"
    counts = {
        "Input videos": len(list((ROOT / "rppg_inference_vbpe/Input_Videos").glob("*.mp4"))),
        "Input photos": len(list((ROOT / "rppg_inference_vbpe/Input_Photos").glob("*.jpg"))),
        "Face PPG CSVs": len(list((ROOT / "rppg_inference_vbpe/Results/PPG/Face").glob("*.csv"))),
        "Hand PPG CSVs": len(list((ROOT / "rppg_inference_vbpe/Results/PPG/Hand").glob("*.csv"))),
        "Processed videos": len(list((ROOT / "rppg_inference_vbpe/Results/MP_Videos_MP4").rglob("*.mp4"))),
    }
    fig, ax = plt.subplots(figsize=(14, 5.8), dpi=160)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    ax.text(0.04, 0.9, "rPPG/V-BPE Pretrained Analysis: Data Used And Outputs", transform=ax.transAxes, fontsize=22, weight="bold", color=COLORS["ink"])
    ax.text(0.04, 0.84, "The folder uses a pretrained CAN model plus MediaPipe task models to turn short videos into BP estimates.", transform=ax.transAxes, fontsize=12, color=COLORS["muted"])
    labels = list(counts.items()) + [("SBP/DBP files", 2)]
    xs = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8]
    for idx, ((label, count), x) in enumerate(zip(labels, xs)):
        color = [COLORS["blue"], COLORS["teal"], COLORS["green"], COLORS["orange"], COLORS["lavender"], COLORS["red"]][idx]
        ax.add_patch(plt.Rectangle((x, 0.42), 0.13, 0.25, transform=ax.transAxes, facecolor=COLORS["paper"], edgecolor=COLORS["line"], linewidth=1.2))
        ax.add_patch(plt.Rectangle((x, 0.65), 0.13, 0.02, transform=ax.transAxes, facecolor=color, edgecolor=color))
        ax.text(x + 0.065, 0.57, str(count), transform=ax.transAxes, ha="center", fontsize=26, weight="bold", color=COLORS["ink"])
        ax.text(x + 0.065, 0.48, label, transform=ax.transAxes, ha="center", fontsize=10, color=COLORS["muted"], weight="bold")
        if idx < len(labels) - 1:
            ax.annotate("", xy=(x + 0.16, 0.545), xytext=(x + 0.13, 0.545), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", lw=2, color=COLORS["muted"]))
    ax.text(0.05, 0.22, "Client meaning: the existing assets already support an end-to-end demonstration from raw capture to estimated systolic and diastolic BP.", transform=ax.transAxes, fontsize=13, color=COLORS["ink"], weight="bold")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_wpin_dataset_model_story() -> list[Path]:
    out_dataset = DIRS["pretrained"] / "wpinn_dataset_story.png"
    out_models = DIRS["pretrained"] / "wpinn_business_model_comparison.png"
    model_results = pd.read_csv(ROOT / "wpin_analysis/WPINN/outputs/model_results.csv")
    inference_metrics = pd.read_csv(ROOT / "wpin_analysis/WPINN/outputs/inference_metrics.csv")

    pkl_path = ROOT / "wpin_analysis/WPINN/data/example_data.pkl"
    dataset_lines = [("Beats", "256"), ("Timesteps", "64"), ("Inputs", "Flow + time + Bio-Z"), ("Output", "BP waveform")]
    try:
        import pickle

        with open(pkl_path, "rb") as handle:
            data = pickle.load(handle)
        dataset_lines = [
            ("Beats", str(np.asarray(data["x_flow"]).shape[0])),
            ("Timesteps", str(np.asarray(data["x_flow"]).shape[1])),
            ("Inputs", "Aortic flow, time, Bio-Z"),
            ("Output", "BP waveform"),
        ]
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(13.5, 6), dpi=160)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    ax.text(0.04, 0.9, "WPINN Dataset Used For Analysis", transform=ax.transAxes, fontsize=22, weight="bold", color=COLORS["ink"])
    ax.text(0.04, 0.84, "Simulated HaeMod beat data are used to evaluate blood-pressure waveform estimation.", transform=ax.transAxes, fontsize=12, color=COLORS["muted"])
    for idx, (label, value) in enumerate(dataset_lines):
        x = 0.06 + idx * 0.23
        ax.add_patch(plt.Rectangle((x, 0.48), 0.18, 0.22, transform=ax.transAxes, facecolor=COLORS["paper"], edgecolor=COLORS["line"]))
        ax.text(x + 0.09, 0.62, value, transform=ax.transAxes, ha="center", fontsize=18 if len(value) < 14 else 12, weight="bold", color=COLORS["ink"])
        ax.text(x + 0.09, 0.53, label, transform=ax.transAxes, ha="center", fontsize=11, color=COLORS["muted"], weight="bold")
    ax.text(0.06, 0.27, "Why it matters: WPINN adds cardiovascular physics to the model so predictions are guided by how blood flow and pressure should behave.", transform=ax.transAxes, fontsize=13, color=COLORS["ink"], weight="bold")
    metrics = " | ".join(f"{row['Metric']}: {row['Value (mmHg)']:.2f} mmHg" for _, row in inference_metrics.iterrows())
    ax.text(0.06, 0.19, f"Saved inference check: {metrics}", transform=ax.transAxes, fontsize=10.5, color=COLORS["muted"])
    fig.savefig(out_dataset, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    labels = model_results["BP_Type"] + " " + model_results["Percentile"].astype(str)
    x = np.arange(len(labels))
    ax.bar(x, model_results["Improvement_pct"], color=[COLORS["green"], COLORS["green"], COLORS["teal"], COLORS["teal"]])
    for idx, value in enumerate(model_results["Improvement_pct"]):
        ax.text(idx, value + 1.2, f"{value:.1f}%", ha="center", fontsize=11, weight="bold", color=COLORS["ink"])
    ax.set_xticks(x, labels)
    ax.set_ylabel("Error reduction vs baseline Conv model")
    ax.set_title("WPINN Business Impact: Lower BP Error In Every Saved Scenario", fontsize=17, weight="bold")
    ax.text(0.01, 0.92, "Dataset used: simulated aortic flow/Bio-Z beats from HaeMod example data", transform=ax.transAxes, fontsize=10, color=COLORS["muted"])
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_models, bbox_inches="tight")
    plt.close(fig)
    return [out_dataset, out_models]


def save_pretrained_analysis_map() -> Path:
    out = DIRS["pretrained"] / "pretrained_analysis_map.png"
    fig, ax = plt.subplots(figsize=(14, 6.2), dpi=160)
    ax.set_axis_off()
    fig.patch.set_facecolor(COLORS["paper"])
    ax.text(0.04, 0.9, "Pretrained And Custom Analysis Map", transform=ax.transAxes, fontsize=24, weight="bold", color=COLORS["ink"])
    cols = [
        ("rPPG/V-BPE", "Pretrained CAN\n+ MediaPipe task models", "5 videos, 5 photos,\ndemographic CSV", "SBP/DBP estimates\n+ ROI quality research", COLORS["blue"]),
        ("Open-rPPG", "Pretrained rPPG\ninference toolbox", "Face videos processed\nby rPPG models", "HR, SQI, latency,\nand HRV metrics", COLORS["teal"]),
        ("WPINN", "Custom physics-informed\nBP model", "256 simulated beats\nwith flow, Bio-Z, BP", "Lower BP error than\nbaseline Conv model", COLORS["green"]),
    ]
    for idx, (title, model, data, output_text, color) in enumerate(cols):
        x = 0.05 + idx * 0.31
        ax.add_patch(plt.Rectangle((x, 0.12), 0.27, 0.63, transform=ax.transAxes, facecolor="white", edgecolor=COLORS["line"], linewidth=1.2))
        ax.add_patch(plt.Rectangle((x, 0.72), 0.27, 0.03, transform=ax.transAxes, facecolor=color, edgecolor=color))
        ax.text(x + 0.02, 0.64, title, transform=ax.transAxes, fontsize=18, weight="bold", color=COLORS["ink"])
        ax.text(x + 0.02, 0.54, "Model", transform=ax.transAxes, fontsize=10, weight="bold", color=color)
        ax.text(x + 0.02, 0.47, model, transform=ax.transAxes, fontsize=10.5, color=COLORS["muted"], linespacing=1.25)
        ax.text(x + 0.02, 0.39, "Dataset", transform=ax.transAxes, fontsize=10, weight="bold", color=color)
        ax.text(x + 0.02, 0.32, data, transform=ax.transAxes, fontsize=10.5, color=COLORS["muted"], linespacing=1.25)
        ax.text(x + 0.02, 0.24, "Client output", transform=ax.transAxes, fontsize=10, weight="bold", color=color)
        ax.text(x + 0.02, 0.17, output_text, transform=ax.transAxes, fontsize=10.5, color=COLORS["muted"], linespacing=1.25)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def draw_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = box
    for offset in range(4):
        draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)
    font = load_font(20, True)
    text_box = draw.textbbox((0, 0), label, font=font)
    tw = text_box[2] - text_box[0]
    th = text_box[3] - text_box[1]
    draw.rectangle((x1, y1 - th - 12, x1 + tw + 16, y1), fill=color)
    draw.text((x1 + 8, y1 - th - 8), label, fill="white", font=font)


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    max_width: int,
    line_gap: int = 6,
) -> None:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        box = draw.textbbox((0, 0), candidate, font=font)
        if box[2] - box[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = xy
    line_height = draw.textbbox((0, 0), "Ag", font=font)[3] + line_gap
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += line_height


def save_subject_roi_overlay() -> Path:
    out = DIRS["pretrained"] / "subject_roi_measurement_overlay.png"
    img = Image.open(ROOT / "rppg_inference_vbpe/Input_Photos/SL428.jpg").convert("RGB")
    img = cover_image(img, (760, 980))
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    # Approximate visual guide regions on the full-body input photo. The code's exact
    # ROI extraction happens frame-by-frame using MediaPipe landmarks.
    face_box = (int(w * 0.39), int(h * 0.08), int(w * 0.62), int(h * 0.28))
    torso_box = (int(w * 0.32), int(h * 0.26), int(w * 0.69), int(h * 0.56))
    left_hand = (int(w * 0.18), int(h * 0.48), int(w * 0.35), int(h * 0.72))
    right_hand = (int(w * 0.62), int(h * 0.48), int(w * 0.82), int(h * 0.72))
    draw_box(draw, face_box, "Face ROI", (47, 128, 237))
    draw_box(draw, left_hand, "Hand ROI", (17, 153, 158))
    draw_box(draw, right_hand, "Hand ROI", (17, 153, 158))
    draw_box(draw, torso_box, "Pose / vessel path", (242, 153, 74))
    draw.line((int(w * 0.52), int(h * 0.31), int(w * 0.72), int(h * 0.61)), fill=(242, 153, 74), width=6)
    draw.ellipse((int(w * 0.50), int(h * 0.29), int(w * 0.55), int(h * 0.34)), fill=(242, 153, 74))
    draw.ellipse((int(w * 0.69), int(h * 0.59), int(w * 0.75), int(h * 0.65)), fill=(242, 153, 74))

    canvas = Image.new("RGB", (1320, 980), "white")
    canvas.paste(img, (0, 0))
    side = ImageDraw.Draw(canvas)
    side.text((800, 70), "Where The Code Measures", fill=COLORS["ink"], font=load_font(36, True))
    notes = [
        ("Face ROI", "MediaPipe face landmarks crop facial regions for rPPG signal extraction.", COLORS["blue"]),
        ("Hand ROI", "MediaPipe hand landmarks crop hand regions as the second pulse source.", COLORS["teal"]),
        ("Pose / vessel path", "Pose landmarks estimate heart-to-hand distance used in BP calculation.", COLORS["orange"]),
        ("Signal step", "Face and hand PPG waveforms are compared to estimate pulse-transit delay.", COLORS["green"]),
    ]
    y = 165
    for title, body, color in notes:
        side.rectangle((800, y, 1240, y + 115), outline=COLORS["line"], width=2, fill=COLORS["paper"])
        side.rectangle((800, y, 812, y + 115), fill=color)
        side.text((832, y + 18), title, fill=COLORS["ink"], font=load_font(22, True))
        draw_wrapped_text(side, (832, y + 52), body, load_font(16), COLORS["muted"], 370)
        y += 140
    canvas.save(out)
    return out


def save_roi_technique_panel() -> Path:
    out = DIRS["pretrained"] / "roi_technique_panel.png"
    face = read_video_frame(ROOT / "rppg_inference_vbpe/Results/MP_Videos_MP4/Face/SL428_face_output.mp4")
    hand = read_video_frame(ROOT / "rppg_inference_vbpe/Results/MP_Videos_MP4/Hand/SL428_hand_output.mp4")
    raw = Image.open(ROOT / "rppg_inference_vbpe/Input_Photos/SL428.jpg")
    tiles = [
        (raw, "1. Original capture", "Subject image/video enters the pipeline", COLORS["blue"]),
        (face or raw, "2. Face tracking", "Landmarks define facial measurement regions", COLORS["green"]),
        (hand or raw, "3. Hand tracking", "Hand signal provides second timing source", COLORS["teal"]),
    ]
    canvas = Image.new("RGB", (1500, 620), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 34), "ROI And Tracking Techniques Used In The Code", fill=COLORS["ink"], font=load_font(34, True))
    draw.text((42, 78), "A visual translation of MediaPipe crops, pretrained CAN signal extraction, and BP timing logic.", fill=COLORS["muted"], font=load_font(18))
    for idx, (image, title, subtitle, color) in enumerate(tiles):
        tile = cover_image(image, (440, 390))
        tile_draw = ImageDraw.Draw(tile, "RGBA")
        if idx == 0:
            draw_box(tile_draw, (155, 35, 275, 150), "Face", (47, 128, 237))
            draw_box(tile_draw, (95, 230, 190, 340), "Hand", (17, 153, 158))
            draw_box(tile_draw, (265, 225, 375, 345), "Hand", (17, 153, 158))
        tile = label_tile(tile, title, subtitle, color)
        canvas.paste(tile, (40 + idx * 485, 145))
    canvas.save(out)
    return out


def save_image_backed_prediction_cards() -> list[Path]:
    out_hr = DIRS["pretrained"] / "image_backed_hr_prediction_card.png"
    out_bp = DIRS["pretrained"] / "image_backed_bp_prediction_card.png"

    raw_img = cover_image(Image.open(ROOT / "rppg_inference_vbpe/Input_Photos/SL428.jpg"), (520, 620))
    dl = pd.read_csv(ROOT / "outputs/predictions/dl_random_sample_predictions.csv")
    best_hr = dl.assign(abs_error=(dl["actual_hr"] - dl["predicted_hr"]).abs()).sort_values("abs_error").iloc[0]
    canvas = Image.new("RGB", (1200, 620), "white")
    canvas.paste(raw_img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((520, 0, 1200, 620), fill=COLORS["paper"])
    draw.text((570, 55), "Actual vs Predicted HR", fill=COLORS["ink"], font=load_font(34, True))
    draw.text((570, 105), f"Saved DL prediction sample: {best_hr['subject_id']} / {best_hr['trial_id']}", fill=COLORS["muted"], font=load_font(17))
    stats = [
        ("Actual HR", f"{best_hr['actual_hr']:.1f} bpm", COLORS["blue"]),
        ("Predicted HR", f"{best_hr['predicted_hr']:.1f} bpm", COLORS["green"]),
        ("Absolute error", f"{best_hr['abs_error']:.1f} bpm", COLORS["orange"]),
        ("BP proxy actual -> predicted", f"{best_hr['actual_bp_proxy']:.2f} -> {best_hr['predicted_bp_proxy']:.2f}", COLORS["teal"]),
    ]
    y = 170
    for label, value, color in stats:
        draw.rectangle((570, y, 1110, y + 78), fill="white", outline=COLORS["line"], width=2)
        draw.rectangle((570, y, 582, y + 78), fill=color)
        draw.text((605, y + 14), label, fill=COLORS["muted"], font=load_font(17, True))
        draw.text((850, y + 14), value, fill=COLORS["ink"], font=load_font(24, True))
        y += 92
    draw.text((570, 550), "Image shown is a project subject photo; HR values come from saved model prediction CSVs.", fill=COLORS["muted"], font=load_font(14))
    canvas.save(out_hr)

    photos = sorted((ROOT / "rppg_inference_vbpe/Input_Photos").glob("*.jpg"))
    subjects = [p.stem for p in photos]
    sbp = pd.read_csv(ROOT / "rppg_inference_vbpe/SBP_new.csv", comment="#", header=None).iloc[:, 0].astype(float).tolist()
    dbp = pd.read_csv(ROOT / "rppg_inference_vbpe/DBP_new.csv", comment="#", header=None).iloc[:, 0].astype(float).tolist()
    demo = pd.read_csv(ROOT / "rppg_inference_vbpe/Input_Data/Demographic_Data.csv").head(len(subjects))
    idx = subjects.index("SL428") if "SL428" in subjects else len(subjects) - 1
    img = cover_image(Image.open(photos[idx]), (520, 620))
    canvas = Image.new("RGB", (1200, 620), "white")
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((520, 0, 1200, 620), fill=COLORS["paper"])
    draw.text((570, 55), "Reference vs Estimated BP", fill=COLORS["ink"], font=load_font(34, True))
    draw.text((570, 105), f"V-BPE sorted subject: {subjects[idx]}", fill=COLORS["muted"], font=load_font(17))
    bp_stats = [
        ("Dataset reference SBP", f"{demo.iloc[idx]['BPS']:.0f} mmHg", COLORS["red"]),
        ("Pipeline estimated SBP", f"{sbp[idx]:.0f} mmHg", COLORS["orange"]),
        ("Dataset reference DBP", f"{demo.iloc[idx]['BPD']:.0f} mmHg", COLORS["blue"]),
        ("Pipeline estimated DBP", f"{dbp[idx]:.0f} mmHg", COLORS["teal"]),
    ]
    y = 170
    for label, value, color in bp_stats:
        draw.rectangle((570, y, 1110, y + 78), fill="white", outline=COLORS["line"], width=2)
        draw.rectangle((570, y, 582, y + 78), fill=color)
        draw.text((605, y + 14), label, fill=COLORS["muted"], font=load_font(17, True))
        draw.text((865, y + 14), value, fill=COLORS["ink"], font=load_font(22, True))
        y += 92
    draw.text((570, 550), "Reference values use the demographic CSV row paired by the pipeline's sorted file order.", fill=COLORS["muted"], font=load_font(14))
    canvas.save(out_bp)
    return [out_hr, out_bp]


def cv_load_label(value: int | float) -> str:
    labels = {0: "Low load", 1: "Moderate load", 2: "High load"}
    return labels.get(int(value), f"Class {int(value)}")


def save_custom_model_scorecard() -> Path:
    out = DIRS["custom"] / "custom_model_scorecard.png"
    ml = pd.read_csv(ROOT / "outputs/predictions/ml_random_sample_predictions.csv")
    dl = pd.read_csv(ROOT / "outputs/predictions/dl_random_sample_predictions.csv")
    wpin = pd.read_csv(ROOT / "wpin_analysis/WPINN/outputs/model_results.csv")

    ml_acc = (ml["actual_cv_load"] == ml["predicted_cv_load"]).mean()
    dl_acc = (dl["actual_cv_load_class"] == dl["predicted_cv_load_class"]).mean()
    ml_hr_mae = (ml["actual_hr"] - ml["predicted_hr"]).abs().mean()
    dl_hr_mae = (dl["actual_hr"] - dl["predicted_hr"]).abs().mean()
    ml_bp_mae = (ml["actual_bp_proxy"] - ml["predicted_bp_proxy"]).abs().mean()
    dl_bp_mae = (dl["actual_bp_proxy"] - dl["predicted_bp_proxy"]).abs().mean()
    best_wpin = wpin.loc[wpin["Improvement_pct"].idxmax()]

    cards = [
        ("ML label accuracy", f"{ml_acc:.0%}", "Cardiovascular-load class", COLORS["blue"]),
        ("DL label accuracy", f"{dl_acc:.0%}", "Cardiovascular-load class", COLORS["green"]),
        ("Best HR error", f"{min(ml_hr_mae, dl_hr_mae):.1f} bpm", "Lower is better", COLORS["orange"]),
        ("Best BP proxy error", f"{min(ml_bp_mae, dl_bp_mae):.2f}", "Lower is better", COLORS["teal"]),
        ("Best WPINN gain", f"{best_wpin['Improvement_pct']:.1f}%", f"{best_wpin['BP_Type']} {best_wpin['Percentile']}", COLORS["lavender"]),
        ("Recommended custom path", "DL + WPINN", "Labels plus BP waveform model", COLORS["red"]),
    ]
    fig, ax = plt.subplots(figsize=(14, 7), dpi=160)
    ax.set_axis_off()
    fig.patch.set_facecolor(COLORS["paper"])
    ax.text(0.04, 0.91, "Custom Model Results Scorecard", transform=ax.transAxes, fontsize=26, weight="bold", color=COLORS["ink"])
    ax.text(0.04, 0.85, "Saved custom-model outputs summarized as client-friendly accuracy, error, and improvement labels.", transform=ax.transAxes, fontsize=13, color=COLORS["muted"])
    positions = [(0.04, 0.54), (0.36, 0.54), (0.68, 0.54), (0.04, 0.22), (0.36, 0.22), (0.68, 0.22)]
    for (title, value, subtitle, color), (x, y) in zip(cards, positions):
        ax.add_patch(plt.Rectangle((x, y), 0.28, 0.22, transform=ax.transAxes, facecolor="white", edgecolor=COLORS["line"], linewidth=1.2))
        ax.add_patch(plt.Rectangle((x, y + 0.205), 0.28, 0.015, transform=ax.transAxes, facecolor=color, edgecolor=color))
        ax.text(x + 0.02, y + 0.155, title, transform=ax.transAxes, fontsize=12, color=COLORS["muted"], weight="bold")
        ax.text(x + 0.02, y + 0.085, value, transform=ax.transAxes, fontsize=22 if len(str(value)) < 12 else 16, color=COLORS["ink"], weight="bold")
        ax.text(x + 0.02, y + 0.035, subtitle, transform=ax.transAxes, fontsize=10, color=COLORS["muted"])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_custom_label_accuracy_chart() -> Path:
    out = DIRS["custom"] / "custom_label_accuracy_chart.png"
    ml = pd.read_csv(ROOT / "outputs/predictions/ml_random_sample_predictions.csv")
    dl = pd.read_csv(ROOT / "outputs/predictions/dl_random_sample_predictions.csv")
    metrics = pd.DataFrame(
        {
            "Model": ["Classic ML", "Deep Multimodal"],
            "Label accuracy": [
                (ml["actual_cv_load"] == ml["predicted_cv_load"]).mean(),
                (dl["actual_cv_load_class"] == dl["predicted_cv_load_class"]).mean(),
            ],
            "HR MAE": [
                (ml["actual_hr"] - ml["predicted_hr"]).abs().mean(),
                (dl["actual_hr"] - dl["predicted_hr"]).abs().mean(),
            ],
            "BP proxy MAE": [
                (ml["actual_bp_proxy"] - ml["predicted_bp_proxy"]).abs().mean(),
                (dl["actual_bp_proxy"] - dl["predicted_bp_proxy"]).abs().mean(),
            ],
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), dpi=160)
    specs = [
        ("Label accuracy", "Higher is better", [COLORS["blue"], COLORS["green"]], "{:.0%}"),
        ("HR MAE", "bpm, lower is better", [COLORS["orange"], COLORS["green"]], "{:.1f}"),
        ("BP proxy MAE", "lower is better", [COLORS["teal"], COLORS["green"]], "{:.2f}"),
    ]
    for ax, (metric, ylabel, colors, fmt) in zip(axes, specs):
        bars = ax.bar(metrics["Model"], metrics[metric], color=colors)
        ax.set_title(metric, fontsize=14, weight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        if metric == "Label accuracy":
            ax.set_ylim(0, 1.08)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt.format(bar.get_height()), ha="center", va="bottom", fontsize=10, weight="bold")
    fig.suptitle("Custom Model Labels And Accuracy", fontsize=19, weight="bold")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_custom_prediction_label_cards() -> list[Path]:
    out_dl = DIRS["custom"] / "custom_dl_labeled_prediction_card.png"
    out_ml = DIRS["custom"] / "custom_ml_labeled_prediction_card.png"
    out_wpin = DIRS["custom"] / "custom_wpinn_labeled_result_card.png"

    def make_card(path: Path, title: str, subtitle: str, rows: list[tuple[str, str, str]], footer: str) -> Path:
        canvas = Image.new("RGB", (1200, 760), COLORS["paper"])
        draw = ImageDraw.Draw(canvas)
        draw.text((55, 48), title, fill=COLORS["ink"], font=load_font(34, True))
        draw_wrapped_text(draw, (58, 100), subtitle, load_font(17), COLORS["muted"], 1020)
        y = 165
        for label, value, color in rows:
            draw.rectangle((70, y, 1130, y + 82), fill="white", outline=COLORS["line"], width=2)
            draw.rectangle((70, y, 84, y + 82), fill=color)
            draw.text((110, y + 17), label, fill=COLORS["muted"], font=load_font(18, True))
            draw.text((650, y + 15), value, fill=COLORS["ink"], font=load_font(24, True))
            y += 98
        draw_wrapped_text(draw, (70, 700), footer, load_font(14), COLORS["muted"], 1040)
        canvas.save(path)
        return path

    dl = pd.read_csv(ROOT / "outputs/predictions/dl_random_sample_predictions.csv")
    dl_acc = (dl["actual_cv_load_class"] == dl["predicted_cv_load_class"]).mean()
    good = dl[dl["actual_cv_load_class"] == dl["predicted_cv_load_class"]].assign(
        hr_error=(dl["actual_hr"] - dl["predicted_hr"]).abs(),
        bp_error=(dl["actual_bp_proxy"] - dl["predicted_bp_proxy"]).abs(),
    ).sort_values("hr_error").iloc[0]
    dl_rows = [
        ("Sample", f"{good['subject_id']} / {good['trial_id']} rows {int(good['start_row'])}-{int(good['end_row'])}", COLORS["blue"]),
        ("Actual label", cv_load_label(good["actual_cv_load_class"]), COLORS["orange"]),
        ("Predicted label", cv_load_label(good["predicted_cv_load_class"]), COLORS["green"]),
        ("Label confidence", f"{max(good['predicted_cv_load_proba_0'], good['predicted_cv_load_proba_1'], good['predicted_cv_load_proba_2']):.0%}", COLORS["teal"]),
        ("Overall saved label accuracy", f"{dl_acc:.0%}", COLORS["lavender"]),
    ]
    make_card(out_dl, "Custom Deep Model: Labeled Prediction", "The deep multimodal model predicts cardiovascular-load class plus HR and BP proxy values from saved window-level outputs.", dl_rows, "Source: outputs/predictions/dl_random_sample_predictions.csv")

    ml = pd.read_csv(ROOT / "outputs/predictions/ml_random_sample_predictions.csv")
    ml_acc = (ml["actual_cv_load"] == ml["predicted_cv_load"]).mean()
    ml_row = ml.assign(hr_error=(ml["actual_hr"] - ml["predicted_hr"]).abs()).sort_values("hr_error").iloc[0]
    conf_cols = ["predicted_cv_load_proba_0", "predicted_cv_load_proba_1", "predicted_cv_load_proba_2"]
    ml_rows = [
        ("Sample", f"{ml_row['subject_id']} / {ml_row['trial_id']}", COLORS["blue"]),
        ("Actual label", cv_load_label(ml_row["actual_cv_load"]), COLORS["orange"]),
        ("Predicted label", cv_load_label(ml_row["predicted_cv_load"]), COLORS["green"]),
        ("Label confidence", f"{ml_row[conf_cols].max():.0%}", COLORS["teal"]),
        ("Overall saved label accuracy", f"{ml_acc:.0%}", COLORS["lavender"]),
    ]
    make_card(out_ml, "Custom Classic ML Model: Labeled Prediction", "The classic ML model provides a compact comparison point for cardiovascular-load labels and physiology proxy predictions.", ml_rows, "Source: outputs/predictions/ml_random_sample_predictions.csv")

    wpin = pd.read_csv(ROOT / "wpin_analysis/WPINN/outputs/model_results.csv")
    best = wpin.loc[wpin["Improvement_pct"].idxmax()]
    avg_gain = wpin["Improvement_pct"].mean()
    wpin_rows = [
        ("Scenario label", f"{best['BP_Type']} at {best['Percentile']} split", COLORS["blue"]),
        ("Baseline Conv RMSE", f"{best['Conv_RMSE_mean']:.2f} mmHg", COLORS["orange"]),
        ("Custom WPINN RMSE", f"{best['WPINN_RMSE_mean']:.2f} mmHg", COLORS["green"]),
        ("Error reduction", f"{best['Improvement_pct']:.1f}%", COLORS["teal"]),
        ("Average saved improvement", f"{avg_gain:.1f}%", COLORS["lavender"]),
    ]
    make_card(out_wpin, "Custom WPINN Model: Labeled Result", "WPINN is evaluated as a regression model, so the client-facing score is error reduction rather than class accuracy.", wpin_rows, "Source: wpin_analysis/WPINN/outputs/model_results.csv")
    return [out_dl, out_ml, out_wpin]


def save_dashboard(summary: dict[str, float | str]) -> Path:
    out = DIRS["dashboard"] / "client_dashboard.png"
    paths = [
        DIRS["executive"] / "executive_scorecard.png",
        DIRS["comparisons"] / "wpinn_rmse_improvement.png",
        DIRS["dashboard"] / "roi_signal_quality.png",
        DIRS["success"] / "video_measurement_examples.png",
    ]
    thumbs = [fit_image(Image.open(path), (700, 360), "white") for path in paths if path.exists()]
    canvas = Image.new("RGB", (1500, 980), COLORS["paper"])
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 28), "Client Visual Analytics Dashboard", fill=COLORS["ink"], font=load_font(34, True))
    draw.text((42, 74), f"Prepared from {summary['n_trials']} existing trials and saved model outputs. No retraining or experiment reruns were performed.", fill=COLORS["muted"], font=load_font(18))
    for idx, thumb in enumerate(thumbs[:4]):
        x = 40 if idx % 2 == 0 else 780
        y = 125 if idx < 2 else 540
        canvas.paste(thumb, (x, y))
        draw.rectangle((x, y, x + 700, y + 360), outline=COLORS["line"], width=2)
    canvas.save(out)
    return out


def inventory() -> tuple[Path, Path]:
    rows = []
    for pattern, label in [
        ("*.csv", "CSV data"),
        ("*.json", "JSON summary"),
        ("*.png", "Chart/image"),
        ("*.jpg", "Photo"),
        ("*.jpeg", "Photo"),
        ("*.mp4", "Video"),
        ("*.avi", "Video"),
        ("*.MOV", "Video"),
        ("*.md", "Documentation"),
        ("*.docx", "Document report"),
    ]:
        for path in ROOT.rglob(pattern):
            if ".venv" in path.parts or "__pycache__" in path.parts:
                continue
            rows.append({"type": label, "path": rel(path), "size_bytes": path.stat().st_size})
    df = pd.DataFrame(rows).sort_values(["type", "path"])
    csv_path = DIRS["presentation"] / "asset_inventory.csv"
    md_path = DIRS["presentation"] / "ASSET_INVENTORY.md"
    df.to_csv(csv_path, index=False)
    counts = df.groupby("type").size().sort_index()
    missing = [
        "No business-facing README section embedded the most important visuals before this update.",
        "Existing charts were technical and scattered across model-specific folders.",
        "Video and photo examples existed, but they were not arranged as client-facing measurement stories.",
        "Failure/challenge explanations existed as data signals, but not as plain-language visuals.",
    ]
    lines = ["# Asset Inventory", "", "## Summary", ""]
    lines += [f"- {kind}: {count}" for kind, count in counts.items()]
    lines += ["", "## Missing Visual Explanations", ""]
    lines += [f"- {item}" for item in missing]
    lines += ["", "## Full Inventory", "", "| Type | Path | Size |", "|---|---|---:|"]
    lines += [f"| {row.type} | `{row.path}` | {row.size_bytes} |" for row in df.itertuples()]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def write_changelog(created: list[Path], modified: list[str]) -> Path:
    out = REPORTS / "VISUALIZATION_CHANGELOG.md"
    lines = [
        "# Visualization Changelog",
        "",
        "## Files Created",
        "",
    ]
    lines += [f"- `{rel(path)}`" for path in created]
    lines += ["", "## Files Modified", ""]
    lines += [f"- `{path}`" for path in modified]
    lines += [
        "",
        "## Visualizations Added",
        "",
        "| Visualization | Purpose | Data source | Output |",
        "|---|---|---|---|",
        f"| Executive scorecard | Summarize the project in client-facing KPIs | `outputs/figures/results.json`, prediction CSVs, WPINN CSVs | `{rel(DIRS['executive'] / 'executive_scorecard.png')}` |",
        f"| Model comparison overview | Compare held-out HR and BP proxy error | `outputs/predictions/*.csv` | `{rel(DIRS['comparisons'] / 'model_comparison_overview.png')}` |",
        f"| WPINN improvement chart | Show custom BP-model value versus baseline | `wpin_analysis/WPINN/outputs/model_results.csv` | `{rel(DIRS['comparisons'] / 'wpinn_rmse_improvement.png')}` |",
        f"| ROI quality chart | Explain which face regions provide clearer measurements | `rppg_inference_vbpe/research_output/roi_summary_table.csv` | `{rel(DIRS['dashboard'] / 'roi_signal_quality.png')}` |",
        f"| Visual measurement examples | Show actual input and tracking outputs | Photos and processed videos in `rppg_inference_vbpe` | `{rel(DIRS['success'] / 'video_measurement_examples.png')}` |",
        f"| Challenge gallery | Translate weaker signal cases into plain-language risks | `per_roi_sqi_all.csv` and video frames | `{rel(DIRS['failure'] / 'client_friendly_challenge_gallery.png')}` |",
        f"| Operational workflow | Explain the end-to-end process | Existing repository workflow | `{rel(DIRS['workflow'] / 'operational_workflow.png')}` |",
        f"| Client dashboard | One-page visual summary | Generated report assets | `{rel(DIRS['dashboard'] / 'client_dashboard.png')}` |",
        f"| rPPG BP summary | Explain pretrained V-BPE blood-pressure estimates | `rppg_inference_vbpe/SBP_new.csv`, `DBP_new.csv` | `{rel(DIRS['pretrained'] / 'rppg_vbpe_bp_summary.png')}` |",
        f"| rPPG dataset/output map | Show which local assets the pretrained rPPG pipeline uses | `Input_Videos`, `Input_Photos`, `Results/PPG` | `{rel(DIRS['pretrained'] / 'rppg_dataset_and_outputs_map.png')}` |",
        f"| WPINN dataset story | Explain the simulated beat dataset in plain language | `wpin_analysis/WPINN/data/example_data.pkl` | `{rel(DIRS['pretrained'] / 'wpinn_dataset_story.png')}` |",
        f"| WPINN business comparison | Show error reduction from physics-informed modeling | `wpin_analysis/WPINN/outputs/model_results.csv` | `{rel(DIRS['pretrained'] / 'wpinn_business_model_comparison.png')}` |",
        f"| Pretrained analysis map | Compare rPPG, Open-rPPG, and WPINN analysis roles | Project reports and result folders | `{rel(DIRS['pretrained'] / 'pretrained_analysis_map.png')}` |",
        f"| Subject ROI overlay | Mark face, hand, and vessel-path regions on a subject photo | `Input_Photos/SL428.jpg`, rPPG code workflow | `{rel(DIRS['pretrained'] / 'subject_roi_measurement_overlay.png')}` |",
        f"| ROI technique panel | Show original capture, face tracking, and hand tracking examples | Input photo and processed MediaPipe videos | `{rel(DIRS['pretrained'] / 'roi_technique_panel.png')}` |",
        f"| Image-backed HR card | Label actual vs predicted HR beside a project subject image | `outputs/predictions/dl_random_sample_predictions.csv` | `{rel(DIRS['pretrained'] / 'image_backed_hr_prediction_card.png')}` |",
        f"| Image-backed BP card | Label reference vs estimated BP beside the paired subject photo | `Demographic_Data.csv`, `SBP_new.csv`, `DBP_new.csv` | `{rel(DIRS['pretrained'] / 'image_backed_bp_prediction_card.png')}` |",
        f"| Custom model scorecard | Summarize custom ML, DL, and WPINN performance labels | Saved prediction CSVs and WPINN results | `{rel(DIRS['custom'] / 'custom_model_scorecard.png')}` |",
        f"| Custom label accuracy chart | Compare custom model label accuracy and regression errors | `outputs/predictions/*.csv` | `{rel(DIRS['custom'] / 'custom_label_accuracy_chart.png')}` |",
        f"| Custom DL labeled card | Show actual label, predicted label, confidence, and accuracy | `dl_random_sample_predictions.csv` | `{rel(DIRS['custom'] / 'custom_dl_labeled_prediction_card.png')}` |",
        f"| Custom ML labeled card | Show actual label, predicted label, confidence, and accuracy | `ml_random_sample_predictions.csv` | `{rel(DIRS['custom'] / 'custom_ml_labeled_prediction_card.png')}` |",
        f"| Custom WPINN labeled card | Show scenario label, RMSE, and error reduction | `wpin_analysis/WPINN/outputs/model_results.csv` | `{rel(DIRS['custom'] / 'custom_wpinn_labeled_result_card.png')}` |",
        "",
        "## Pending Enhancements",
        "",
        "- Add a short slide deck using the generated assets.",
        "- Add more frame-level examples after additional model-output screenshots are available.",
        "- Add confidence-band visuals if future experiments save uncertainty estimates.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_key_findings(summary: dict[str, float | str]) -> Path:
    out = DIRS["executive"] / "KEY_FINDINGS.md"
    lines = [
        "# Client-Focused Key Findings",
        "",
        f"- The dataset includes {summary['n_trials']} analyzed trials with signal quality reported as {summary['quality']}.",
        f"- The deep multimodal model produced a held-out heart-rate error of about {summary['dl_hr_mae']:.1f} bpm on the saved sample predictions.",
        f"- The custom WPINN blood-pressure model delivered its largest saved improvement on {summary['best_wpin_label']}, reducing RMSE by {summary['best_wpin_improvement']:.1f}% versus the baseline.",
        f"- Face/hand region quality checks reached {summary['real_roi_acc']:.0%} accuracy in the saved real-data quality classifier summary.",
        "- Recommended client story: combine ROI quality screening, pulse morphology checks, and the custom BP model for the most reliable reporting workflow.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    ensure_dirs()
    summary = metric_summary()
    created: list[Path] = []
    created.append(save_scorecard(summary))
    created.extend(save_model_comparison(summary))
    created.append(save_roi_quality())
    created.extend(save_visual_examples())
    created.append(save_workflow())
    created.append(save_rppg_bp_results())
    created.append(save_rppg_dataset_story())
    created.extend(save_wpin_dataset_model_story())
    created.append(save_pretrained_analysis_map())
    created.append(save_subject_roi_overlay())
    created.append(save_roi_technique_panel())
    created.extend(save_image_backed_prediction_cards())
    created.append(save_custom_model_scorecard())
    created.append(save_custom_label_accuracy_chart())
    created.extend(save_custom_prediction_label_cards())
    created.append(save_dashboard(summary))
    created.extend(inventory())
    created.append(write_key_findings(summary))
    created.append(write_changelog(created, ["README.md", "generate_client_visuals.py"]))
    print("Generated client visual analytics assets:")
    for path in created:
        print(f"- {rel(path)}")


if __name__ == "__main__":
    main()
