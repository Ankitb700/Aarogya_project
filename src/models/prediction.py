from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .datasets import AlignedWindowDataset, inverse_standard_scaler
from .dl_models import DLModelConfig, MultimodalPhysiologyNet


def split_80_20(df: pd.DataFrame, random_state: int = 42, stratify_col: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = None
    if stratify_col and stratify_col in df and df[stratify_col].nunique() > 1:
        counts = df[stratify_col].value_counts()
        if counts.min() >= 2:
            stratify = df[stratify_col]
    train, test = train_test_split(df, test_size=0.2, random_state=random_state, shuffle=True, stratify=stratify)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def random_sample(df: pd.DataFrame, n: int = 5, random_state: int = 42) -> pd.DataFrame:
    if len(df) <= n:
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=random_state).reset_index(drop=True)


def load_ml_model(path: str | Path):
    return joblib.load(path)


def predict_ml_samples(samples: pd.DataFrame, model_paths: dict[str, str | Path]) -> pd.DataFrame:
    result = samples[["subject_id", "trial_id"]].copy()
    target_columns = {
        "hr": "hr_mean",
        "cv_load": "cv_load_class",
        "bp_proxy": "bp_proxy_score",
    }
    for prefix, target_col in target_columns.items():
        if target_col in samples:
            result[f"actual_{prefix}"] = samples[target_col].to_numpy()

    for prefix, path in model_paths.items():
        model = load_ml_model(path)
        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is None and hasattr(model, "named_steps"):
            feature_names = getattr(model.named_steps.get("imputer"), "feature_names_in_", None)
        if feature_names is None:
            numeric = samples.select_dtypes(include=[np.number]).columns
            feature_names = [c for c in numeric if not c.startswith("actual_")]
        X = samples.reindex(columns=list(feature_names), fill_value=np.nan)
        pred = model.predict(X)
        result[f"predicted_{prefix}"] = pred
        if hasattr(model, "predict_proba") and prefix == "cv_load":
            proba = model.predict_proba(X)
            for class_idx in range(proba.shape[1]):
                result[f"predicted_cv_load_proba_{class_idx}"] = proba[:, class_idx]
    return result


def load_dl_checkpoint(path: str | Path, map_location: str = "cpu") -> tuple[MultimodalPhysiologyNet, dict]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    config_data = checkpoint.get("config", {})
    config = DLModelConfig(**config_data)
    model = MultimodalPhysiologyNet(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict_dl_samples(
    windows: pd.DataFrame,
    feature_table: pd.DataFrame,
    checkpoint_path: str | Path,
    sequence_length: int = 64,
    frame_size: int = 32,
    video_frames: int = 8,
    use_video: bool = False,
    batch_size: int = 2,
    device: str = "cpu",
) -> pd.DataFrame:
    model, checkpoint = load_dl_checkpoint(checkpoint_path, map_location=device)
    model.to(device)
    target_scalers = checkpoint.get("target_scalers", {})
    dataset = AlignedWindowDataset(
        windows,
        feature_table=feature_table,
        sequence_length=sequence_length,
        frame_size=frame_size,
        video_frames=video_frames,
        use_video=use_video,
        static_scaler=checkpoint.get("static_scaler"),
        target_scalers=target_scalers,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    rows = []
    offset = 0
    for batch in loader:
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        outputs = model(batch["video"], batch["bvp"], batch["physiology"], batch["static"])
        predicted_hr = inverse_standard_scaler(outputs["hr"].detach().cpu().numpy(), target_scalers.get("hr")).reshape(-1)
        predicted_bp = inverse_standard_scaler(outputs["bp_proxy"].detach().cpu().numpy(), target_scalers.get("bp_proxy")).reshape(-1)
        load_probs = torch.softmax(outputs["cv_load_logits"], dim=1)
        load_pred = load_probs.argmax(dim=1)
        batch_size_actual = outputs["hr"].shape[0]
        meta = windows.iloc[offset: offset + batch_size_actual].reset_index(drop=True)
        offset += batch_size_actual
        for i in range(batch_size_actual):
            row = {
                "subject_id": meta.loc[i, "subject_id"],
                "trial_id": meta.loc[i, "trial_id"],
                "start_row": int(meta.loc[i, "start_row"]),
                "end_row": int(meta.loc[i, "end_row"]),
                "actual_hr": float(batch["hr_target_raw"][i].detach().cpu()),
                "predicted_hr": float(predicted_hr[i]),
                "predicted_hr_model_scale": float(outputs["hr"][i].detach().cpu()),
                "actual_cv_load_class": int(batch["cv_load_class"][i].detach().cpu()),
                "predicted_cv_load_class": int(load_pred[i].detach().cpu()),
                "actual_bp_proxy": float(batch["bp_proxy_target_raw"][i].detach().cpu()),
                "predicted_bp_proxy": float(predicted_bp[i]),
                "predicted_bp_proxy_model_scale": float(outputs["bp_proxy"][i].detach().cpu()),
            }
            for class_idx in range(load_probs.shape[1]):
                row[f"predicted_cv_load_proba_{class_idx}"] = float(load_probs[i, class_idx].detach().cpu())
            rows.append(row)
    return pd.DataFrame(rows)
