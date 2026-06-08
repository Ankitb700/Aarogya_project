from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dl_losses import MultitaskPhysiologyLoss


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def train_one_epoch(model, loader: DataLoader, optimizer, loss_fn: MultitaskPhysiologyLoss, device: torch.device) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["video"], batch["bvp"], batch["physiology"], batch["static"])
        loss, metrics = loss_fn(outputs, batch)
        loss.backward()
        optimizer.step()
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate(model, loader: DataLoader, loss_fn: MultitaskPhysiologyLoss, device: torch.device) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    correct = 0
    n_items = 0
    for batch in loader:
        batch = to_device(batch, device)
        outputs = model(batch["video"], batch["bvp"], batch["physiology"], batch["static"])
        _, metrics = loss_fn(outputs, batch)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        pred = outputs["cv_load_logits"].argmax(dim=1)
        correct += int((pred == batch["cv_load_class"]).sum().detach().cpu())
        n_items += int(pred.numel())
        count += 1
    out = {key: value / max(count, 1) for key, value in totals.items()}
    out["load_accuracy"] = correct / max(n_items, 1)
    return out


def fit_multitask_model(model, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 5, lr: float = 1e-3, device: str | None = None) -> list[dict[str, float]]:
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = MultitaskPhysiologyLoss()
    history = []
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device_obj)
        val_metrics = evaluate(model, val_loader, loss_fn, device_obj)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
    return history


def save_checkpoint(
    model,
    path: str | Path,
    config: dict | None = None,
    history: list[dict] | None = None,
    static_scaler: dict | None = None,
    target_scalers: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config or {},
            "history": history or [],
            "static_scaler": static_scaler,
            "target_scalers": target_scalers or {},
        },
        path,
    )
    return path
