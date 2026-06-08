from __future__ import annotations

import torch
from torch import nn


class MultitaskPhysiologyLoss(nn.Module):
    def __init__(self, hr_weight: float = 1.0, load_weight: float = 0.5, bp_weight: float = 0.5):
        super().__init__()
        self.hr_weight = hr_weight
        self.load_weight = load_weight
        self.bp_weight = bp_weight
        self.regression = nn.SmoothL1Loss()
        self.classification = nn.CrossEntropyLoss()

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        hr_loss = self.regression(outputs["hr"], batch["hr_target"])
        load_loss = self.classification(outputs["cv_load_logits"], batch["cv_load_class"])
        bp_loss = self.regression(outputs["bp_proxy"], batch["bp_proxy_target"])
        total = self.hr_weight * hr_loss + self.load_weight * load_loss + self.bp_weight * bp_loss
        metrics = {
            "loss": float(total.detach().cpu()),
            "hr_loss": float(hr_loss.detach().cpu()),
            "load_loss": float(load_loss.detach().cpu()),
            "bp_loss": float(bp_loss.detach().cpu()),
        }
        return total, metrics
