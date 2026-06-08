from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DLModelConfig:
    video_channels: int = 3
    bvp_channels: int = 1
    physiology_channels: int = 6
    static_dim: int = 0
    hidden_dim: int = 128
    temporal_dim: int = 128
    n_load_classes: int = 3
    dropout: float = 0.15


class VideoROIEncoder(nn.Module):
    """3D CNN branch for ROI frame clips shaped as N,C,T,H,W."""

    def __init__(self, in_channels: int = 3, out_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((8, 1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Conv1d(64, out_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        x = self.net(video).squeeze(-1).squeeze(-1)
        return self.proj(x)


class BVPEncoder(nn.Module):
    """1D CNN branch for pulse/BVP sequences shaped as N,C,T."""

    def __init__(self, in_channels: int = 1, out_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, out_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, bvp: torch.Tensor) -> torch.Tensor:
        return self.net(bvp)


class PhysiologyEncoder(nn.Module):
    """Temporal CNN branch for EDA, TEMP, ACC, HRV/context sequences."""

    def __init__(self, in_channels: int, out_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, physiology: torch.Tensor) -> torch.Tensor:
        return self.net(physiology)


class StaticFeatureEncoder(nn.Module):
    """MLP for engineered trial/window-level features."""

    def __init__(self, static_dim: int, out_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        if static_dim <= 0:
            self.net = None
            self.out_dim = 0
        else:
            self.net = nn.Sequential(
                nn.Linear(static_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim, out_dim),
                nn.ReLU(),
            )
            self.out_dim = out_dim

    def forward(self, static: torch.Tensor) -> torch.Tensor | None:
        if self.net is None:
            return None
        return self.net(static)


class TemporalFusionCNN(nn.Module):
    """Fast temporal CNN fusion block used after branch concatenation."""

    def __init__(self, in_channels: int, hidden_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding=2, dilation=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=8, dilation=4),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return self.pool(x).squeeze(-1)


class MultimodalPhysiologyNet(nn.Module):
    """Video + BVP + physiology + engineered-feature fusion model."""

    def __init__(self, config: DLModelConfig):
        super().__init__()
        self.config = config
        self.video_encoder = VideoROIEncoder(config.video_channels, config.hidden_dim, config.dropout)
        self.bvp_encoder = BVPEncoder(config.bvp_channels, config.hidden_dim, config.dropout)
        self.physiology_encoder = PhysiologyEncoder(config.physiology_channels, config.hidden_dim, config.dropout)
        self.static_encoder = StaticFeatureEncoder(config.static_dim, config.hidden_dim, config.dropout)

        static_channels = config.hidden_dim if config.static_dim > 0 else 0
        fusion_channels = config.hidden_dim * 3 + static_channels
        self.fusion = TemporalFusionCNN(fusion_channels, config.temporal_dim, config.dropout)

        self.hr_head = nn.Sequential(nn.Linear(config.temporal_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.load_head = nn.Sequential(nn.Linear(config.temporal_dim, 64), nn.ReLU(), nn.Linear(64, config.n_load_classes))
        self.bp_proxy_head = nn.Sequential(nn.Linear(config.temporal_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, video: torch.Tensor, bvp: torch.Tensor, physiology: torch.Tensor, static: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        video_features = self.video_encoder(video)
        bvp_features = self.bvp_encoder(bvp)
        physiology_features = self.physiology_encoder(physiology)

        min_t = min(video_features.shape[-1], bvp_features.shape[-1], physiology_features.shape[-1])
        parts = [video_features[..., :min_t], bvp_features[..., :min_t], physiology_features[..., :min_t]]

        if static is not None and self.static_encoder.out_dim > 0:
            static_features = self.static_encoder(static).unsqueeze(-1).expand(-1, -1, min_t)
            parts.append(static_features)

        fused = self.fusion(torch.cat(parts, dim=1))
        return {
            "hr": self.hr_head(fused).squeeze(-1),
            "cv_load_logits": self.load_head(fused),
            "bp_proxy": self.bp_proxy_head(fused).squeeze(-1),
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
