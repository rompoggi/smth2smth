"""Dual-stream TSM: RGB (ResNet-50) + consecutive-frame differences (ResNet-34).

Fuses per-time-step RGB embeddings with motion embeddings (``T`` tokens:
frame~0 uses zero-padded motion; frames~``1..T-1`` concatenate the difference
``I_t - I_{t-1}`` embedding). A shared temporal head (mean or 1-query attention)
and linear classifier follow :class:`AvancedResNet50TSM`.
"""

from __future__ import annotations

from types import MethodType

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchvision import models

from smth2smth.shared.models.avanced_resnet50_tsm import (
    AttentionPool,
    DropPath,
    _inject_drop_path,
    _make_temporal_shift_resnet,
    _validate_residual_paths,
)
from smth2smth.shared.models.registry import register_model


def _bottleneck_resnet34_blocks(backbone: nn.Module) -> list[nn.Module]:
    blocks: list[nn.Module] = []
    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        for block in getattr(backbone, layer_name):
            blocks.append(block)
    return blocks


def _inject_drop_path_basic_block(backbone: nn.Module, drop_path_rate: float) -> None:
    """Stochastic depth for ResNet-34 *BasicBlock* branches (two conv layers)."""
    blocks = _bottleneck_resnet34_blocks(backbone)
    n = max(1, len(blocks))
    for i, block in enumerate(blocks):
        prob = float(drop_path_rate) * float(i) / float(n - 1) if n > 1 else 0.0
        block.drop_path = DropPath(drop_prob=prob)

        def _basic_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out = self.drop_path(out)
            out = out + identity
            out = self.relu(out)
            return out

        block.forward = MethodType(_basic_forward, block)


class DualStreamRgbDiffTSM(nn.Module):
    """RGB ResNet-50 + TSM and difference ResNet-34 + TSM with fused temporal head."""

    def __init__(
        self,
        num_classes: int,
        num_frames: int,
        shift_div: int = 8,
        shift_place: str = "blockres",
        dropout: float = 0.5,
        drop_path_rate: float = 0.0,
        head: str = "mean",
        head_num_heads: int = 4,
        fuse_dim: int = 512,
    ) -> None:
        super().__init__()
        if num_frames < 2:
            raise ValueError(f"DualStreamRgbDiffTSM needs num_frames >= 2, got {num_frames}.")
        if head not in {"mean", "attn"}:
            raise ValueError(f"head must be 'mean' or 'attn', got {head!r}.")
        if fuse_dim % head_num_heads != 0 and head == "attn":
            raise ValueError(
                f"fuse_dim={fuse_dim} must be divisible by head_num_heads={head_num_heads} for attn head."
            )

        self.num_frames = int(num_frames)
        self.num_diff = self.num_frames - 1

        rgb = models.resnet50(weights=None)
        self.rgb_dim = rgb.fc.in_features
        rgb.fc = nn.Identity()
        self.rgb_backbone = _make_temporal_shift_resnet(
            backbone=rgb,
            n_segment=self.num_frames,
            shift_div=shift_div,
            shift_place=shift_place,
        )
        _validate_residual_paths(self.rgb_backbone)
        if drop_path_rate > 0.0:
            _inject_drop_path(self.rgb_backbone, drop_path_rate=drop_path_rate)

        motion = models.resnet34(weights=None)
        self.motion_dim = motion.fc.in_features
        motion.fc = nn.Identity()
        self.motion_backbone = _make_temporal_shift_resnet(
            backbone=motion,
            n_segment=self.num_diff,
            shift_div=shift_div,
            shift_place=shift_place,
        )
        if drop_path_rate > 0.0:
            _inject_drop_path_basic_block(self.motion_backbone, drop_path_rate=drop_path_rate)

        fuse_in = self.rgb_dim + self.motion_dim
        self.fuse = nn.Sequential(
            nn.LayerNorm(fuse_in),
            nn.Linear(fuse_in, fuse_dim),
            nn.ReLU(inplace=True),
        )
        self.fuse_dim = int(fuse_dim)

        self.head_kind = head
        self.attn_pool: AttentionPool | None = (
            AttentionPool(feature_dim=self.fuse_dim, num_heads=head_num_heads)
            if head == "attn"
            else None
        )
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.fuse_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """``video_batch``: ``(B, T, C, H, W)`` -> logits ``(B, num_classes)``."""
        b, t, c, h, w = video_batch.shape
        if t != self.num_frames:
            raise ValueError(f"Expected T={self.num_frames}, got {t}.")

        rgb_flat = video_batch.reshape(b * t, c, h, w)
        rgb_feat = self.rgb_backbone(rgb_flat)
        rgb_feat = torch.flatten(rgb_feat, start_dim=1).view(b, t, self.rgb_dim)

        diffs = video_batch[:, 1:, ...] - video_batch[:, :-1, ...]
        diffs_flat = diffs.reshape(b * (t - 1), c, h, w)
        motion_feat = self.motion_backbone(diffs_flat)
        motion_feat = torch.flatten(motion_feat, start_dim=1).view(b, t - 1, self.motion_dim)

        motion_pad = rgb_feat.new_zeros(b, self.motion_dim)
        fused_list: list[torch.Tensor] = []
        fused_list.append(self.fuse(torch.cat([rgb_feat[:, 0, :], motion_pad], dim=-1)))
        for i in range(1, t):
            fused_list.append(
                self.fuse(torch.cat([rgb_feat[:, i, :], motion_feat[:, i - 1, :]], dim=-1))
            )
        sequence = torch.stack(fused_list, dim=1)

        if self.attn_pool is not None:
            consensus = self.attn_pool(sequence)
        else:
            consensus = sequence.mean(dim=1)
        return self.classifier(self.dropout(consensus))


@register_model("dual_stream_rgb_diff_tsm")
def build_dual_stream_rgb_diff_tsm(cfg: DictConfig) -> nn.Module:
    return DualStreamRgbDiffTSM(
        num_classes=int(cfg.model.num_classes),
        num_frames=int(cfg.dataset.num_frames),
        shift_div=int(cfg.model.get("shift_div", 8)),
        shift_place=str(cfg.model.get("shift_place", "blockres")),
        dropout=float(cfg.model.get("dropout", 0.5)),
        drop_path_rate=float(cfg.model.get("drop_path_rate", 0.0)),
        head=str(cfg.model.get("head", "mean")),
        head_num_heads=int(cfg.model.get("head_num_heads", 4)),
        fuse_dim=int(cfg.model.get("fuse_dim", 512)),
    )
