"""ResNet-50 with Temporal Shift Module (TSM) and consensus head."""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchvision import models

from smth2smth.shared.models.registry import register_model


class TemporalShift(nn.Module):
    """Shift channels along the temporal axis with zero parameters."""

    def __init__(self, n_segment: int, fold_div: int = 8) -> None:
        super().__init__()
        if n_segment <= 1:
            raise ValueError(f"n_segment must be > 1 for temporal shift, got {n_segment}.")
        if fold_div <= 0:
            raise ValueError(f"fold_div must be positive, got {fold_div}.")
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nt, c, h, w = x.size()
        if nt % self.n_segment != 0:
            raise ValueError(
                f"Input first dim {nt} not divisible by n_segment={self.n_segment}."
            )
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        if fold == 0:
            return x.view(nt, c, h, w)

        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold : 2 * fold] = x[:, :-1, fold : 2 * fold]
        out[:, :, 2 * fold :] = x[:, :, 2 * fold :]
        return out.view(nt, c, h, w)


def _make_temporal_shift_resnet(
    backbone: nn.Module,
    n_segment: int,
    shift_div: int,
    shift_place: str = "blockres",
) -> nn.Module:
    if shift_place != "blockres":
        raise ValueError(f"Unsupported shift_place={shift_place!r}. Use 'blockres'.")

    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        layer = getattr(backbone, layer_name)
        for block in layer:
            block.conv1 = nn.Sequential(
                TemporalShift(n_segment=n_segment, fold_div=shift_div),
                block.conv1,
            )
    return backbone


def _validate_residual_paths(backbone: nn.Module) -> None:
    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        layer = getattr(backbone, layer_name)
        for block in layer:
            if not hasattr(block, "downsample"):
                raise ValueError("Backbone block has no residual path metadata ('downsample').")
            if not hasattr(block, "bn3"):
                raise ValueError("Backbone block is not a bottleneck-style residual block.")


class AvancedResNet50TSM(nn.Module):
    """ResNet-50 + TSM + temporal consensus average classifier."""

    def __init__(
        self,
        num_classes: int,
        num_frames: int,
        shift_div: int = 8,
        shift_place: str = "blockres",
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        backbone = models.resnet50(weights=None)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = _make_temporal_shift_resnet(
            backbone=backbone,
            n_segment=num_frames,
            shift_div=shift_div,
            shift_place=shift_place,
        )
        _validate_residual_paths(self.backbone)

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self._init_weights_xavier()

    def _init_weights_xavier(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = video_batch.shape
        frames = video_batch.reshape(batch_size * num_frames, channels, height, width)
        frame_features = self.backbone(frames)
        frame_features = torch.flatten(frame_features, start_dim=1)
        sequence_features = frame_features.view(batch_size, num_frames, -1)
        consensus_features = sequence_features.mean(dim=1)
        return self.classifier(self.dropout(consensus_features))


@register_model("avanced_resnet50_tsm")
def build_avanced_resnet50_tsm(cfg: DictConfig) -> nn.Module:
    return AvancedResNet50TSM(
        num_classes=int(cfg.model.num_classes),
        num_frames=int(cfg.dataset.num_frames),
        shift_div=int(cfg.model.get("shift_div", 8)),
        shift_place=str(cfg.model.get("shift_place", "blockres")),
        dropout=float(cfg.model.get("dropout", 0.5)),
    )
