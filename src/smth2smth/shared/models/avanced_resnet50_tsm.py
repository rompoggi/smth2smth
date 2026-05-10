"""ResNet-50 with Temporal Shift Module (TSM) and consensus head.

Adds two opt-in Phase-2 knobs (default off ⇒ legacy behavior):
* ``model.drop_path_rate`` -- linear-scaled Stochastic Depth on the backbone's
  bottleneck residual branches (Huang et al. 2016).
* ``model.head`` -- temporal aggregation: ``"mean"`` (default, plain consensus)
  or ``"attn"`` (a single learnable query × frame keys attention pool).
"""

from __future__ import annotations

from types import MethodType

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
            raise ValueError(f"Input first dim {nt} not divisible by n_segment={self.n_segment}.")
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


class DropPath(nn.Module):
    """Per-sample Stochastic Depth (Huang et al. 2016).

    With probability ``drop_prob`` the residual branch contribution is zeroed
    out for that sample; the remaining contributions are rescaled by
    ``1 / (1 - drop_prob)`` so the expectation matches the no-drop case.
    Identity at eval time. Setting ``drop_prob=0`` is a no-op (no params).
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}.")
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Per-sample mask broadcast over (C, H, W) (or any tail dims).
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob


def _patched_bottleneck_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for torchvision Bottleneck.forward that calls
    ``self.drop_path`` on the residual branch before the residual addition.
    """
    identity = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)
    out = self.conv3(out)
    out = self.bn3(out)
    if self.downsample is not None:
        identity = self.downsample(x)
    out = self.drop_path(out)
    out = out + identity
    out = self.relu(out)
    return out


def _inject_drop_path(backbone: nn.Module, drop_path_rate: float) -> None:
    """Attach a :class:`DropPath` to every bottleneck block on a linear schedule
    (deeper blocks drop more), and rebind their ``forward`` to the patched
    variant. ``drop_path_rate=0.0`` makes every block use ``DropPath(0)`` which
    short-circuits and behaves identically to the original ResNet."""
    blocks: list[nn.Module] = []
    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        for block in getattr(backbone, layer_name):
            blocks.append(block)
    n = max(1, len(blocks))
    for i, block in enumerate(blocks):
        prob = float(drop_path_rate) * float(i) / float(n - 1) if n > 1 else 0.0
        block.drop_path = DropPath(drop_prob=prob)
        block.forward = MethodType(_patched_bottleneck_forward, block)


class AttentionPool(nn.Module):
    """1-query attention pooling over T frame features.

    Equivalent to a single multi-head attention with one learnable query token
    and the frame features as keys/values. Designed for very small T (e.g. 4):
    cheap (~``feature_dim`` extra params for the query plus a single qkv
    projection block) and a strict superset of the mean-consensus baseline
    (uniform attention weights ⇒ identical to mean).
    """

    def __init__(self, feature_dim: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError(
                f"feature_dim={feature_dim} must be divisible by num_heads={num_heads}."
            )
        self.query = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(feature_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, sequence_features: torch.Tensor) -> torch.Tensor:
        """``sequence_features``: ``(B, T, D)`` -> ``(B, D)`` pooled feature."""
        b = sequence_features.size(0)
        q = self.query.expand(b, -1, -1)  # (B, 1, D)
        pooled, _ = self.attn(q, sequence_features, sequence_features, need_weights=False)
        pooled = pooled.squeeze(1)  # (B, D)
        return self.norm(pooled)


class AvancedResNet50TSM(nn.Module):
    """ResNet-50 + TSM + temporal aggregation classifier.

    Backwards-compatible: with ``drop_path_rate=0`` and ``head="mean"`` the
    architecture is bit-identical to the original (no extra modules in the
    state_dict for old checkpoints).
    """

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
    ) -> None:
        super().__init__()
        if head not in {"mean", "attn"}:
            raise ValueError(f"head must be 'mean' or 'attn', got {head!r}.")

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
        if drop_path_rate > 0.0:
            _inject_drop_path(self.backbone, drop_path_rate=drop_path_rate)

        self.head_kind = head
        self.attn_pool: AttentionPool | None = (
            AttentionPool(feature_dim=feature_dim, num_heads=head_num_heads)
            if head == "attn"
            else None
        )

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        """ResNet-style init: Kaiming He fan-out for ReLU convs, BN=1/0,
        small-Gaussian for the final classifier. Designed for from-scratch
        training (no ImageNet weights) and matches the original ResNet/TSM
        recipe (He et al. 2015; Lin et al. 2019)."""
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
        # Small-Gaussian classifier head (TSM and ResNet defaults).
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = video_batch.shape
        frames = video_batch.reshape(batch_size * num_frames, channels, height, width)
        frame_features = self.backbone(frames)
        frame_features = torch.flatten(frame_features, start_dim=1)
        sequence_features = frame_features.view(batch_size, num_frames, -1)
        if self.attn_pool is not None:
            consensus_features = self.attn_pool(sequence_features)
        else:
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
        drop_path_rate=float(cfg.model.get("drop_path_rate", 0.0)),
        head=str(cfg.model.get("head", "mean")),
        head_num_heads=int(cfg.model.get("head_num_heads", 4)),
    )
