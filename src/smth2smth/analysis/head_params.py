"""Trainable head-parameter counts for diverse VideoMAE classifier heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from smth2smth.pipelines.train import _is_new_module_param
from smth2smth.shared.models.video_mae import VideoMAEViT

NUM_CLASSES = 33
EMBED_DIM = 768
DEPTH = 12
NUM_HEADS = 12
HEAD_MLP_RATIO = 4.0
HEAD_NUM_HEADS = 12


@dataclass(frozen=True)
class HeadSpec:
    """Architecture label and kwargs for :class:`VideoMAEViT`."""

    family: str
    label: str
    head: str = "mean"
    head_queries: int = 16
    temporal_mode: str = "none"
    temporal_layers: int = 0


def build_videomae_head_model(spec: HeadSpec) -> VideoMAEViT:
    """Instantiate a VideoMAE ViT-B model for analytic head-param counting."""
    return VideoMAEViT(
        num_classes=NUM_CLASSES,
        num_frames=4,
        tube_t=1,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        head=spec.head,
        head_queries=spec.head_queries,
        head_num_heads=HEAD_NUM_HEADS,
        head_mlp_ratio=HEAD_MLP_RATIO,
        temporal_mode=spec.temporal_mode,
        temporal_layers=spec.temporal_layers,
    )


def count_trainable_head_params(model: nn.Module) -> int:
    """Count parameters in pool/classifier/temporal head modules only."""
    return sum(p.numel() for name, p in model.named_parameters() if _is_new_module_param(name))


def count_head_params_for_spec(spec: HeadSpec) -> int:
    """Return trainable head params for one architecture specification."""
    model = build_videomae_head_model(spec)
    return count_trainable_head_params(model)


def meanpool_spec() -> HeadSpec:
    return HeadSpec("meanpool", "Mean pool", head="mean")


def perceiver_spec(q: int) -> HeadSpec:
    return HeadSpec("perceiver", f"Perceiver Q{q}", head="perceiver", head_queries=q)


def divspace_spec(k: int) -> HeadSpec:
    return HeadSpec(
        "divspace",
        f"DivST K{k}",
        head="perceiver",
        head_queries=16,
        temporal_mode="divided_st",
        temporal_layers=k,
    )
