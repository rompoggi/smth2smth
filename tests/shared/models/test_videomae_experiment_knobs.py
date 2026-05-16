"""Unit tests for VideoMAE experiment-batch helpers (2026-05-16)."""

from __future__ import annotations

import pytest
import torch

from smth2smth.pipelines.train import RepeatedAugSampler, _build_llrd_param_groups
from smth2smth.shared.engine.trainer import apply_video_mixing
from smth2smth.shared.models.video_mae import VideoMAEViT, interpolate_pos_embed


def test_interpolate_pos_embed_resolution_change() -> None:
    pe = torch.randn(1, 2 * 14 * 14, 64)
    out = interpolate_pos_embed(
        pe,
        src_num_frames=4,
        src_img_size=224,
        dst_num_frames=4,
        dst_img_size=256,
        tube_t=2,
        patch_size=16,
    )
    assert out.shape == (1, 2 * 16 * 16, 64)


def test_llrd_param_groups_cover_all_trainable() -> None:
    model = VideoMAEViT(
        num_classes=10,
        num_frames=4,
        img_size=224,
        embed_dim=384,
        depth=12,
        num_heads=6,
    )
    groups = _build_llrd_param_groups(
        model, base_lr=1e-3, weight_decay=0.05, layer_decay=0.75, depth=12
    )
    grouped = {id(p) for g in groups for p in g["params"]}
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert grouped == trainable
    top_lr = max(g["lr"] for g in groups)
    assert top_lr == pytest.approx(1e-3)


def test_repeated_aug_sampler_length() -> None:
    sampler = RepeatedAugSampler(100, repeats=2, shuffle=False)
    indices = list(sampler)
    assert len(indices) == 100


def test_mixup_cutmix_switch_produces_soft_targets() -> None:
    videos = torch.randn(4, 2, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])
    mixed, _, soft = apply_video_mixing(
        videos,
        labels,
        num_classes=10,
        alpha=1.0,
        mode="mixup_cutmix_switch",
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        switch_prob=0.5,
    )
    assert mixed.shape == videos.shape
    assert soft.shape == (4, 10)
