"""Tests for the four diverse classifier-head / temporal architectures.

Covers the experiment-doc "top failure mode" checks:
  - Arch 1: head MLP liveness diagnostic is finite and positive.
  - Arch 2: query pairwise cosine is low at init (no collapse) and the metric
    flags an artificially-collapsed query set.
  - Arch 3 & 4: identity-at-init — the temporal path contributes zero at step 0,
    so the encoder output equals the plain pretrained backbone (the doc's single
    highest-payoff sanity check).
  - LLRD routing: the new temporal modules + the pooling head land in the
    full-base-LR group, not the LLRD-decayed block rate.

All models are built tiny so the suite runs on CPU in well under a second.
"""

from __future__ import annotations

import pytest
import torch

from smth2smth.pipelines.train import (
    _build_llrd_param_groups,
    _build_llrd_stabilized_param_groups,
    _is_new_temporal_param,
    _videomae_layer_id,
)
from smth2smth.shared.models.video_mae import (
    AIMReuseBlock,
    CrossAttnPoolHead,
    DividedSpaceTimeBlock,
    VideoMAEViT,
)

# Tiny ViT-B-shaped config: embed_dim divisible by both backbone and head heads.
DIM = 96
HEADS = 6
DEPTH = 4
FRAMES = 4
IMG = 32  # → n_h = n_w = 2, n_t = 4 → N = 16 tokens


def _make_model(**overrides) -> VideoMAEViT:
    kwargs = dict(
        num_classes=7,
        num_frames=FRAMES,
        img_size=IMG,
        tube_t=1,
        patch_size=16,
        embed_dim=DIM,
        depth=DEPTH,
        num_heads=HEADS,
    )
    kwargs.update(overrides)
    return VideoMAEViT(**kwargs)


def _dummy_clip(batch: int = 2) -> torch.Tensor:
    return torch.randn(batch, FRAMES, 3, IMG, IMG)


# ── Forward shapes ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "head_kwargs",
    [
        {"head": "mean"},
        {"head": "attn", "head_num_heads": HEADS},
        {"head": "attn_probe", "head_num_heads": HEADS},
        {"head": "perceiver", "head_num_heads": HEADS, "head_queries": 16},
        {
            "head": "perceiver",
            "head_num_heads": HEADS,
            "head_queries": 16,
            "temporal_mode": "divided_st",
            "temporal_layers": 2,
        },
        {
            "head": "perceiver",
            "head_num_heads": HEADS,
            "head_queries": 16,
            "temporal_mode": "aim_reuse",
            "temporal_layers": DEPTH,
            "aim_bottleneck": 16,
        },
    ],
)
def test_forward_shapes(head_kwargs) -> None:
    model = _make_model(**head_kwargs).eval()
    out = model(_dummy_clip())
    assert out.shape == (2, 7)
    assert torch.isfinite(out).all()


# ── Arch 3 / Arch 4 identity-at-init ─────────────────────────────────────────


@pytest.mark.parametrize("temporal_mode", ["divided_st", "aim_reuse"])
def test_temporal_identity_at_init(temporal_mode) -> None:
    """The temporal path must be a no-op at step 0: with the same backbone
    weights, encoder(temporal) == encoder(plain)."""
    plain = _make_model(head="mean").eval()
    temporal = _make_model(
        head="mean",
        temporal_mode=temporal_mode,
        temporal_layers=DEPTH,
        aim_bottleneck=16,
    ).eval()
    # Copy the plain backbone into the temporal model (shared keys only); the
    # temporal modules keep their identity init.
    missing, unexpected = temporal.encoder.load_state_dict(plain.encoder.state_dict(), strict=False)
    # Every plain key must be consumed; only new temporal keys may be "missing".
    assert not unexpected
    assert all(_is_temporal_leaf(k) for k in missing), missing

    x = _dummy_clip()
    with torch.no_grad():
        ref = plain.encoder(x)
        got = temporal.encoder(x)
    assert torch.allclose(ref, got, atol=1e-5, rtol=1e-4)


def _is_temporal_leaf(key: str) -> bool:
    return any(
        seg in key
        for seg in ("norm_t.", "temporal_attn.", "temporal_fc.", "t_adapter.", "joint_adapter.")
    )


def test_temporal_fc_and_adapters_are_zero_at_init() -> None:
    div = _make_model(temporal_mode="divided_st", temporal_layers=DEPTH)
    for blk in div.encoder.blocks:
        if isinstance(blk, DividedSpaceTimeBlock):
            assert torch.count_nonzero(blk.temporal_fc.weight) == 0
            assert torch.count_nonzero(blk.temporal_fc.bias) == 0
    aim = _make_model(temporal_mode="aim_reuse", temporal_layers=DEPTH, aim_bottleneck=16)
    for blk in aim.encoder.blocks:
        if isinstance(blk, AIMReuseBlock):
            for adapter in (blk.t_adapter, blk.joint_adapter):
                assert torch.count_nonzero(adapter.up.weight) == 0
                assert torch.count_nonzero(adapter.up.bias) == 0


def test_divided_st_wraps_only_last_k_blocks() -> None:
    model = _make_model(temporal_mode="divided_st", temporal_layers=2)
    kinds = [isinstance(b, DividedSpaceTimeBlock) for b in model.encoder.blocks]
    assert kinds == [False, False, True, True]


# ── LLRD routing ────────────────────────────────────────────────────────────


def test_new_temporal_params_route_to_top_lr() -> None:
    model = _make_model(
        head="perceiver",
        head_num_heads=HEADS,
        head_queries=16,
        temporal_mode="divided_st",
        temporal_layers=2,
    )
    base_lr = 5e-4
    groups = _build_llrd_param_groups(
        model, base_lr=base_lr, weight_decay=0.05, layer_decay=0.75, depth=DEPTH
    )
    # Map every param id to the lr of the group it landed in.
    lr_of: dict[int, float] = {}
    for g in groups:
        for p in g["params"]:
            lr_of[id(p)] = g["lr"]

    n_temporal = 0
    n_head = 0
    for name, param in model.named_parameters():
        if _is_new_temporal_param(name):
            n_temporal += 1
            assert lr_of[id(param)] == pytest.approx(base_lr), f"{name} not at base LR"
        if name.startswith(("pool_head.", "classifier.")):
            n_head += 1
            assert lr_of[id(param)] == pytest.approx(base_lr), f"{name} not at base LR"
    assert n_temporal > 0 and n_head > 0
    # A pretrained early block must be strictly below base LR (LLRD active).
    block0_lr = next(
        lr_of[id(p)] for n, p in model.named_parameters() if n.startswith("encoder.blocks.0.")
    )
    assert block0_lr < base_lr


def test_stabilized_llrd_lowers_new_module_lr() -> None:
    model = _make_model(
        head="perceiver",
        head_num_heads=HEADS,
        head_queries=16,
        temporal_mode="divided_st",
        temporal_layers=2,
    )
    base_lr = 5e-4
    new_lr = 1e-4
    groups = _build_llrd_stabilized_param_groups(
        model,
        base_lr=base_lr,
        new_module_lr=new_lr,
        weight_decay=0.05,
        layer_decay=0.75,
        depth=DEPTH,
        backbone_warmup_epochs=5,
        new_module_warmup_epochs=10,
    )
    lr_of: dict[int, float] = {}
    warm_of: dict[int, int] = {}
    for g in groups:
        for p in g["params"]:
            lr_of[id(p)] = g["lr"]
            warm_of[id(p)] = int(g.get("warmup_epochs", 0))
    for name, param in model.named_parameters():
        if name.startswith(("pool_head.", "classifier.")) or _is_new_temporal_param(name):
            assert lr_of[id(param)] == pytest.approx(new_lr), name
            assert warm_of[id(param)] == 10, name
    block0_lr = next(
        lr_of[id(p)] for n, p in model.named_parameters() if n.startswith("encoder.blocks.0.")
    )
    assert block0_lr < base_lr
    assert block0_lr > new_lr


def test_is_new_temporal_param_name_matching() -> None:
    assert _is_new_temporal_param("encoder.blocks.6.temporal_fc.weight")
    assert _is_new_temporal_param("encoder.blocks.11.t_adapter.up.weight")
    assert _is_new_temporal_param("encoder.blocks.0.norm_t.weight")
    # Pretrained block params and head params are NOT temporal.
    assert not _is_new_temporal_param("encoder.blocks.6.attn.in_proj_weight")
    assert not _is_new_temporal_param("encoder.blocks.6.mlp.fc1.weight")
    assert not _is_new_temporal_param("pool_head.queries")
    assert _videomae_layer_id("encoder.blocks.6.temporal_fc.weight", DEPTH) == DEPTH + 1


# ── Head diagnostics (Arch 1 MLP liveness, Arch 2 query collapse) ─────────────


def test_query_pairwise_cosine_low_at_init_high_when_collapsed() -> None:
    head = CrossAttnPoolHead(DIM, num_heads=HEADS, num_queries=16)
    # Random trunc-normal queries → near-orthogonal → low mean cosine.
    assert head.query_pairwise_cosine() < 0.5
    # Force collapse: all queries identical → cosine ≈ 1.
    with torch.no_grad():
        head.queries.copy_(head.queries[:, :1, :].expand_as(head.queries))
    assert head.query_pairwise_cosine() > 0.7
    # Single-query head reports 0 (no pairs).
    single = CrossAttnPoolHead(DIM, num_heads=HEADS, num_queries=1)
    assert single.query_pairwise_cosine() == 0.0


def test_mlp_activity_ratio_refreshed_on_eval_forward() -> None:
    model = _make_model(head="attn_probe", head_num_heads=HEADS).eval()
    assert model.pool_head is not None
    _ = model(_dummy_clip())
    ratio = model.pool_head.last_mlp_activity_ratio
    assert ratio == ratio  # not NaN
    assert ratio > 0.0
    diag = model.head_diagnostics()
    assert "head/mlp_activity_ratio" in diag


def test_invalid_head_and_temporal_mode_raise() -> None:
    with pytest.raises(ValueError):
        _make_model(head="bogus")
    with pytest.raises(ValueError):
        _make_model(temporal_mode="bogus")
    # Temporal modes require the prenorm residual variant.
    with pytest.raises(ValueError):
        _make_model(temporal_mode="divided_st", temporal_layers=2, residual_variant="shc")
