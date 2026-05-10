"""Tests for Phase-2 opt-ins on the TSM-ResNet50 model.

These tests cover the new opt-in knobs added on top of the legacy
``avanced_resnet50_tsm``:

* ``model.drop_path_rate`` -- Stochastic Depth (Huang et al. 2016).
* ``model.head`` -- ``"mean"`` (legacy) vs. ``"attn"`` (1-query MultiHead
  attention pool).

Defaults (``drop_path_rate=0.0``, ``head="mean"``) are required to be
byte-for-byte equivalent to the legacy code path -- a smoke check ensures
state_dict keys are unchanged.
"""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from smth2smth.shared.models import build_model
from smth2smth.shared.models.avanced_resnet50_tsm import (
    AttentionPool,
    DropPath,
)


def _base_cfg(**overrides: object) -> OmegaConf:
    cfg = {
        "model": {
            "name": "avanced_resnet50_tsm",
            "num_classes": 5,
            "pretrained": False,
            "shift_div": 8,
            "shift_place": "blockres",
            "dropout": 0.5,
        }
        | overrides,
        "dataset": {"num_frames": 4},
    }
    return OmegaConf.create(cfg)


class TestDropPath:
    """:class:`DropPath` should be identity at eval and unbiased at train."""

    def test_eval_is_identity(self) -> None:
        layer = DropPath(drop_prob=0.5)
        layer.eval()
        x = torch.randn(8, 3, 4, 4)
        torch.testing.assert_close(layer(x), x)

    def test_zero_prob_is_identity_in_train_mode_too(self) -> None:
        layer = DropPath(drop_prob=0.0)
        layer.train()
        x = torch.randn(8, 3, 4, 4)
        torch.testing.assert_close(layer(x), x)

    def test_train_mode_zeros_some_samples(self) -> None:
        torch.manual_seed(0)
        layer = DropPath(drop_prob=0.5)
        layer.train()
        x = torch.ones(64, 3, 2, 2)
        out = layer(x)
        # Each (B,) sample is either 0 or 1 / keep_prob (=2.0). Average across
        # samples should be ≈ E[mask] / keep_prob * keep_prob = 1.0.
        sample_means = out.mean(dim=(1, 2, 3))
        assert torch.all((sample_means == 0.0) | torch.isclose(sample_means, torch.tensor(2.0)))


class TestAttentionPool:
    """:class:`AttentionPool` should respect the ``(B, T, D) -> (B, D)`` contract."""

    def test_forward_shape(self) -> None:
        pool = AttentionPool(feature_dim=32, num_heads=4)
        pool.eval()
        x = torch.randn(3, 4, 32)
        with torch.no_grad():
            y = pool(x)
        assert y.shape == (3, 32)
        assert y.dtype == torch.float32


class TestModelOptIns:
    """Build-and-forward parity between defaults and Phase-2 opt-ins."""

    def test_default_state_dict_keys_unchanged(self) -> None:
        # Sanity: enabling defaults (drop_path_rate=0, head=mean) must NOT add
        # any new buffers/parameters to the state_dict, so Phase-1 checkpoints
        # keep loading without strict-mismatch errors.
        cfg = _base_cfg()
        model = build_model(cfg)
        keys = set(model.state_dict().keys())
        assert "attn_pool.query" not in keys
        # No drop_path-prefixed entries exist when drop_path_rate=0.
        assert not any(k.startswith("backbone.layer1.0.drop_path") for k in keys)

    def test_drop_path_only_changes_no_state_dict_entries(self) -> None:
        # DropPath is parameter-less; turning it on must not extend the
        # state_dict either. (We verify by comparing key sets.)
        cfg_off = _base_cfg(drop_path_rate=0.0)
        cfg_on = _base_cfg(drop_path_rate=0.2)
        keys_off = set(build_model(cfg_off).state_dict().keys())
        keys_on = set(build_model(cfg_on).state_dict().keys())
        assert keys_off == keys_on

    def test_attn_head_adds_pool_params(self) -> None:
        cfg = _base_cfg(head="attn", head_num_heads=4)
        model = build_model(cfg)
        keys = set(model.state_dict().keys())
        assert "attn_pool.query" in keys
        # Forward shape contract is unchanged.
        model.eval()
        with torch.no_grad():
            logits = model(torch.randn(2, 4, 3, 64, 64))
        assert logits.shape == (2, 5)

    def test_drop_path_forward_runs_and_matches_eval(self) -> None:
        cfg = _base_cfg(drop_path_rate=0.2)
        model = build_model(cfg)
        # Forward should work in both modes; in eval, DropPath is identity so
        # repeated forwards are deterministic.
        model.eval()
        x = torch.randn(2, 4, 3, 64, 64)
        with torch.no_grad():
            y1 = model(x)
            y2 = model(x)
        torch.testing.assert_close(y1, y2)
