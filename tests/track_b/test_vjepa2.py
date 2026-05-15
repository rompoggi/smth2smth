"""Unit tests for the V-JEPA 2 Track-B probe head.

These tests exercise the architectural wiring of
:class:`smth2smth.track_b.vjepa2.VJEPA2Probe` *without* downloading the
actual 300 M-param V-JEPA 2 checkpoint from HuggingFace Hub. We install a
fake ``transformers`` module into ``sys.modules`` whose ``AutoModel``
returns a tiny stub backbone with the same API surface (forward signature
expects ``pixel_values_videos``, exposes ``config.hidden_size``).

For pure head shape/grad checks (:class:`AttentiveProbe`,
:class:`MeanLinearHead`), no mock is needed since those classes do not
touch transformers.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf


class _FakeOutput:
    """Minimal stand-in for ``transformers.modeling_outputs.ModelOutput``."""

    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class _FakeConfig:
    """Minimal stand-in for ``transformers.VJEPA2Config``."""

    hidden_size: int = 64
    crop_size: int = 8
    tubelet_size: int = 2


class _FakeVJEPA2Backbone(nn.Module):
    """Tiny stand-in for :class:`transformers.VJEPA2Model`.

    Implements the same (B, T, C, H, W) -> ``last_hidden_state`` interface
    used by V-JEPA 2: flattens each frame and projects to a fixed token
    sequence of shape ``(B, T, hidden_size)``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = _FakeConfig()
        self.embed = nn.Linear(3 * 8 * 8, self.config.hidden_size)

    def forward(
        self,
        pixel_values_videos: torch.Tensor | None = None,
        skip_predictor: bool = True,
        **kwargs: Any,
    ) -> _FakeOutput:
        assert pixel_values_videos is not None
        b, t, c, h, w = pixel_values_videos.shape
        flat = pixel_values_videos.reshape(b * t, c * h * w)
        feats = self.embed(flat).reshape(b, t, -1)
        return _FakeOutput(last_hidden_state=feats)


@pytest.fixture
def fake_transformers(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a fake ``transformers`` module exposing ``AutoModel.from_pretrained``."""
    fake_module = types.ModuleType("transformers")

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(repo_id: str, **kwargs: Any) -> _FakeVJEPA2Backbone:
            _ = repo_id, kwargs
            return _FakeVJEPA2Backbone()

    fake_module.AutoModel = _FakeAutoModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_module)
    return fake_module


class TestAttentiveProbe:
    """Architectural contracts for :class:`AttentiveProbe`."""

    def test_forward_returns_expected_shape(self) -> None:
        from smth2smth.track_b.vjepa2 import AttentiveProbe

        head = AttentiveProbe(feature_dim=64, num_classes=10, num_heads=8)
        head.eval()
        tokens = torch.randn(3, 17, 64)
        with torch.no_grad():
            logits = head(tokens)
        assert logits.shape == (3, 10)
        assert logits.dtype == torch.float32

    def test_classifier_grads_flow(self) -> None:
        from smth2smth.track_b.vjepa2 import AttentiveProbe

        head = AttentiveProbe(feature_dim=32, num_classes=5, num_heads=4)
        tokens = torch.randn(2, 8, 32)
        loss = head(tokens).sum()
        loss.backward()
        assert head.classifier.weight.grad is not None
        assert head.classifier.weight.grad.abs().sum().item() > 0.0

    def test_invalid_num_heads_raises_explicit_error(self) -> None:
        from smth2smth.track_b.vjepa2 import AttentiveProbe

        # 7 does not divide 64; must raise before nn.MultiheadAttention does.
        with pytest.raises(ValueError):
            AttentiveProbe(feature_dim=64, num_classes=3, num_heads=7)

    def test_multi_query_forward_shape(self) -> None:
        from smth2smth.track_b.vjepa2 import AttentiveProbe

        head = AttentiveProbe(
            feature_dim=64, num_classes=7, num_heads=8, num_queries=4, dropout=0.0
        )
        head.eval()
        tokens = torch.randn(2, 10, 64)
        with torch.no_grad():
            logits = head(tokens)
        assert logits.shape == (2, 7)

    def test_num_queries_zero_raises(self) -> None:
        from smth2smth.track_b.vjepa2 import AttentiveProbe

        with pytest.raises(ValueError, match="num_queries"):
            AttentiveProbe(feature_dim=64, num_classes=3, num_heads=8, num_queries=0)


class TestMeanLinearHead:
    """Architectural contracts for :class:`MeanLinearHead`."""

    def test_forward_returns_expected_shape(self) -> None:
        from smth2smth.track_b.vjepa2 import MeanLinearHead

        head = MeanLinearHead(feature_dim=64, num_classes=10, dropout=0.1)
        head.eval()
        tokens = torch.randn(3, 17, 64)
        with torch.no_grad():
            logits = head(tokens)
        assert logits.shape == (3, 10)

    def test_grads_flow(self) -> None:
        from smth2smth.track_b.vjepa2 import MeanLinearHead

        head = MeanLinearHead(feature_dim=32, num_classes=5, dropout=0.0)
        tokens = torch.randn(2, 4, 32)
        head(tokens).sum().backward()
        assert head.classifier.weight.grad is not None


class TestVJEPA2Probe:
    """End-to-end checks against a mocked HuggingFace V-JEPA 2 backbone."""

    def test_forward_returns_expected_shape(self, fake_transformers: types.ModuleType) -> None:
        from smth2smth.track_b.vjepa2 import VJEPA2Probe

        model = VJEPA2Probe(
            num_classes=33,
            hf_repo="stub-vjepa2",
            head_type="attentive",
            head_num_heads=8,
        )
        model.eval()
        video = torch.randn(2, 4, 3, 8, 8)
        with torch.no_grad():
            logits = model(video)
        assert logits.shape == (2, 33)
        assert model.feature_dim == 64
        assert model.hf_repo == "stub-vjepa2"

    @pytest.mark.parametrize("head_type", ["attentive", "linear", "mean_linear"])
    def test_all_head_types_produce_logits(
        self, fake_transformers: types.ModuleType, head_type: str
    ) -> None:
        from smth2smth.track_b.vjepa2 import VJEPA2Probe

        model = VJEPA2Probe(
            num_classes=12,
            hf_repo="stub-vjepa2",
            head_type=head_type,
            head_num_heads=8,
        )
        model.eval()
        video = torch.randn(2, 4, 3, 8, 8)
        with torch.no_grad():
            logits = model(video)
        assert logits.shape == (2, 12)

    def test_invalid_head_type_raises(self, fake_transformers: types.ModuleType) -> None:
        from smth2smth.track_b.vjepa2 import VJEPA2Probe

        with pytest.raises(ValueError):
            VJEPA2Probe(num_classes=5, hf_repo="stub-vjepa2", head_type="bogus")  # type: ignore[arg-type]

    def test_freeze_backbone_disables_grads(self, fake_transformers: types.ModuleType) -> None:
        from smth2smth.track_b.vjepa2 import VJEPA2Probe

        model = VJEPA2Probe(
            num_classes=5,
            hf_repo="stub-vjepa2",
            freeze_backbone=True,
            head_num_heads=8,
        )
        backbone_grads = [p.requires_grad for p in model.backbone.parameters()]
        head_grads = [p.requires_grad for p in model.head.parameters()]
        assert not any(backbone_grads), "frozen backbone should have requires_grad=False everywhere"
        assert all(head_grads), "head parameters must remain trainable"

    def test_unfreeze_backbone_enables_grads(self, fake_transformers: types.ModuleType) -> None:
        from smth2smth.track_b.vjepa2 import VJEPA2Probe

        model = VJEPA2Probe(
            num_classes=5,
            hf_repo="stub-vjepa2",
            freeze_backbone=False,
            head_num_heads=8,
        )
        assert any(p.requires_grad for p in model.backbone.parameters())

    def test_train_keeps_frozen_backbone_in_eval_mode(
        self, fake_transformers: types.ModuleType
    ) -> None:
        from smth2smth.track_b.vjepa2 import VJEPA2Probe

        model = VJEPA2Probe(
            num_classes=5,
            hf_repo="stub-vjepa2",
            freeze_backbone=True,
            head_num_heads=8,
        )
        model.train()
        assert not model.backbone.training, "frozen backbone must stay in eval mode"
        assert model.head.training, "head must follow the requested mode"
        model.eval()
        assert not model.backbone.training
        assert not model.head.training

    def test_frozen_forward_does_not_accumulate_backbone_grads(
        self, fake_transformers: types.ModuleType
    ) -> None:
        from smth2smth.track_b.vjepa2 import VJEPA2Probe

        model = VJEPA2Probe(
            num_classes=5,
            hf_repo="stub-vjepa2",
            freeze_backbone=True,
            head_num_heads=8,
        )
        model.train()
        video = torch.randn(2, 4, 3, 8, 8)
        loss = model(video).sum()
        loss.backward()
        backbone_with_grad = [p for p in model.backbone.parameters() if p.grad is not None]
        assert backbone_with_grad == [], "frozen backbone must not receive gradients"
        head_with_grad = [p for p in model.head.parameters() if p.grad is not None]
        assert len(head_with_grad) > 0, "head must accumulate gradients"


class TestRegistryIntegration:
    """The ``vjepa2`` name is exposed through the shared model registry."""

    def test_vjepa2_is_registered(self) -> None:
        # Importing ``shared.models`` already pulls in ``track_b`` via the
        # registration import added there, so the name must be present.
        from smth2smth.shared.models import list_registered_models

        assert "vjepa2" in list_registered_models()

    def test_build_model_dispatches_via_registry(self, fake_transformers: types.ModuleType) -> None:
        from smth2smth.shared.models import build_model
        from smth2smth.track_b.vjepa2 import VJEPA2Probe

        cfg = OmegaConf.create(
            {
                "num_classes": 4,
                "model": {
                    "name": "vjepa2",
                    "pretrained": True,
                    "num_classes": 4,
                    "hf_repo": "stub-vjepa2",
                "head_type": "attentive",
                "head_num_heads": 8,
                "head_num_queries": 1,
                "head_dropout": 0.0,
                "freeze_backbone": True,
                "lora_enabled": False,
                "attn_implementation": "eager",
                },
            }
        )
        model = build_model(cfg)
        assert isinstance(model, VJEPA2Probe)
        model.eval()
        with torch.no_grad():
            logits = model(torch.randn(1, 4, 3, 8, 8))
        assert logits.shape == (1, 4)
