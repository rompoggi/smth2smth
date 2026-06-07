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

    def __init__(
        self,
        last_hidden_state: torch.Tensor,
        hidden_states: tuple[torch.Tensor, ...] | None = None,
    ) -> None:
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states


class _FakeConfig:
    """Minimal stand-in for ``transformers.VJEPA2Config``."""

    hidden_size: int = 64
    crop_size: int = 8
    tubelet_size: int = 2


class _FakeVJEPA2Backbone(nn.Module):
    """Tiny stand-in for :class:`transformers.VJEPA2Model`.

    Implements the same (B, T, C, H, W) -> ``last_hidden_state`` interface
    used by V-JEPA 2: flattens each frame and projects to a fixed token
    sequence of shape ``(B, T, hidden_size)``. When ``output_hidden_states``
    is requested it also returns a tuple of per-layer hidden states (here a
    short stack of identical projections) so the last-K-block concat path in
    :meth:`VJEPA2Probe._encode` can be exercised.
    """

    def __init__(self, num_hidden_states: int = 3) -> None:
        super().__init__()
        self.config = _FakeConfig()
        self.embed = nn.Linear(3 * 8 * 8, self.config.hidden_size)
        self._num_hidden_states = int(num_hidden_states)

    def forward(
        self,
        pixel_values_videos: torch.Tensor | None = None,
        skip_predictor: bool = True,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> _FakeOutput:
        assert pixel_values_videos is not None
        b, t, c, h, w = pixel_values_videos.shape
        flat = pixel_values_videos.reshape(b * t, c * h * w)
        feats = self.embed(flat).reshape(b, t, -1)
        hidden_states = None
        if output_hidden_states:
            # Distinct per-layer states so concatenation is not a no-op.
            hidden_states = tuple(feats + float(i) for i in range(self._num_hidden_states))
        return _FakeOutput(last_hidden_state=feats, hidden_states=hidden_states)


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


class TestMultiBlockAttentiveProbe:
    """Architectural contracts for Meta's 4-block attentive classifier (E3)."""

    def test_forward_returns_expected_shape(self) -> None:
        from smth2smth.track_b.vjepa2 import MultiBlockAttentiveProbe

        head = MultiBlockAttentiveProbe(feature_dim=64, num_classes=10, num_heads=16, depth=4)
        head.eval()
        tokens = torch.randn(3, 17, 64)
        with torch.no_grad():
            logits = head(tokens)
        assert logits.shape == (3, 10)

    def test_grads_flow_through_self_attn_blocks(self) -> None:
        from smth2smth.track_b.vjepa2 import MultiBlockAttentiveProbe

        head = MultiBlockAttentiveProbe(feature_dim=32, num_classes=5, num_heads=8, depth=2)
        head(torch.randn(2, 8, 32)).sum().backward()
        assert head.classifier.weight.grad is not None
        assert head.query.grad is not None

    def test_invalid_num_heads_raises(self) -> None:
        from smth2smth.track_b.vjepa2 import MultiBlockAttentiveProbe

        with pytest.raises(ValueError):
            MultiBlockAttentiveProbe(feature_dim=64, num_classes=3, num_heads=7)

    def test_zero_depth_raises(self) -> None:
        from smth2smth.track_b.vjepa2 import MultiBlockAttentiveProbe

        with pytest.raises(ValueError, match="depth"):
            MultiBlockAttentiveProbe(feature_dim=64, num_classes=3, depth=0)


class TestLastKBlockConcat:
    """Last-K-block concat widens the head's input channel dim (E3)."""

    def test_feature_dim_scales_with_k(self, fake_transformers: types.ModuleType) -> None:
        from smth2smth.track_b.vjepa2 import VJEPA2Probe

        model = VJEPA2Probe(
            num_classes=7,
            hf_repo="stub-vjepa2",
            head_type="multi_block_attentive",
            head_num_heads=8,
            head_depth=2,
            head_last_k_blocks=2,
        )
        # Backbone hidden is 64; K=2 concat -> 128-wide head input.
        assert model.feature_dim == 128
        model.eval()
        with torch.no_grad():
            logits = model(torch.randn(2, 4, 3, 8, 8))
        assert logits.shape == (2, 7)

    def test_k1_matches_single_block_dim(self, fake_transformers: types.ModuleType) -> None:
        from smth2smth.track_b.vjepa2 import VJEPA2Probe

        model = VJEPA2Probe(
            num_classes=4,
            hf_repo="stub-vjepa2",
            head_type="attentive",
            head_num_heads=8,
            head_last_k_blocks=1,
        )
        assert model.feature_dim == 64


class TestDoraPlumbing:
    """``use_dora`` selects PEFT DoRA over vanilla LoRA (E5).

    Exercises the *real* ``peft`` package against a tiny attention-shaped
    module so the assertion tracks PEFT's actual parameter layout.
    """

    @staticmethod
    def _tiny_attn() -> nn.Module:
        class _TinyAttn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.query = nn.Linear(8, 8)
                self.key = nn.Linear(8, 8)
                self.value = nn.Linear(8, 8)
                self.proj = nn.Linear(8, 8)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.proj(self.value(x))

        return _TinyAttn()

    def test_dora_adds_magnitude_vector(self) -> None:
        pytest.importorskip("peft")
        from smth2smth.track_b.vjepa2 import _try_apply_peft_lora

        wrapped = _try_apply_peft_lora(
            self._tiny_attn(),
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["query", "key", "value", "proj"],
            use_dora=True,
        )
        names = [n for n, _ in wrapped.named_parameters()]
        assert any("lora_magnitude_vector" in n for n in names), names
        assert any("lora_A" in n for n in names)

    def test_vanilla_lora_has_no_magnitude_vector(self) -> None:
        pytest.importorskip("peft")
        from smth2smth.track_b.vjepa2 import _try_apply_peft_lora

        wrapped = _try_apply_peft_lora(
            self._tiny_attn(),
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["query", "key", "value", "proj"],
            use_dora=False,
        )
        names = [n for n, _ in wrapped.named_parameters()]
        assert any("lora_A" in n for n in names)
        assert not any("lora_magnitude_vector" in n for n in names)


class _FakeSSv2FTConfig:
    """Stand-in for ``VJEPA2ForVideoClassification.config``."""

    def __init__(self, hidden_size: int, id2label: dict[int, str]) -> None:
        self.hidden_size = hidden_size
        self.id2label = id2label
        self.num_labels = len(id2label)


class _FakeLogitsOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _FakeSSv2FTModel(nn.Module):
    """Stand-in for :class:`transformers.VJEPA2ForVideoClassification`.

    The ``classifier`` rows are seeded deterministically (row ``j`` is filled
    with the constant ``j``) so a head-slice can be verified row-by-row
    against the local->SSv2 index map.
    """

    def __init__(self, hidden_size: int, id2label: dict[int, str]) -> None:
        super().__init__()
        self.config = _FakeSSv2FTConfig(hidden_size, id2label)
        self.vjepa2 = _FakeVJEPA2Backbone()
        n = len(id2label)
        self.classifier = nn.Linear(hidden_size, n)
        with torch.no_grad():
            for j in range(n):
                self.classifier.weight[j].fill_(float(j))
                self.classifier.bias[j].fill_(float(j))
        self.num_labels = n

    def forward(
        self, pixel_values_videos: torch.Tensor | None = None, **kwargs: Any
    ) -> _FakeLogitsOutput:
        feats = self.vjepa2(pixel_values_videos=pixel_values_videos).last_hidden_state
        pooled = feats.mean(dim=1)
        return _FakeLogitsOutput(self.classifier(pooled))


@pytest.fixture
def fake_transformers_ssv2ft(monkeypatch: pytest.MonkeyPatch) -> dict[int, str]:
    """Install a fake ``transformers`` exposing ``VJEPA2ForVideoClassification``.

    Returns the ``id2label`` map used by the stub so tests can assert the
    head-slice picked the right SSv2 rows.
    """
    # SSv2 idx deliberately differs from the local folder idx to prove the
    # mapping is by *name*, not by position.
    id2label = {
        0: "Closing something",
        1: "Opening something",
        2: "Approaching something with your camera",
        3: "Bending something so that it deforms",
    }
    fake_module = types.ModuleType("transformers")

    class _FakeVJEPA2ForVideoClassification:
        @staticmethod
        def from_pretrained(repo_id: str, **kwargs: Any) -> _FakeSSv2FTModel:
            _ = repo_id, kwargs
            return _FakeSSv2FTModel(hidden_size=64, id2label=id2label)

    fake_module.VJEPA2ForVideoClassification = _FakeVJEPA2ForVideoClassification  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_module)
    return id2label


def _make_local_class_dirs(tmp_path: Any, names: list[str]) -> Any:
    """Create ``tmp_path/<name>/`` folders and return ``tmp_path``."""
    for name in names:
        (tmp_path / name).mkdir()
    return tmp_path


class TestVJEPA2SSv2FTProbe:
    """SSv2-FT checkpoint swap + name-aligned head-slice (E1)."""

    def test_head_slice_copies_mapped_rows(
        self, fake_transformers_ssv2ft: dict[int, str], tmp_path: Any
    ) -> None:
        from smth2smth.track_b.vjepa2 import VJEPA2SSv2FTProbe

        # Local folders for indices 0 and 2 only; indices 1 and 3 have no
        # folder, so their rows must fall back to random init.
        label_dir = _make_local_class_dirs(
            tmp_path,
            ["000_Approaching_something_with_your_camera", "002_Closing_something"],
        )
        probe = VJEPA2SSv2FTProbe(
            num_classes=4,
            hf_repo="stub-ssv2ft",
            label_source_dir=label_dir,
            init_head_from_ssv2=True,
            lora_enabled=False,
        )
        # local 0 ("approaching...") -> ssv2 idx 2; local 2 ("closing...") -> ssv2 idx 0.
        assert probe.local_to_ssv2_idx.tolist() == [2, -1, 0, -1]
        w = probe.model.classifier.weight.detach()
        # Resolved rows copy the constant-seeded SSv2 rows exactly.
        assert torch.allclose(w[0], torch.full_like(w[0], 2.0))
        assert torch.allclose(w[2], torch.full_like(w[2], 0.0))
        # Unresolved rows are small-Gaussian, not a clean integer constant.
        assert not torch.allclose(w[1], torch.full_like(w[1], w[1][0].item()))

    def test_forward_shape(self, fake_transformers_ssv2ft: dict[int, str], tmp_path: Any) -> None:
        from smth2smth.track_b.vjepa2 import VJEPA2SSv2FTProbe

        label_dir = _make_local_class_dirs(
            tmp_path,
            ["000_Approaching_something_with_your_camera", "002_Closing_something"],
        )
        probe = VJEPA2SSv2FTProbe(
            num_classes=4,
            hf_repo="stub-ssv2ft",
            label_source_dir=label_dir,
            init_head_from_ssv2=True,
            lora_enabled=False,
        )
        probe.eval()
        with torch.no_grad():
            logits = probe(torch.randn(2, 4, 3, 8, 8))
        assert logits.shape == (2, 4)

    def test_frozen_probe_keeps_encoder_frozen(
        self, fake_transformers_ssv2ft: dict[int, str], tmp_path: Any
    ) -> None:
        from smth2smth.track_b.vjepa2 import VJEPA2SSv2FTProbe

        label_dir = _make_local_class_dirs(tmp_path, ["000_Approaching_something_with_your_camera"])
        probe = VJEPA2SSv2FTProbe(
            num_classes=4,
            hf_repo="stub-ssv2ft",
            label_source_dir=label_dir,
            init_head_from_ssv2=True,
            freeze_backbone_base=True,
            lora_enabled=False,
        )
        assert not any(p.requires_grad for p in probe.model.vjepa2.parameters())
        assert probe.model.classifier.weight.requires_grad

    def test_unmatched_folder_raises(
        self, fake_transformers_ssv2ft: dict[int, str], tmp_path: Any
    ) -> None:
        from smth2smth.track_b.vjepa2 import VJEPA2SSv2FTProbe

        # Folder name has no SSv2 counterpart -> head-slice must fail loudly.
        label_dir = _make_local_class_dirs(tmp_path, ["000_Totally_unknown_action"])
        with pytest.raises(RuntimeError, match="align"):
            VJEPA2SSv2FTProbe(
                num_classes=4,
                hf_repo="stub-ssv2ft",
                label_source_dir=label_dir,
                init_head_from_ssv2=True,
                lora_enabled=False,
            )

    def test_registry_exposes_ssv2ft(self) -> None:
        from smth2smth.shared.models import list_registered_models

        assert "vjepa2_ssv2ft" in list_registered_models()


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
