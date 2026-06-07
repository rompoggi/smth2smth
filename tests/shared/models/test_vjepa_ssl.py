"""Tests for the V-JEPA-style clip-level SSL trunk, loss, and round-trip.

The V-JEPA SSL trunk *is* the supervised model's TSM-wrapped ResNet-50
backbone, so its state_dict (after stripping the ``trunk.backbone.``
prefix that ``pretrain_vjepa.run`` strips at save time) must be byte-for-
byte loadable into :class:`AvancedResNet50TSM.backbone` via
:func:`_ssl_trunk_to_supervised_keys`. These tests pin that contract and
guard against future drift.

We also check:
* the masking helper produces non-degenerate masks for any sane T;
* :func:`vjepa_feature_loss` returns 0 when no frames are masked and a
  finite, backprop-able value otherwise;
* :func:`update_vjepa_teacher_ema` matches the analytic midpoint formula
  at momentum 0.5.
"""

from __future__ import annotations

import copy

import torch
from omegaconf import OmegaConf

from smth2smth.pipelines.train import _ssl_trunk_to_supervised_keys
from smth2smth.shared.models import build_model
from smth2smth.shared.models.vjepa_ssl import (
    VJepaModel,
    VJepaTrunk,
    apply_frame_mask,
    make_frame_mask,
    update_vjepa_teacher_ema,
    vjepa_feature_loss,
)


def _make_vjepa_trunk_state(num_frames: int) -> dict[str, torch.Tensor]:
    """Reproduce exactly what ``pretrain_vjepa.run`` writes to disk."""
    student = VJepaModel(num_frames=num_frames, shift_div=8)
    return {
        k.removeprefix("trunk.backbone."): v
        for k, v in student.state_dict().items()
        if k.startswith("trunk.backbone.")
    }


class TestVJepaTrunkKeyRoundTrip:
    def test_remapped_trunk_keys_cover_supervised_backbone_exactly(self) -> None:
        trunk_state = _make_vjepa_trunk_state(num_frames=4)
        remapped = _ssl_trunk_to_supervised_keys(trunk_state)

        cfg = OmegaConf.create(
            {
                "model": {
                    "name": "avanced_resnet50_tsm",
                    "num_classes": 5,
                    "pretrained": False,
                    "shift_div": 8,
                    "shift_place": "blockres",
                    "dropout": 0.5,
                },
                "dataset": {"num_frames": 4},
            }
        )
        sup_model = build_model(cfg)
        sup_state = sup_model.state_dict()
        sup_backbone = {k for k in sup_state if k.startswith("backbone.")}

        # Every remapped key must exist in the supervised model.
        missing_in_sup = [k for k in remapped if k not in sup_state]
        assert missing_in_sup == [], (
            f"V-JEPA-remapped keys missing from supervised model: {missing_in_sup[:8]}"
        )
        # Every supervised backbone tensor is covered by the SSL trunk.
        not_covered = sup_backbone - set(remapped.keys())
        assert not_covered == set(), (
            f"backbone keys not covered by V-JEPA trunk: {sorted(not_covered)[:8]}"
        )

    def test_load_state_dict_only_misses_classifier(self) -> None:
        trunk_state = _make_vjepa_trunk_state(num_frames=4)
        remapped = _ssl_trunk_to_supervised_keys(trunk_state)
        cfg = OmegaConf.create(
            {
                "model": {
                    "name": "avanced_resnet50_tsm",
                    "num_classes": 5,
                    "pretrained": False,
                    "shift_div": 8,
                    "shift_place": "blockres",
                    "dropout": 0.5,
                    "drop_path_rate": 0.0,
                    "head": "mean",
                },
                "dataset": {"num_frames": 4},
            }
        )
        sup_model = build_model(cfg)
        missing, unexpected = sup_model.load_state_dict(remapped, strict=False)
        assert unexpected == [], f"unexpected keys: {unexpected[:4]}"
        backbone_missing = [k for k in missing if k.startswith("backbone.")]
        assert backbone_missing == [], f"backbone keys not loaded: {backbone_missing[:4]}"
        # The only allowed misses are the classifier head.
        assert set(missing) == {"classifier.weight", "classifier.bias"}, missing


class TestVJepaForwardAndLoss:
    def test_trunk_forward_shape(self) -> None:
        trunk = VJepaTrunk(num_frames=4)
        trunk.eval()
        with torch.no_grad():
            y = trunk(torch.randn(2, 4, 3, 32, 32))
        assert y.shape == (2, 4, 2048)

    def test_loss_zero_on_empty_mask(self) -> None:
        torch.manual_seed(0)
        predicted = torch.randn(3, 4, 16, requires_grad=True)
        teacher = torch.randn(3, 4, 16)
        mask = torch.zeros(3, 4, dtype=torch.bool)
        loss = vjepa_feature_loss(predicted, teacher, mask)
        assert loss.ndim == 0
        assert torch.allclose(loss, torch.zeros_like(loss))
        # ...and still has a grad_fn so the autograd graph is well-defined.
        assert loss.requires_grad

    def test_loss_finite_and_backprops(self) -> None:
        torch.manual_seed(0)
        model = VJepaModel(num_frames=4, predictor_hidden_dim=64)
        teacher = copy.deepcopy(model)
        for p in teacher.parameters():
            p.requires_grad_(False)

        clip = torch.randn(2, 4, 3, 32, 32)
        mask = torch.tensor(
            [[True, False, False, True], [False, True, True, False]], dtype=torch.bool
        )
        masked = apply_frame_mask(clip, mask)
        with torch.no_grad():
            teacher_feats = teacher.encode(clip)
        predicted, _ = model(masked)
        loss = vjepa_feature_loss(predicted, teacher_feats.detach(), mask)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        loss.backward()
        assert any(
            p.grad is not None and float(p.grad.abs().sum()) > 0.0 for p in model.parameters()
        )

    def test_apply_frame_mask_zeroes_correct_positions(self) -> None:
        clip = torch.ones(1, 3, 3, 4, 4)
        mask = torch.tensor([[False, True, False]], dtype=torch.bool)
        out = apply_frame_mask(clip, mask)
        assert float(out[0, 0].sum()) > 0  # frame 0 kept
        assert float(out[0, 1].abs().sum()) == 0.0  # frame 1 zeroed
        assert float(out[0, 2].sum()) > 0  # frame 2 kept

    def test_loss_fp16_inputs_do_not_saturate_denominator(self) -> None:
        """Regression: with ``B=32, T=4, D=2048`` and fp16 inputs (i.e.
        the realistic AMP setup), the denominator ``mask.sum() * D`` is
        ``~ 50 * 2048 = 102400``, which overflows fp16's max representable
        value (65,504). Before the fp32-internal fix, this silently sent
        ``n_elems`` to ``+inf`` and the loss to identically ``0.0`` with
        no gradient signal -- the bug surfaced as a real V-JEPA pretrain
        run logging ``loss 0.0000`` for thousands of steps."""
        torch.manual_seed(0)
        B, T, D = 32, 4, 2048
        predicted = torch.randn(B, T, D, dtype=torch.float16, requires_grad=True)
        teacher = torch.randn(B, T, D, dtype=torch.float16)
        # ~1.5 masked frames per clip on average, same distribution as
        # the realistic mask_prob=0.5, min=1, max=2 setting.
        mask = torch.zeros(B, T, dtype=torch.bool)
        for b in range(B):
            n = int(torch.randint(1, 3, (1,)).item())
            idx = torch.randperm(T)[:n]
            mask[b, idx] = True
        assert mask.sum().item() * D > 65504, (
            "Test premise broken: the denominator must exceed fp16's max."
        )

        loss = vjepa_feature_loss(predicted, teacher, mask)
        assert torch.isfinite(loss)
        assert float(loss) > 0.0, "Loss collapsed to 0 -- the fp16 saturation regression is back."
        # And gradient must flow.
        loss.backward()
        assert predicted.grad is not None
        assert float(predicted.grad.abs().sum()) > 0.0


class TestMaskSampling:
    def test_make_frame_mask_respects_bounds(self) -> None:
        torch.manual_seed(0)
        mask = make_frame_mask(batch_size=32, num_frames=4, mask_prob=0.5, min_mask=1, max_mask=2)
        per_clip = mask.sum(dim=1)
        assert int(per_clip.min().item()) >= 1
        assert int(per_clip.max().item()) <= 2

    def test_make_frame_mask_never_all_masked(self) -> None:
        torch.manual_seed(0)
        # Even with mask_prob=0.99 we keep at least one unmasked frame per clip.
        mask = make_frame_mask(
            batch_size=16, num_frames=4, mask_prob=0.99, min_mask=1, max_mask=None
        )
        per_clip = mask.sum(dim=1)
        assert int(per_clip.max().item()) <= 3


class TestTeacherEma:
    def test_midpoint_at_momentum_half(self) -> None:
        student = VJepaModel(num_frames=4, predictor_hidden_dim=32)
        teacher = copy.deepcopy(student)
        for p in student.parameters():
            p.data.add_(0.5 * torch.ones_like(p.data))
        old = {k: v.clone() for k, v in teacher.state_dict().items()}
        update_vjepa_teacher_ema(student, teacher, momentum=0.5)
        for k, v in teacher.state_dict().items():
            if not v.dtype.is_floating_point or v.ndim == 0:
                continue
            sup_param = student.state_dict()[k]
            expected = 0.5 * old[k] + 0.5 * sup_param
            torch.testing.assert_close(v, expected, atol=1e-6, rtol=1e-6)
            break
