"""Smoke tests for the DINO SSL trunk + head + loss.

Asserts:
* trunk-only state_dict keys are a strict superset of the supervised
  ``AvancedResNet50TSM.backbone`` state_dict keys -- i.e. the SSL trunk can
  be loaded into the supervised model via ``model.init_from`` without
  missing any backbone tensors;
* :class:`DinoLoss` runs end-to-end on a single batch and produces a finite
  scalar gradient on the student;
* :func:`update_teacher_ema` actually moves the teacher towards the student
  (a sanity check on the in-place update).
"""

from __future__ import annotations

import copy

import torch
from omegaconf import OmegaConf

from smth2smth.pipelines.train import _ssl_trunk_to_supervised_keys
from smth2smth.shared.models import build_model
from smth2smth.shared.models.dino_ssl import (
    DinoLoss,
    DinoModel,
    update_teacher_ema,
)


class TestDinoModel:
    def test_remapped_trunk_keys_match_supervised_backbone(self) -> None:
        # The SSL trunk has plain ResNet-50 keys; the supervised model wraps
        # every block's ``conv1`` in ``Sequential(TemporalShift, conv1)``,
        # shifting the parameter key from ``layerN.M.conv1.weight`` to
        # ``layerN.M.conv1.1.weight``. ``_ssl_trunk_to_supervised_keys``
        # performs this remap; the remapped keys must be a subset of the
        # supervised model's state_dict keys (no missing backbone keys).
        student = DinoModel(out_dim=128, hidden_dim=64, bottleneck_dim=32, n_layers=3)
        trunk_state = {
            k.removeprefix("trunk."): v
            for k, v in student.state_dict().items()
            if k.startswith("trunk.")
        }
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
        # Every remapped key exists in the supervised model.
        missing = [k for k in remapped if k not in sup_state]
        assert missing == [], f"remapped keys missing from supervised model: {missing[:8]}"
        # And vice versa, every backbone key in the supervised model is
        # provided by the SSL trunk (no orphans).
        sup_backbone = {k for k in sup_state if k.startswith("backbone.")}
        not_covered = sup_backbone - set(remapped.keys())
        assert not_covered == set(), f"backbone keys not covered by SSL trunk: {not_covered}"

    def test_forward_shape(self) -> None:
        m = DinoModel(out_dim=64, hidden_dim=32, bottleneck_dim=16, n_layers=3)
        m.eval()
        with torch.no_grad():
            y = m(torch.randn(2, 3, 32, 32))
        assert y.shape == (2, 64)


class TestDinoLossAndEma:
    def test_loss_is_finite_and_backprops(self) -> None:
        torch.manual_seed(0)
        out_dim = 16
        student = DinoModel(out_dim=out_dim, hidden_dim=32, bottleneck_dim=16, n_layers=3)
        teacher = copy.deepcopy(student)
        for p in teacher.parameters():
            p.requires_grad_(False)
        loss_fn = DinoLoss(out_dim=out_dim)

        x_g1 = torch.randn(2, 3, 32, 32)
        x_g2 = torch.randn(2, 3, 32, 32)
        x_l1 = torch.randn(2, 3, 16, 16)

        with torch.no_grad():
            t_logits = [teacher(x_g1), teacher(x_g2)]
        s_logits = [student(x_g1), student(x_g2), student(x_l1)]

        loss = loss_fn(s_logits, t_logits)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        loss.backward()
        # At least one parameter has a non-zero grad.
        assert any(
            p.grad is not None and float(p.grad.abs().sum()) > 0.0 for p in student.parameters()
        )

    def test_teacher_ema_moves_towards_student(self) -> None:
        student = DinoModel(out_dim=16, hidden_dim=32, bottleneck_dim=16, n_layers=3)
        teacher = copy.deepcopy(student)
        # Perturb the student so the EMA has somewhere to move.
        for p in student.parameters():
            p.data.add_(0.5 * torch.ones_like(p.data))
        # With momentum=0.5, after one step teacher should be exactly at the
        # midpoint between its old self and the student.
        old_teacher_state = {k: v.clone() for k, v in teacher.state_dict().items()}
        update_teacher_ema(student, teacher, momentum=0.5)
        # Spot-check one parameter.
        for k, v in teacher.state_dict().items():
            if v.ndim == 0:
                continue
            old = old_teacher_state[k]
            sup_param = student.state_dict()[k]
            if v.dtype.is_floating_point:
                expected = 0.5 * old + 0.5 * sup_param
                torch.testing.assert_close(v, expected, atol=1e-6, rtol=1e-6)
                break
