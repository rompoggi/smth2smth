"""End-to-end smoke test for V-JEPA ``model.init_from`` warm-start."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from smth2smth.pipelines.train import _ssl_trunk_to_supervised_keys
from smth2smth.shared.models import build_model
from smth2smth.shared.models.vjepa_ssl import VJepaModel


def test_vjepa_trunk_loads_into_supervised_model_with_remap(tmp_path: Path) -> None:
    """Build a VJepaModel, save its trunk-only state_dict to disk exactly as
    ``pretrain_vjepa.run`` does, then load it into a freshly-built
    ``AvancedResNet50TSM`` via the same remap + non-strict load the
    supervised pipeline uses.

    Asserts:
    * No backbone key is left at its random init.
    * The supervised model's classifier and attention pool keys are *not*
      touched.
    * The on-disk ``trunk_state_dict`` keys are plain ResNet keys (no
      double ``backbone.`` prefix); this guards against the regression
      where ``VJepaTrunk`` wraps the resnet in ``self.backbone`` and a
      naive prefix-strip would produce ``backbone.backbone.*`` after
      remap.
    """
    torch.manual_seed(0)
    student = VJepaModel(num_frames=4, predictor_hidden_dim=64)

    # Mirror the save logic in ``pretrain_vjepa.run`` (strip
    # ``trunk.backbone.`` rather than just ``trunk.``).
    trunk_state_dict = {
        k.removeprefix("trunk.backbone."): v
        for k, v in student.state_dict().items()
        if k.startswith("trunk.backbone.")
    }
    # Sanity: no key starts with ``backbone.`` (we stripped it).
    assert not any(k.startswith("backbone.") for k in trunk_state_dict), (
        "V-JEPA trunk save must strip the ``backbone.`` nesting so the "
        "supervised loader can add it back exactly once."
    )
    ssl_path = tmp_path / "vjepa_trunk.pt"
    torch.save({"trunk_state_dict": trunk_state_dict, "epoch": 1}, ssl_path)

    cfg = OmegaConf.create(
        {
            "model": {
                "name": "avanced_resnet50_tsm",
                "num_classes": 5,
                "pretrained": False,
                "shift_div": 8,
                "shift_place": "blockres",
                "dropout": 0.5,
                "head": "attn",  # exercise the head path too
                "head_num_heads": 4,
                "drop_path_rate": 0.1,
            },
            "dataset": {"num_frames": 4},
        }
    )
    sup_model = build_model(cfg)
    classifier_before = sup_model.classifier.weight.detach().clone()

    remapped = _ssl_trunk_to_supervised_keys(trunk_state_dict)
    missing, unexpected = sup_model.load_state_dict(remapped, strict=False)

    backbone_missing = [k for k in missing if k.startswith("backbone.")]
    assert backbone_missing == [], (
        f"backbone keys not covered by V-JEPA trunk: {backbone_missing[:4]}"
    )
    assert unexpected == [], f"unexpected SSL keys: {unexpected[:4]}"
    assert "classifier.weight" in missing
    assert any(k.startswith("attn_pool") for k in missing)
    torch.testing.assert_close(sup_model.classifier.weight, classifier_before)

    sup_state = sup_model.state_dict()
    sample_key = next(iter(remapped.keys()))
    torch.testing.assert_close(sup_state[sample_key], remapped[sample_key])
