"""End-to-end smoke test for ``model.init_from`` (SSL warm-start)."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from smth2smth.pipelines.train import _ssl_trunk_to_supervised_keys
from smth2smth.shared.models import build_model
from smth2smth.shared.models.dino_ssl import DinoModel


def test_ssl_trunk_loads_into_supervised_model_with_remap(tmp_path: Path) -> None:
    """Build a DinoModel, save its trunk-only state_dict to disk, then load
    it into a freshly-built ``AvancedResNet50TSM`` via the same remap +
    ``load_state_dict(strict=False)`` path the supervised pipeline uses.

    Asserts:
    * No backbone key is left at its random init (every backbone tensor is
      replaced by the SSL one).
    * The supervised model's classifier and (optional) attention pool keys
      are *not* touched by the load.
    """
    torch.manual_seed(0)
    student = DinoModel(out_dim=64, hidden_dim=32, bottleneck_dim=16, n_layers=3)

    trunk_state_dict = {
        k.removeprefix("trunk."): v
        for k, v in student.state_dict().items()
        if k.startswith("trunk.")
    }
    ssl_path = tmp_path / "ssl_trunk.pt"
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
            },
            "dataset": {"num_frames": 4},
        }
    )
    sup_model = build_model(cfg)
    classifier_before = sup_model.classifier.weight.detach().clone()

    remapped = _ssl_trunk_to_supervised_keys(trunk_state_dict)
    missing, unexpected = sup_model.load_state_dict(remapped, strict=False)

    backbone_missing = [k for k in missing if k.startswith("backbone.")]
    assert backbone_missing == []
    # ``unexpected`` should be empty too -- our remap targets only valid keys.
    assert unexpected == [], f"unexpected SSL keys: {unexpected[:4]}"
    # Classifier and attention pool keys are in ``missing`` (we didn't load
    # them) but were left untouched.
    assert "classifier.weight" in missing
    assert any(k.startswith("attn_pool") for k in missing)
    torch.testing.assert_close(sup_model.classifier.weight, classifier_before)

    # Spot-check: a backbone tensor in the supervised model should now equal
    # the corresponding SSL tensor (modulo the remap).
    sup_state = sup_model.state_dict()
    sample_key = next(iter(remapped.keys()))
    torch.testing.assert_close(sup_state[sample_key], remapped[sample_key])
