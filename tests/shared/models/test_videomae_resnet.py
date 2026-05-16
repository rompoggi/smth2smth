"""Smoke tests for VideoMAE ResNet-50 + TSM pretraining."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from smth2smth.pipelines.train import _ssl_trunk_to_supervised_keys
from smth2smth.shared.models import build_model
from smth2smth.shared.models.video_mae_resnet import (
    VideoMAEResNetPretrainModel,
    videomae_resnet_feature_loss,
)


def test_videomae_resnet_forward_and_loss() -> None:
    model = VideoMAEResNetPretrainModel(num_frames=4, img_size=224, mask_ratio=0.75)
    clips = torch.randn(2, 4, 3, 224, 224)
    pred, target, ids_keep, ids_mask = model(clips)
    assert pred.shape[0] == 2
    assert pred.shape == target.shape
    assert ids_keep.shape[1] + ids_mask.shape[1] == 4 * 7 * 7
    loss = videomae_resnet_feature_loss(pred, target)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_resnet_trunk_loads_into_avanced_resnet50_tsm() -> None:
    pretrain = VideoMAEResNetPretrainModel(num_frames=4, img_size=224)
    trunk_state = {k: v for k, v in pretrain.backbone.state_dict().items()}
    assert any(k.startswith("layer1.") for k in trunk_state)
    assert any("conv1.1." in k for k in trunk_state)

    cfg = OmegaConf.create(
        {
            "model": {
                "name": "avanced_resnet50_tsm",
                "num_classes": 33,
                "shift_div": 8,
                "shift_place": "blockres",
                "dropout": 0.5,
                "drop_path_rate": 0.1,
                "head": "attn",
                "head_num_heads": 4,
            },
            "dataset": {"num_frames": 4},
            "num_classes": 33,
        }
    )
    supervised = build_model(cfg)
    prefixed = _ssl_trunk_to_supervised_keys(trunk_state)
    missing, unexpected = supervised.load_state_dict(prefixed, strict=False)
    backbone_missing = [k for k in missing if k.startswith("backbone.")]
    assert len(unexpected) == 0
    assert len(backbone_missing) == 0
