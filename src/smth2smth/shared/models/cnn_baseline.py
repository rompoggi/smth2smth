"""CNN baseline: ResNet18 per-frame features + temporal average pooling.

Forward (conceptually)::

    Input:    (B, T, C, H, W)
    Reshape:  (B*T, C, H, W)
    Backbone: ResNet18 -> (B*T, 512)
    Reshape:  (B, T, 512)
    Mean:     (B, 512)
    Linear:   (B, num_classes)

Used by both Track A (``pretrained=False``) and Track B (``pretrained=True``)
in the professor baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchvision import models

from smth2smth.shared.models.registry import register_model


class CNNBaseline(nn.Module):
    """ResNet18 backbone + temporal average pooling + linear classifier.

    Args:
        num_classes: Number of output classes.
        pretrained: When ``True`` initializes the backbone from the
            ``IMAGENET1K_V1`` weights; otherwise random initialization.
    """

    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """Run the full per-frame CNN + temporal-mean head.

        Args:
            video_batch: Float tensor of shape ``(B, T, C, H, W)``.

        Returns:
            Logits tensor of shape ``(B, num_classes)``.
        """
        batch_size, num_frames, channels, height, width = video_batch.shape

        frames = video_batch.reshape(batch_size * num_frames, channels, height, width)

        frame_features = self.backbone(frames)
        frame_features = torch.flatten(frame_features, start_dim=1)

        sequence_features = frame_features.view(batch_size, num_frames, -1)
        pooled_features = sequence_features.mean(dim=1)

        return self.classifier(pooled_features)


@register_model("cnn_baseline")
def build_cnn_baseline(cfg: DictConfig) -> nn.Module:
    """Builder hook used by :func:`smth2smth.shared.models.registry.build_model`."""
    return CNNBaseline(
        num_classes=int(cfg.model.num_classes),
        pretrained=bool(cfg.model.pretrained),
    )
