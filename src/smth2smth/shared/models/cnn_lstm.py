"""CNN + LSTM: ResNet18 per-frame features fed into an LSTM over time.

Forward::

    Input:    (B, T, C, H, W)
    Frame CNN:(B*T, C, H, W) -> (B*T, 512)
    Reshape:  (B, T, 512)
    LSTM:     (B, T, hidden)
    Linear:   (B, num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchvision import models

from smth2smth.shared.models.registry import register_model

DEFAULT_LSTM_HIDDEN_SIZE: int = 512


class CNNLSTM(nn.Module):
    """ResNet18 frame encoder + single-layer LSTM + linear classifier.

    Args:
        num_classes: Number of output classes.
        pretrained: When ``True`` initializes the backbone from ImageNet weights.
        lstm_hidden_size: Hidden state size of the LSTM.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        lstm_hidden_size: int = DEFAULT_LSTM_HIDDEN_SIZE,
    ) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """Run the per-frame CNN, LSTM over time, then classify the last hidden state.

        Args:
            video_batch: Float tensor of shape ``(B, T, C, H, W)``.

        Returns:
            Logits tensor of shape ``(B, num_classes)``.
        """
        batch_size, num_frames, channels, height, width = video_batch.shape
        frames = video_batch.reshape(batch_size * num_frames, channels, height, width)

        frame_features = self.backbone(frames)
        frame_features = torch.flatten(frame_features, start_dim=1)

        sequence = frame_features.view(batch_size, num_frames, -1)

        lstm_out, _ = self.lstm(sequence)
        last_hidden = lstm_out[:, -1, :]

        return self.classifier(last_hidden)


@register_model("cnn_lstm")
def build_cnn_lstm(cfg: DictConfig) -> nn.Module:
    """Builder hook used by :func:`smth2smth.shared.models.registry.build_model`."""
    hidden = cfg.model.get("lstm_hidden_size", DEFAULT_LSTM_HIDDEN_SIZE)
    return CNNLSTM(
        num_classes=int(cfg.model.num_classes),
        pretrained=bool(cfg.model.pretrained),
        lstm_hidden_size=int(hidden),
    )
