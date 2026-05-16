"""VideoMAE-style masked modeling for TSM ResNet-50 (Track A, Phase-2 backbone).

Adapts the VideoMAE recipe (Tong et al., NeurIPS 2022) to the same
:class:`~smth2smth.shared.models.avanced_resnet50_tsm.AvancedResNet50TSM`
backbone used in ``track_a_phase2``:

  * Tube masking on a 7×7 spatial grid across T frames (75% masked by default).
  * Masked input regions are zeroed before the ResNet+TSM forward pass.
  * Encoder: TSM ResNet-50 through ``layer4`` → per-cell 2048-d features.
  * Lightweight transformer on *visible* spatiotemporal tokens.
  * Decoder: predicts normalised features at masked cells (feature MSE).

The saved ``trunk_state_dict`` uses **plain ResNet+TSM keys** (no ``encoder.``
prefix), identical to DINO/V-JEPA SSL checkpoints, so
``model.init_from`` on ``avanced_resnet50_tsm`` works via
:func:`~smth2smth.pipelines.train._ssl_trunk_to_supervised_keys`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from smth2smth.shared.models.avanced_resnet50_tsm import (
    _make_temporal_shift_resnet,
    _validate_residual_paths,
)
from smth2smth.shared.models.video_mae import TransformerBlock, make_tube_mask


def build_resnet50_tsm_backbone(
    num_frames: int,
    shift_div: int = 8,
    shift_place: str = "blockres",
) -> nn.Module:
    """TSM-wrapped ResNet-50 trunk matching ``AvancedResNet50TSM.backbone``."""
    backbone = models.resnet50(weights=None)
    backbone.fc = nn.Identity()
    backbone = _make_temporal_shift_resnet(
        backbone=backbone,
        n_segment=num_frames,
        shift_div=shift_div,
        shift_place=shift_place,
    )
    _validate_residual_paths(backbone)
    return backbone


def forward_resnet_spatial_features(
    backbone: nn.Module,
    clips: torch.Tensor,
) -> torch.Tensor:
    """Run the trunk through ``layer4`` and return spatiotemporal feature tokens.

    Args:
        backbone: TSM ResNet-50 without ``fc``.
        clips: ``(B, T, C, H, W)``.

    Returns:
        ``(B, T, D, H', W')`` with ``D=2048`` and ``H'=W'=7`` for 224×224 input.
    """
    batch_size, num_frames, channels, height, width = clips.shape
    frames = clips.reshape(batch_size * num_frames, channels, height, width)
    x = backbone.conv1(frames)
    x = backbone.bn1(x)
    x = backbone.relu(x)
    x = backbone.maxpool(x)
    x = backbone.layer1(x)
    x = backbone.layer2(x)
    x = backbone.layer3(x)
    x = backbone.layer4(x)
    _, feat_dim, fh, fw = x.shape
    return x.view(batch_size, num_frames, feat_dim, fh, fw)


def apply_input_tube_mask(
    clips: torch.Tensor,
    ids_mask: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    """Zero out input cells corresponding to masked spatiotemporal tokens.

    Each token covers a ``H/grid_h × W/grid_w`` rectangle on every frame.
    ``ids_mask`` indexes into the flattened ``T × grid_h × grid_w`` grid.
    """
    b, t, c, h, w = clips.shape
    cell_h = h // grid_h
    cell_w = w // grid_w
    n_spatial = grid_h * grid_w
    out = clips.clone()
    for bi in range(b):
        for idx in ids_mask[bi].tolist():
            ti = idx // n_spatial
            si = idx % n_spatial
            row = si // grid_w
            col = si % grid_w
            r0, r1 = row * cell_h, (row + 1) * cell_h
            c0, c1 = col * cell_w, (col + 1) * cell_w
            out[bi, ti, :, r0:r1, c0:c1] = 0.0
    return out


class FeatureMAEDecoder(nn.Module):
    """Scatter visible tokens + mask tokens, decode, predict feature vectors."""

    def __init__(
        self,
        encoder_dim: int,
        num_tokens: int,
        decoder_dim: int = 512,
        decoder_depth: int = 4,
        decoder_heads: int = 8,
        target_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_tokens, decoder_dim))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(decoder_dim, decoder_heads, mlp_ratio=4.0, drop_path=0.0)
                for _ in range(decoder_depth)
            ]
        )
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, target_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        visible_tokens: torch.Tensor,
        ids_keep: torch.Tensor,
        ids_mask: torch.Tensor,
    ) -> torch.Tensor:
        b = visible_tokens.shape[0]
        n_visible = ids_keep.shape[1]
        n_masked = ids_mask.shape[1]
        n_tokens = n_visible + n_masked

        tokens = self.proj(visible_tokens)
        full = self.mask_token.expand(b, n_tokens, -1).clone()
        # AMP can yield bf16 activations while params stay fp32; scatter requires matching dtypes.
        if tokens.dtype != full.dtype:
            tokens = tokens.to(dtype=full.dtype)
        idx_v = ids_keep.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        full.scatter_(1, idx_v, tokens)
        full = full + self.decoder_pos_embed

        for block in self.blocks:
            full = block(full)
        full = self.norm(full)

        idx_m = ids_mask.unsqueeze(-1).expand(-1, -1, full.shape[-1])
        masked_out = torch.gather(full, 1, idx_m)
        return self.head(masked_out)


class VideoMAEResNetPretrainModel(nn.Module):
    """Masked feature reconstruction pretraining for TSM ResNet-50."""

    def __init__(
        self,
        num_frames: int = 4,
        img_size: int = 224,
        shift_div: int = 8,
        shift_place: str = "blockres",
        mask_ratio: float = 0.75,
        encoder_depth: int = 2,
        encoder_heads: int = 8,
        feature_dim: int = 2048,
    ) -> None:
        super().__init__()
        if img_size % 32 != 0:
            raise ValueError(f"img_size={img_size} must be divisible by 32 for a 7×7 feature grid.")
        self.num_frames = num_frames
        self.mask_ratio = mask_ratio
        self.feature_dim = feature_dim
        self.grid_h = img_size // 32
        self.grid_w = img_size // 32
        self.num_tokens = num_frames * self.grid_h * self.grid_w

        self.backbone = build_resnet50_tsm_backbone(
            num_frames=num_frames,
            shift_div=shift_div,
            shift_place=shift_place,
        )

        enc_dpr = [0.0] * max(1, encoder_depth)
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    feature_dim,
                    encoder_heads,
                    mlp_ratio=4.0,
                    drop_path=enc_dpr[i],
                )
                for i in range(encoder_depth)
            ]
        )
        self.encoder_norm = nn.LayerNorm(feature_dim)

        self.decoder = FeatureMAEDecoder(
            encoder_dim=feature_dim,
            num_tokens=self.num_tokens,
            target_dim=feature_dim,
        )

    def _tokens_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """``(B, T, D, H, W)`` → ``(B, T*H*W, D)``."""
        b, t, d, fh, fw = features.shape
        return features.permute(0, 1, 3, 4, 2).reshape(b, t * fh * fw, d)

    def forward(
        self,
        clips: torch.Tensor,
        ids_keep: torch.Tensor | None = None,
        ids_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            pred: masked feature predictions ``(B, n_masked, D)``
            target: ground-truth features at masked positions ``(B, n_masked, D)``
            ids_keep, ids_mask
        """
        if ids_keep is None:
            ids_keep, ids_mask = make_tube_mask(
                clips.shape[0],
                self.num_frames,
                self.grid_h,
                self.grid_w,
                self.mask_ratio,
                device=clips.device,
            )

        masked_clips = apply_input_tube_mask(clips, ids_mask, self.grid_h, self.grid_w)
        spatial = forward_resnet_spatial_features(self.backbone, masked_clips)
        tokens = self._tokens_from_features(spatial)

        visible = torch.gather(
            tokens,
            1,
            ids_keep.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
        )
        for block in self.encoder_blocks:
            visible = block(visible)
        visible = self.encoder_norm(visible)

        pred = self.decoder(visible, ids_keep, ids_mask)

        idx_m = ids_mask.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        target = torch.gather(tokens, 1, idx_m)
        return pred, target, ids_keep, ids_mask


def videomae_resnet_feature_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    norm_feat: bool = True,
) -> torch.Tensor:
    """MSE on (optionally normalised) feature vectors at masked positions."""
    if norm_feat:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True, unbiased=False)
        target = (target - mean) / (var + 1e-6).sqrt()
    return ((pred - target) ** 2).mean()


__all__ = [
    "FeatureMAEDecoder",
    "VideoMAEResNetPretrainModel",
    "apply_input_tube_mask",
    "build_resnet50_tsm_backbone",
    "forward_resnet_spatial_features",
    "videomae_resnet_feature_loss",
]
