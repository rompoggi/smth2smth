"""Forward passes on the holdout val set with optional TTA."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from smth2smth.pipelines.submit import (
    _build_flip_class_permutation,
    _build_logit_adjustment,
    _build_untrained_mask,
    _logits_for_batch,
    _rescale_video,
)
from smth2smth.pipelines.train import _resolve_device
from smth2smth.shared.data import VideoFrameDataset, build_transforms
from smth2smth.shared.io.checkpoints import cfg_from_checkpoint, load_checkpoint
from smth2smth.shared.models import build_model
from smth2smth.shared.models.video_mae import interpolate_pos_embed
from smth2smth.shared.utils.splits import VideoSample as VS


class TtaMode(str, Enum):  # noqa: UP042 - keep str-mixin for JSON value round-trip
    """How to run inference when caching logits."""

    NONE = "none"
    CHAMPION = "champion"
    OFFICIAL_2X3 = "official_2x3"
    DENSE_2X3 = "official_2x3"  # alias


def _tta_settings_from_checkpoint(
    saved_cfg: DictConfig, mode: TtaMode
) -> tuple[bool, bool, list[float], int, int, bool]:
    """Return TTA flags for :func:`collect_holdout_logits`."""
    if mode == TtaMode.NONE:
        return False, False, [1.0], 1, 1, False
    if mode == TtaMode.CHAMPION:
        # Sweep winner: 3-scale + flip (scales875_flip ≡ scales3_flip on ViT 224).
        scales_cfg = saved_cfg.training.get("tta_scales", None)
        scales = [float(s) for s in scales_cfg] if scales_cfg else [0.857, 1.0, 1.143]
        flip = bool(saved_cfg.training.get("tta_flip", True))
        return True, flip, scales, 1, 1, False
    if mode in (TtaMode.OFFICIAL_2X3, TtaMode.DENSE_2X3):
        return False, False, [1.0], 2, 3, False
    raise ValueError(f"Unknown TtaMode: {mode}")


def _videomae_encoder_features(encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run ``VideoMAEEncoder`` with ``pos_embed`` resized to match ``x`` grid."""
    patch_embed = encoder.patch_embed
    patch_size = int(patch_embed.proj.kernel_size[2])
    tube_t = int(patch_embed.proj.kernel_size[0])
    src_frames = int(patch_embed.n_t * tube_t)
    src_img = int(patch_embed.n_h * patch_size)
    _b, t, _c, h, w = x.shape
    tokens = patch_embed(x)
    pe = encoder.pos_embed
    if tokens.shape[1] != pe.shape[1]:
        pe = interpolate_pos_embed(
            pe,
            src_num_frames=src_frames,
            src_img_size=src_img,
            dst_num_frames=int(t),
            dst_img_size=int(h),
            tube_t=tube_t,
            patch_size=patch_size,
        )
    tokens = tokens + pe.to(dtype=tokens.dtype)
    if encoder.residual_variant == "prenorm":
        for block in encoder.blocks:
            tokens = block(tokens)
        return encoder.norm(tokens)
    h_state = tokens.unsqueeze(1).expand(-1, encoder.hc_n, -1, -1).contiguous()
    for block in encoder.blocks:
        h_state = block(h_state)
    alpha_out = encoder.alpha_out.to(h_state.dtype)
    collapsed = torch.einsum("n,bntd->btd", alpha_out, h_state)
    return encoder.norm(collapsed)


@torch.no_grad()
def _videomae_logits_batch(
    model: nn.Module,
    video_batch: torch.Tensor,
    *,
    untrained_mask: torch.Tensor | None,
    logit_adjust: torch.Tensor | None,
    amp_infer: bool,
) -> torch.Tensor:
    """Classifier logits for VideoMAE with spatial size implied by ``video_batch``."""
    features = _videomae_encoder_features(model.encoder, video_batch)
    if model.attn_pool is not None:
        pooled = model.attn_pool(features)
    elif getattr(model, "pool_head", None) is not None:
        pooled = model.pool_head(features)
    else:
        pooled = features.mean(dim=1)
    logits = model.classifier(model.dropout(pooled))
    if untrained_mask is not None:
        logits = logits + untrained_mask
    if logit_adjust is not None:
        logits = logits + logit_adjust
    return logits


@torch.no_grad()
def _logits_for_model_batch(
    model: nn.Module,
    video_batch: torch.Tensor,
    *,
    untrained_mask: torch.Tensor | None,
    logit_adjust: torch.Tensor | None,
    amp_infer: bool,
    patch_size: int | None,
) -> torch.Tensor:
    """Forward one batch; VideoMAE uses interpolated ``pos_embed`` when needed."""
    if patch_size is not None:
        if amp_infer and video_batch.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return _videomae_logits_batch(
                    model,
                    video_batch,
                    untrained_mask=untrained_mask,
                    logit_adjust=logit_adjust,
                    amp_infer=False,
                )
        return _videomae_logits_batch(
            model,
            video_batch,
            untrained_mask=untrained_mask,
            logit_adjust=logit_adjust,
            amp_infer=amp_infer,
        )
    return _logits_for_batch(
        model,
        video_batch,
        untrained_mask,
        logit_adjust=logit_adjust,
        amp_infer=amp_infer,
    )


@torch.no_grad()
def _forward_batch_logits(
    model: nn.Module,
    video_batch: torch.Tensor,
    *,
    device: torch.device,
    untrained_mask: torch.Tensor | None,
    tta_enabled: bool,
    tta_flip: bool,
    flip_perm: torch.Tensor | None,
    tta_scales: list[float],
    logit_adjust: torch.Tensor | None,
    amp_infer: bool,
    patch_size: int | None,
) -> torch.Tensor:
    """Per-sample logits ``(B, C)``, TTA-averaged when enabled."""
    video_batch = video_batch.to(device, non_blocking=True)
    if not tta_enabled:
        return _logits_for_model_batch(
            model,
            video_batch,
            untrained_mask=untrained_mask,
            logit_adjust=logit_adjust,
            amp_infer=amp_infer,
            patch_size=patch_size,
        )

    logits_total: torch.Tensor | None = None
    n_views = 0
    for scale in tta_scales:
        scaled = _rescale_video(video_batch, scale, patch_size=patch_size)
        scaled_logits = _logits_for_model_batch(
            model,
            scaled,
            untrained_mask=untrained_mask,
            logit_adjust=logit_adjust,
            amp_infer=amp_infer,
            patch_size=patch_size,
        )
        logits_total = scaled_logits if logits_total is None else logits_total + scaled_logits
        n_views += 1
        if tta_flip:
            flipped = torch.flip(scaled, dims=[-1])
            flipped_logits = _logits_for_model_batch(
                model,
                flipped,
                untrained_mask=untrained_mask,
                logit_adjust=logit_adjust,
                amp_infer=amp_infer,
                patch_size=patch_size,
            )
            if flip_perm is not None:
                flipped_logits = flipped_logits.index_select(dim=1, index=flip_perm)
            logits_total = logits_total + flipped_logits
            n_views += 1
    assert logits_total is not None
    return logits_total / float(n_views)


@torch.no_grad()
def collect_logits_for_videos(
    checkpoint_path: Path,
    samples: list[VS],
    *,
    data_root: Path,
    train_dir: Path,
    tta_mode: TtaMode,
    batch_size: int = 8,
    num_workers: int = 4,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Run inference on ``samples`` and return logits ``(N, num_classes)`` on CPU."""
    checkpoint_path = checkpoint_path.resolve()
    device = device or _resolve_device("cuda")
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    saved_cfg = cfg_from_checkpoint(checkpoint)

    model = build_model(saved_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    num_classes = int(saved_cfg.model.num_classes)
    extra = checkpoint.get("extra") or {}
    untrained_mask = _build_untrained_mask(
        trained_class_indices=extra.get("trained_class_indices"),
        num_classes=num_classes,
        device=device,
    )

    use_imagenet_norm = bool(saved_cfg.model.get("pretrained", False))
    augment_cfg = saved_cfg.get("augment") if hasattr(saved_cfg, "get") else None
    eval_transform = build_transforms(
        image_size=int(saved_cfg.dataset.image_size),
        is_training=False,
        use_imagenet_norm=use_imagenet_norm,
        augment=augment_cfg,
    )
    num_frames = int(saved_cfg.dataset.num_frames)

    tta_enabled, tta_flip, tta_scales, num_segment, num_crop, flip_dense = (
        _tta_settings_from_checkpoint(saved_cfg, tta_mode)
    )
    if tta_mode in (TtaMode.OFFICIAL_2X3, TtaMode.DENSE_2X3):
        logits, _probs = _collect_holdout_dense_tta(
            model=model,
            holdout=samples,
            val_root=data_root,
            train_dir=train_dir,
            saved_cfg=saved_cfg,
            num_frames=num_frames,
            image_size=int(saved_cfg.dataset.image_size),
            use_imagenet_norm=use_imagenet_norm,
            untrained_mask=untrained_mask,
            num_segment=num_segment,
            num_crop=num_crop,
            flip_tta=flip_dense,
            device=device,
            amp_infer=bool(saved_cfg.training.get("amp", False)) and device.type == "cuda",
        )
        return logits

    dataset = VideoFrameDataset(
        root_dir=data_root,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=list(samples),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    flip_perm = None
    if tta_flip:
        flip_perm = _build_flip_class_permutation(
            train_dir=train_dir,
            num_classes=num_classes,
            device=device,
        )
    tau = float(saved_cfg.training.get("tta_logit_adjust", 0.0)) if tta_enabled else 0.0
    logit_adjust = (
        _build_logit_adjustment(train_dir, num_classes, tau, device) if tau > 0.0 else None
    )
    patch_size = (
        int(saved_cfg.model.patch_size)
        if str(saved_cfg.model.get("name", "")) == "video_mae_vit"
        else None
    )
    amp_infer = bool(saved_cfg.training.get("amp", False)) and device.type == "cuda"

    chunks: list[torch.Tensor] = []
    for video_batch, _labels in loader:
        logits = _forward_batch_logits(
            model,
            video_batch,
            device=device,
            untrained_mask=untrained_mask,
            tta_enabled=tta_enabled,
            tta_flip=tta_flip,
            flip_perm=flip_perm,
            tta_scales=tta_scales,
            logit_adjust=logit_adjust,
            amp_infer=amp_infer,
            patch_size=patch_size,
        )
        chunks.append(logits.cpu())
    if not chunks:
        return torch.zeros(0, num_classes)
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def collect_holdout_logits(
    checkpoint_path: Path,
    holdout: list[VS],
    *,
    val_root: Path,
    train_dir: Path,
    tta_mode: TtaMode,
    batch_size: int = 8,
    num_workers: int = 4,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Run inference on holdout clips and return logits ``(N, C)``."""
    return collect_logits_for_videos(
        checkpoint_path,
        holdout,
        data_root=val_root,
        train_dir=train_dir,
        tta_mode=tta_mode,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )


@torch.no_grad()
def _collect_holdout_dense_tta(
    *,
    model: nn.Module,
    holdout: list[VS],
    val_root: Path,
    train_dir: Path,
    saved_cfg: DictConfig,
    num_frames: int,
    image_size: int,
    use_imagenet_norm: bool,
    untrained_mask: torch.Tensor | None,
    num_segment: int,
    num_crop: int,
    flip_tta: bool,
    device: torch.device,
    amp_infer: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-clip official 2×3 TTA: mean logits and mean softmax (submit-style)."""
    from smth2smth.pipelines.submit import (
        _eval_normalize,
        _load_tta_video_views,
    )

    num_classes = int(saved_cfg.model.num_classes)
    normalize = _eval_normalize(use_imagenet_norm)
    flip_perm = _build_flip_class_permutation(train_dir, num_classes, device) if flip_tta else None
    micro_batch = max(1, int(saved_cfg.training.get("batch_size", 8)))

    all_logits: list[torch.Tensor] = []
    all_probs: list[torch.Tensor] = []
    for video_dir, _label in holdout:
        views = _load_tta_video_views(
            video_dir,
            num_frames=num_frames,
            image_size=image_size,
            normalize=normalize,
            num_segment=num_segment,
            num_crop=num_crop,
            flip_tta=flip_tta,
        )
        logits_total: torch.Tensor | None = None
        probs_total: torch.Tensor | None = None
        n_views = 0
        for start in range(0, len(views), micro_batch):
            batch_views = views[start : start + micro_batch]
            video_batch = torch.stack(batch_views, dim=0).to(device, non_blocking=True)
            logits = _logits_for_batch(model, video_batch, untrained_mask, amp_infer=amp_infer)
            batch_probs = torch.softmax(logits, dim=1)
            for local_i, row in enumerate(logits):
                view_logits = row
                view_probs = batch_probs[local_i]
                global_view = start + local_i
                if flip_tta and flip_perm is not None and global_view % 2 == 1:
                    view_logits = view_logits.index_select(dim=0, index=flip_perm)
                    view_probs = view_probs.index_select(dim=0, index=flip_perm)
                logits_total = view_logits if logits_total is None else logits_total + view_logits
                probs_total = view_probs if probs_total is None else probs_total + view_probs
                n_views += 1
        assert logits_total is not None and probs_total is not None and n_views > 0
        all_logits.append((logits_total / float(n_views)).cpu())
        all_probs.append((probs_total / float(n_views)).cpu())

    if not all_logits:
        z = torch.zeros(0, num_classes)
        return z, z
    return torch.stack(all_logits, dim=0), torch.stack(all_probs, dim=0)


def collect_logits_and_probs_for_videos(
    checkpoint_path: Path,
    samples: list[VS],
    *,
    data_root: Path,
    train_dir: Path,
    tta_mode: TtaMode,
    batch_size: int = 8,
    num_workers: int = 4,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return ``(logits, probs)``; ``probs`` is ``None`` unless ``tta_mode`` is official 2×3."""
    if tta_mode in (TtaMode.OFFICIAL_2X3, TtaMode.DENSE_2X3):
        checkpoint_path = checkpoint_path.resolve()
        device = device or _resolve_device("cuda")
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)
        saved_cfg = cfg_from_checkpoint(checkpoint)
        model = build_model(saved_cfg).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        num_classes = int(saved_cfg.model.num_classes)
        extra = checkpoint.get("extra") or {}
        untrained_mask = _build_untrained_mask(
            trained_class_indices=extra.get("trained_class_indices"),
            num_classes=num_classes,
            device=device,
        )
        use_imagenet_norm = bool(saved_cfg.model.get("pretrained", False))
        holdout = samples
        logits, probs = _collect_holdout_dense_tta(
            model=model,
            holdout=holdout,
            val_root=data_root,
            train_dir=train_dir,
            saved_cfg=saved_cfg,
            num_frames=int(saved_cfg.dataset.num_frames),
            image_size=int(saved_cfg.dataset.image_size),
            use_imagenet_norm=use_imagenet_norm,
            untrained_mask=untrained_mask,
            num_segment=2,
            num_crop=3,
            flip_tta=False,
            device=device,
            amp_infer=bool(saved_cfg.training.get("amp", False)) and device.type == "cuda",
        )
        return logits, probs
    logits = collect_logits_for_videos(
        checkpoint_path,
        samples,
        data_root=data_root,
        train_dir=train_dir,
        tta_mode=tta_mode,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    return logits, None
