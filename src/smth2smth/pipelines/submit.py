"""Submission pipeline.

Loads a trained checkpoint and runs inference on the test split, writing
``video_name,predicted_class`` to the configured output CSV.

Run from the repo root::

    PYTHONPATH=src uv run python -m smth2smth.pipelines.submit \\
        training.checkpoint_path=best_model.pt
"""

from __future__ import annotations

import math
from pathlib import Path

import hydra
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from smth2smth.pipelines.train import CONFIGS_DIR, _resolve_device
from smth2smth.shared.data import VideoFrameDataset, build_transforms, collect_video_samples
from smth2smth.shared.data import transforms as transform_constants
from smth2smth.shared.data.video_dataset import (
    _list_frame_paths,
    parse_class_index,
    pick_segment_frame_indices,
)
from torchvision.transforms import Normalize
from torchvision.transforms import functional as TF
from smth2smth.shared.io.checkpoints import cfg_from_checkpoint, load_checkpoint
from smth2smth.shared.io.submission import (
    discover_all_test_videos,
    load_manifest_video_names,
    resolve_video_dirs,
    write_submission_csv,
)
from smth2smth.shared.models import build_model
from smth2smth.shared.models.video_mae import interpolate_pos_embed
from smth2smth.shared.utils import class_counts, set_seed


def _resolve_test_videos(
    test_root: Path, manifest_path: Path | None
) -> tuple[list[str], list[Path]]:
    """Pick test video order from manifest if provided, otherwise from disk."""
    if manifest_path is not None:
        names = load_manifest_video_names(manifest_path)
        dirs = resolve_video_dirs(test_root, names)
        return names, dirs
    return discover_all_test_videos(test_root)


def run(cfg: DictConfig) -> Path:
    """Generate the submission CSV from a checkpoint.

    Args:
        cfg: Hydra configuration. Must have ``cfg.training.checkpoint_path``,
            ``cfg.dataset.test_dir`` and ``cfg.dataset.submission_output``.

    Returns:
        Path to the written CSV.
    """
    print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.seed))
    device = _resolve_device(str(cfg.training.device))

    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    saved_cfg = cfg_from_checkpoint(checkpoint)

    model = build_model(saved_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    use_imagenet_norm = (
        bool(saved_cfg.model.get("pretrained", False))
        if hasattr(saved_cfg.model, "get")
        else bool(getattr(saved_cfg.model, "pretrained", False))
    )
    augment_cfg = saved_cfg.get("augment") if hasattr(saved_cfg, "get") else None
    eval_transform = build_transforms(
        image_size=int(cfg.dataset.image_size),
        is_training=False,
        use_imagenet_norm=use_imagenet_norm,
        augment=augment_cfg,
    )
    num_frames = (
        int(saved_cfg.dataset.num_frames) if "dataset" in saved_cfg else int(cfg.dataset.num_frames)
    )

    test_root = Path(cfg.dataset.test_dir).resolve()
    manifest_cfg = cfg.dataset.get("test_manifest")
    manifest_path = Path(str(manifest_cfg)).resolve() if manifest_cfg else None

    print(f"Indexing video folders under: {test_root}")
    video_names, video_dirs = _resolve_test_videos(test_root, manifest_path)
    print(f"Found {len(video_names)} test videos.")

    test_cfg = cfg.get("test")
    num_segment = int(test_cfg.num_segment) if test_cfg is not None else 1
    num_crop = int(test_cfg.num_crop) if test_cfg is not None else 1
    flip_tta = bool(test_cfg.flip_tta) if test_cfg is not None else False
    dense_tta = num_segment > 1 or num_crop > 1 or flip_tta

    # If the checkpoint recorded which class indices actually had training
    # samples (added in Phase 1), mask the never-trained logits to ``-inf``
    # before argmax. Older checkpoints without this metadata fall back to
    # plain argmax, exactly as before.
    extra = checkpoint.get("extra") or {}
    trained_class_indices = extra.get("trained_class_indices")
    untrained_mask = _build_untrained_mask(
        trained_class_indices=trained_class_indices,
        num_classes=int(cfg.num_classes),
        device=device,
    )

    train_dir = Path(str(cfg.dataset.train_dir)).resolve()
    flip_perm: torch.Tensor | None = None
    if flip_tta or bool(cfg.training.get("tta_flip", True)):
        flip_perm = _build_flip_class_permutation(
            train_dir=train_dir,
            num_classes=int(cfg.num_classes),
            device=device,
        )

    tta_enabled = bool(cfg.training.get("tta", False))
    tta_flip_cfg = bool(cfg.training.get("tta_flip", True))

    # Multi-scale TTA. Default ``[1.0]`` is a no-op (single forward at the
    # training resolution). Adding e.g. ``[0.875, 1.0, 1.125]`` evaluates the
    # clip at three input sizes (centred around the trained ``image_size``)
    # and averages the softmaxes. ResNet-50 + GAP is fully convolutional so
    # this is well-defined; the cost is a linear-in-len(scales) forward pass.
    tta_scales_cfg = cfg.training.get("tta_scales", None) if tta_enabled else None
    tta_scales: list[float] = (
        [float(s) for s in tta_scales_cfg] if tta_scales_cfg else [1.0]
    )

    patch_size: int | None = None
    model_name = str(saved_cfg.model.get("name", "")) if hasattr(saved_cfg, "model") else ""
    if model_name == "video_mae_vit":
        patch_size = int(saved_cfg.model.get("patch_size", 16))
        base_side = int(saved_cfg.dataset.get("image_size", cfg.dataset.image_size))
        for scale in tta_scales:
            side = _round_spatial_to_patch_multiple(
                max(patch_size, int(round(base_side * float(scale)))), patch_size
            )
            if side % patch_size != 0:
                raise SystemExit(
                    f"TTA scale {scale} yields side {side}, not divisible by patch_size={patch_size}."
                )
            print(f"[tta] ViT scale {scale:g} -> {side}x{side} (patch_size={patch_size})")

    # Logit adjustment for long-tailed inference (Menon et al. 2021).
    # ``tta_logit_adjust > 0`` subtracts ``tau * log(p_c)`` from each logit,
    # where ``p_c`` is the empirical class frequency in the training folder.
    # Cheap and complementary to class-balanced training. Off by default.
    tau = float(cfg.training.get("tta_logit_adjust", 0.0)) if tta_enabled else 0.0
    logit_adjust: torch.Tensor | None = None
    if tau > 0.0:
        logit_adjust = _build_logit_adjustment(
            train_dir=train_dir,
            num_classes=int(cfg.num_classes),
            tau=tau,
            device=device,
        )

    amp_infer = bool(cfg.training.get("amp", False)) and device.type == "cuda"
    if amp_infer:
        print("[submit] inference autocast fp16 enabled (matches training.amp).")

    image_size = int(cfg.dataset.image_size)

    if dense_tta:
        n_views = num_segment * num_crop * (2 if flip_tta else 1)
        print(
            f"[test-tta] num_segment={num_segment} num_crop={num_crop} "
            f"flip_tta={flip_tta} -> {n_views} views/video"
        )
        predictions = _predict_dense_tta(
            model=model,
            video_dirs=video_dirs,
            num_frames=num_frames,
            image_size=image_size,
            use_imagenet_norm=use_imagenet_norm,
            device=device,
            untrained_mask=untrained_mask,
            flip_perm=flip_perm if flip_tta else None,
            logit_adjust=logit_adjust,
            amp_infer=amp_infer,
            num_segment=num_segment,
            num_crop=num_crop,
            flip_tta=flip_tta,
            micro_batch=int(cfg.training.batch_size),
            patch_size=patch_size,
        )
    else:
        sample_list = [(p, 0) for p in video_dirs]
        dataset = VideoFrameDataset(
            root_dir=test_root,
            num_frames=num_frames,
            transform=eval_transform,
            sample_list=sample_list,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(cfg.training.batch_size),
            shuffle=False,
            num_workers=int(cfg.training.num_workers),
            pin_memory=(device.type == "cuda"),
        )
        predictions = _predict(
            model=model,
            loader=loader,
            device=device,
            untrained_mask=untrained_mask,
            tta_enabled=tta_enabled,
            tta_flip=tta_flip_cfg,
            flip_perm=flip_perm if tta_enabled and tta_flip_cfg else None,
            tta_scales=tta_scales,
            logit_adjust=logit_adjust if tta_enabled else None,
            amp_infer=amp_infer,
            patch_size=patch_size,
        )
    if len(predictions) != len(video_names):
        raise RuntimeError(f"Prediction count {len(predictions)} != video count {len(video_names)}")

    output_path = Path(cfg.dataset.submission_output).resolve()
    csv_path = write_submission_csv(output_path, video_names, predictions)
    print(f"Wrote {len(predictions)} rows to {csv_path}")
    return csv_path


def _build_untrained_mask(
    trained_class_indices: list[int] | None,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Return a ``(num_classes,)`` additive mask (0 for trained classes, -inf
    for never-trained ones), or ``None`` if no masking is needed.

    Args:
        trained_class_indices: Class indices observed in the training set when
            the checkpoint was produced. ``None`` (legacy checkpoints) ⇒ no
            masking. Identical to ``range(num_classes)`` ⇒ no masking.
        num_classes: Width of the model's output head.
        device: Device the mask should live on (matches the logits' device).
    """
    if not trained_class_indices:
        return None
    trained = {int(i) for i in trained_class_indices if 0 <= int(i) < num_classes}
    if len(trained) >= num_classes:
        return None
    mask = torch.zeros(num_classes, dtype=torch.float32, device=device)
    untrained = [c for c in range(num_classes) if c not in trained]
    mask[untrained] = -math.inf
    print(f"[submit] masking {len(untrained)} never-trained class index/indices: {untrained}")
    return mask


def _build_flip_class_permutation(
    train_dir: Path,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Permutation aligning logits of an h-flipped clip to the original frame.

    For most action classes a horizontal flip preserves the label, so the
    permutation maps ``c -> c``. Some classes have left/right semantics
    (e.g. ``"Pulling something from left to right"`` vs. ``"... right to left"``)
    and a flipped clip should predict the *paired* class instead. We auto-derive
    these pairs from the class folder names: if class A's name contains
    ``"from_left_to_right"`` and B's name contains ``"from_right_to_left"`` and
    the rest of the names match, we pair them.

    Returns:
        A 1-D ``LongTensor`` of length ``num_classes`` such that
        ``softmax_orig[c] ~= softmax_flipped[perm[c]]``. Returns ``None`` if the
        train folder is not readable (no remap will be applied; flipped softmax
        is averaged in directly).
    """
    if not train_dir.is_dir():
        print(
            f"[tta] flip-pair remap disabled: train_dir not found ({train_dir}); "
            "flipped softmax will be averaged with no class permutation."
        )
        return None
    import re

    class_dirs = sorted(p for p in train_dir.iterdir() if p.is_dir())
    # Strip a leading ``\d+_`` prefix so pairs like ``018_..._left_to_right`` and
    # ``019_..._right_to_left`` match -- only the action name needs to align.
    stem_by_idx: dict[int, str] = {}
    for d in class_dirs:
        idx = parse_class_index(d.name)
        if idx is None or not (0 <= idx < num_classes):
            continue
        stem_by_idx[idx] = re.sub(r"^\d+_", "", d.name)
    perm = list(range(num_classes))
    paired: list[tuple[int, int]] = []
    LR = "from_left_to_right"
    RL = "from_right_to_left"
    for idx, stem in stem_by_idx.items():
        if LR in stem:
            mirror_stem = stem.replace(LR, RL)
            for other_idx, other_stem in stem_by_idx.items():
                if other_stem == mirror_stem:
                    perm[idx] = other_idx
                    perm[other_idx] = idx
                    paired.append((idx, other_idx))
                    break
    if paired:
        print(f"[tta] flip-pair remap: {paired} (others identity).")
    else:
        print("[tta] no left/right class pair detected; flip remap is identity.")
    return torch.tensor(perm, dtype=torch.long, device=device)


def _eval_normalize(use_imagenet_norm: bool) -> Normalize:
    """Return the eval normalization transform matching :func:`build_transforms`."""
    if use_imagenet_norm:
        return Normalize(
            mean=transform_constants._IMAGENET_MEAN,
            std=transform_constants._IMAGENET_STD,
        )
    return Normalize(
        mean=transform_constants._SYMMETRIC_MEAN,
        std=transform_constants._SYMMETRIC_STD,
    )


def _crop_x_offsets(num_crop: int, crop_size: int, resized_w: int) -> list[int]:
    """Horizontal crop offsets for ``num_crop`` evenly spaced ``crop_size`` windows."""
    if num_crop <= 1:
        return [max(0, (resized_w - crop_size) // 2)]
    max_off = max(0, resized_w - crop_size)
    return [int(round(i * max_off / (num_crop - 1))) for i in range(num_crop)]


def _load_tta_video_views(
    video_dir: Path,
    *,
    num_frames: int,
    image_size: int,
    normalize: Normalize,
    num_segment: int,
    num_crop: int,
    flip_tta: bool,
) -> list[torch.Tensor]:
    """Build ``(T, C, H, W)`` tensors for each dense-TTA view of one video."""
    frame_paths = _list_frame_paths(video_dir)
    num_available = len(frame_paths)
    if num_available == 0:
        raise RuntimeError(f"No frames under {video_dir}")

    resized_h = image_size
    resized_w = image_size if num_crop <= 1 else int(round(image_size * 1.14))
    crop_lefts = _crop_x_offsets(num_crop, image_size, resized_w)

    views: list[torch.Tensor] = []
    for seg_idx in range(num_segment):
        indices = pick_segment_frame_indices(
            num_available, num_frames, seg_idx, num_segment
        )
        raw_frames: list[Image.Image] = []
        for frame_index in indices:
            with Image.open(frame_paths[frame_index]) as image:
                raw_frames.append(image.convert("RGB"))

        for crop_left in crop_lefts:
            frame_tensors = []
            for frame in raw_frames:
                x = TF.resize(frame, [resized_h, resized_w])
                x = TF.crop(x, 0, crop_left, image_size, image_size)
                frame_tensors.append(normalize(TF.to_tensor(x)))
            clip = torch.stack(frame_tensors, dim=0)
            views.append(clip)
            if flip_tta:
                views.append(torch.flip(clip, dims=[-1]))
    return views


@torch.no_grad()
def _predict_dense_tta(
    *,
    model: nn.Module,
    video_dirs: list[Path],
    num_frames: int,
    image_size: int,
    use_imagenet_norm: bool,
    device: torch.device,
    untrained_mask: torch.Tensor | None,
    flip_perm: torch.Tensor | None,
    logit_adjust: torch.Tensor | None,
    amp_infer: bool,
    num_segment: int,
    num_crop: int,
    flip_tta: bool,
    micro_batch: int,
    patch_size: int | None = None,
) -> list[int]:
    """Softmax-average predictions over segment × crop × optional flip views."""
    normalize = _eval_normalize(use_imagenet_norm)
    model.eval()
    predictions: list[int] = []
    micro_batch = max(1, micro_batch)
    log_every = 200

    for vid_idx, video_dir in enumerate(video_dirs):
        views = _load_tta_video_views(
            video_dir,
            num_frames=num_frames,
            image_size=image_size,
            normalize=normalize,
            num_segment=num_segment,
            num_crop=num_crop,
            flip_tta=flip_tta,
        )
        probs_total: torch.Tensor | None = None
        n_views = 0
        for start in range(0, len(views), micro_batch):
            batch_views = views[start : start + micro_batch]
            video_batch = torch.stack(batch_views, dim=0).to(device, non_blocking=True)
            logits = _logits_for_batch(
                model,
                video_batch,
                untrained_mask,
                logit_adjust=logit_adjust,
                amp_infer=amp_infer,
                patch_size=patch_size,
            )
            batch_probs = torch.softmax(logits, dim=1)
            for local_i, probs in enumerate(batch_probs):
                view_probs = probs
                global_view = start + local_i
                # Flipped views are appended immediately after their unflipped pair.
                if flip_tta and flip_perm is not None and global_view % 2 == 1:
                    view_probs = view_probs.index_select(dim=0, index=flip_perm)
                probs_total = view_probs if probs_total is None else probs_total + view_probs
                n_views += 1
        assert probs_total is not None and n_views > 0
        predictions.append(int((probs_total / float(n_views)).argmax().item()))

        if (vid_idx + 1) % log_every == 0 or (vid_idx + 1) == len(video_dirs):
            print(f"[test-tta] {vid_idx + 1}/{len(video_dirs)} videos", flush=True)

    return predictions


def _is_videomae_vit(model: nn.Module) -> bool:
    """True when ``model`` is a :class:`~smth2smth.shared.models.video_mae.VideoMAEViT`."""
    encoder = getattr(model, "encoder", None)
    return encoder is not None and hasattr(encoder, "pos_embed")


@torch.no_grad()
def _videomae_encoder_features(encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run ``VideoMAEEncoder`` with ``pos_embed`` resized to match ``x`` (TTA scales)."""
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
) -> torch.Tensor:
    """Classifier logits for VideoMAE with spatial size implied by ``video_batch``."""
    features = _videomae_encoder_features(model.encoder, video_batch)
    if model.attn_pool is not None:
        pooled = model.attn_pool(features)
    else:
        pooled = features.mean(dim=1)
    logits = model.classifier(model.dropout(pooled))
    if untrained_mask is not None:
        logits = logits + untrained_mask
    if logit_adjust is not None:
        logits = logits + logit_adjust
    return logits


@torch.no_grad()
def _logits_for_batch(
    model: nn.Module,
    video_batch: torch.Tensor,
    untrained_mask: torch.Tensor | None,
    logit_adjust: torch.Tensor | None = None,
    *,
    amp_infer: bool = False,
    patch_size: int | None = None,
) -> torch.Tensor:
    if patch_size is not None and _is_videomae_vit(model):
        if amp_infer and video_batch.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return _videomae_logits_batch(
                    model,
                    video_batch,
                    untrained_mask=untrained_mask,
                    logit_adjust=logit_adjust,
                )
        return _videomae_logits_batch(
            model,
            video_batch,
            untrained_mask=untrained_mask,
            logit_adjust=logit_adjust,
        )
    if amp_infer:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(video_batch)
    else:
        logits = model(video_batch)
    if untrained_mask is not None:
        logits = logits + untrained_mask
    if logit_adjust is not None:
        logits = logits + logit_adjust
    return logits


def _round_spatial_to_patch_multiple(size: int, patch_size: int) -> int:
    """Round ``size`` to the nearest multiple of ``patch_size`` (minimum ``patch_size``)."""
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}.")
    return max(patch_size, int(round(size / patch_size)) * patch_size)


def _rescale_video(
    video_batch: torch.Tensor,
    scale: float,
    *,
    patch_size: int | None = None,
) -> torch.Tensor:
    """Bilinear rescale a ``(B, T, C, H, W)`` clip in the spatial dims.

    ``scale == 1.0`` is a no-op (returns the input unchanged). When
    ``patch_size`` is set (ViT / VideoMAE), spatial sides are rounded to the
    nearest multiple of ``patch_size`` so :class:`PatchEmbed3D` does not crash.
    """
    if abs(scale - 1.0) < 1e-6:
        return video_batch
    b, t, c, h, w = video_batch.shape
    min_side = patch_size if patch_size is not None else 8
    new_h = max(min_side, int(round(h * scale)))
    new_w = max(min_side, int(round(w * scale)))
    if patch_size is not None:
        new_h = _round_spatial_to_patch_multiple(new_h, patch_size)
        new_w = _round_spatial_to_patch_multiple(new_w, patch_size)
    flat = video_batch.reshape(b * t, c, h, w)
    flat = F.interpolate(
        flat, size=(new_h, new_w), mode="bilinear", align_corners=False
    )
    return flat.reshape(b, t, c, new_h, new_w)


@torch.no_grad()
def _predict(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    untrained_mask: torch.Tensor | None,
    tta_enabled: bool,
    tta_flip: bool,
    flip_perm: torch.Tensor | None,
    tta_scales: list[float] | None = None,
    logit_adjust: torch.Tensor | None = None,
    amp_infer: bool = False,
    patch_size: int | None = None,
) -> list[int]:
    """Argmax inference, optionally with multi-view TTA.

    With ``tta_enabled=False`` (default), behavior is byte-for-byte identical
    to the legacy :func:`_predict_with_optional_mask` path: a single forward
    pass, optional ``-inf`` masking on never-trained classes, then argmax.

    With ``tta_enabled=True``, the prediction is the argmax of a softmax
    average over up to ``2 * len(tta_scales)`` views (each scale ×
    ``{original, h-flipped}``). The flipped softmax is permuted by
    ``flip_perm`` so left/right-paired classes (e.g. 018 ↔ 019) collapse
    back into agreement. ``logit_adjust`` is added once per forward to
    correct for long-tail bias if set.
    """
    if tta_scales is None or len(tta_scales) == 0:
        tta_scales = [1.0]

    model.eval()
    predictions: list[int] = []
    for video_batch, _ in loader:
        video_batch = video_batch.to(device, non_blocking=True)
        if not tta_enabled:
            logits = _logits_for_batch(
                model,
                video_batch,
                untrained_mask,
                logit_adjust=logit_adjust,
                amp_infer=amp_infer,
                patch_size=patch_size,
            )
            predictions.extend(int(p) for p in logits.argmax(dim=1).cpu().tolist())
            continue

        probs_total: torch.Tensor | None = None
        n_views = 0
        for scale in tta_scales:
            scaled = _rescale_video(video_batch, scale, patch_size=patch_size)
            scaled_logits = _logits_for_batch(
                model,
                scaled,
                untrained_mask,
                logit_adjust=logit_adjust,
                amp_infer=amp_infer,
                patch_size=patch_size,
            )
            scaled_probs = torch.softmax(scaled_logits, dim=1)
            probs_total = scaled_probs if probs_total is None else probs_total + scaled_probs
            n_views += 1
            if tta_flip:
                flipped = torch.flip(scaled, dims=[-1])
                flipped_logits = _logits_for_batch(
                    model,
                    flipped,
                    untrained_mask,
                    logit_adjust=logit_adjust,
                    amp_infer=amp_infer,
                    patch_size=patch_size,
                )
                flipped_probs = torch.softmax(flipped_logits, dim=1)
                if flip_perm is not None:
                    flipped_probs = flipped_probs.index_select(dim=1, index=flip_perm)
                probs_total = probs_total + flipped_probs
                n_views += 1
        assert probs_total is not None  # at least one scale
        probs_total = probs_total / float(n_views)
        predictions.extend(int(p) for p in probs_total.argmax(dim=1).cpu().tolist())
    return predictions


def _build_logit_adjustment(
    train_dir: Path,
    num_classes: int,
    tau: float,
    device: torch.device,
) -> torch.Tensor | None:
    """Build the additive logit-adjustment vector for long-tailed inference.

    From Menon et al. 2021, *Long-tail learning via logit adjustment* (ICLR):
    at inference time, subtract ``tau * log(p_c)`` from each logit, where
    ``p_c`` is the empirical training class frequency. This shifts mass
    from over-represented classes to under-represented ones with zero
    training cost. ``tau ∈ [0.5, 1.5]`` typically works well; smaller
    values are gentler.

    Args:
        train_dir: Path to the training folder (used to count per-class
            video folders).
        num_classes: Width of the classifier head.
        tau: Strength of the adjustment.
        device: Device the returned vector lives on.

    Returns:
        ``(num_classes,)`` float tensor or ``None`` if the train folder is
        unavailable.
    """
    if not train_dir.is_dir():
        print(
            f"[tta] logit-adjust disabled: train_dir not found ({train_dir})."
        )
        return None
    samples = collect_video_samples(train_dir)
    counts = class_counts(samples, num_classes=num_classes)
    total = sum(counts)
    if total == 0:
        return None
    probs = [max(n, 1) / float(total) for n in counts]  # avoid log(0)
    adjust = -tau * torch.log(torch.tensor(probs, dtype=torch.float32, device=device))
    print(
        f"[tta] logit-adjust enabled: tau={tau}, min={float(adjust.min()):.3f}, "
        f"max={float(adjust.max()):.3f} (rare classes get boosted)."
    )
    return adjust


@hydra.main(version_base=None, config_path=CONFIGS_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra CLI entrypoint."""
    run(cfg)


if __name__ == "__main__":
    main()
