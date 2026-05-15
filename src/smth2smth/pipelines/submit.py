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
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from smth2smth.pipelines.train import CONFIGS_DIR, _resolve_device
from smth2smth.shared.data import VideoFrameDataset, build_transforms, collect_video_samples
from smth2smth.shared.data.video_dataset import parse_class_index
from smth2smth.shared.io.checkpoints import cfg_from_checkpoint, load_checkpoint
from smth2smth.shared.io.submission import (
    discover_all_test_videos,
    load_manifest_video_names,
    resolve_video_dirs,
    write_submission_csv,
)
from smth2smth.shared.models import build_model
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

    use_imagenet_norm = bool(saved_cfg.model.pretrained)
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

    tta_enabled = bool(cfg.training.get("tta", False))
    tta_flip = bool(cfg.training.get("tta_flip", True))
    flip_perm: torch.Tensor | None = None
    if tta_enabled and tta_flip:
        train_dir = Path(str(cfg.dataset.train_dir)).resolve()
        flip_perm = _build_flip_class_permutation(
            train_dir=train_dir,
            num_classes=int(cfg.num_classes),
            device=device,
        )

    # Multi-scale TTA. Default ``[1.0]`` is a no-op (single forward at the
    # training resolution). Adding e.g. ``[0.875, 1.0, 1.125]`` evaluates the
    # clip at three input sizes (centred around the trained ``image_size``)
    # and averages the softmaxes. ResNet-50 + GAP is fully convolutional so
    # this is well-defined; the cost is a linear-in-len(scales) forward pass.
    tta_scales_cfg = cfg.training.get("tta_scales", None) if tta_enabled else None
    tta_scales: list[float] = (
        [float(s) for s in tta_scales_cfg] if tta_scales_cfg else [1.0]
    )

    # Logit adjustment for long-tailed inference (Menon et al. 2021).
    # ``tta_logit_adjust > 0`` subtracts ``tau * log(p_c)`` from each logit,
    # where ``p_c`` is the empirical class frequency in the training folder.
    # Cheap and complementary to class-balanced training. Off by default.
    tau = float(cfg.training.get("tta_logit_adjust", 0.0)) if tta_enabled else 0.0
    logit_adjust: torch.Tensor | None = None
    if tau > 0.0:
        train_dir = Path(str(cfg.dataset.train_dir)).resolve()
        logit_adjust = _build_logit_adjustment(
            train_dir=train_dir,
            num_classes=int(cfg.num_classes),
            tau=tau,
            device=device,
        )

    amp_infer = bool(cfg.training.get("amp", False)) and device.type == "cuda"
    if amp_infer:
        print("[submit] inference autocast fp16 enabled (matches training.amp).")

    predictions = _predict(
        model=model,
        loader=loader,
        device=device,
        untrained_mask=untrained_mask,
        tta_enabled=tta_enabled,
        tta_flip=tta_flip,
        flip_perm=flip_perm,
        tta_scales=tta_scales,
        logit_adjust=logit_adjust,
        amp_infer=amp_infer,
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


@torch.no_grad()
def _logits_for_batch(
    model: nn.Module,
    video_batch: torch.Tensor,
    untrained_mask: torch.Tensor | None,
    logit_adjust: torch.Tensor | None = None,
    *,
    amp_infer: bool = False,
) -> torch.Tensor:
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


def _rescale_video(video_batch: torch.Tensor, scale: float) -> torch.Tensor:
    """Bilinear rescale a ``(B, T, C, H, W)`` clip in the spatial dims.

    ``scale == 1.0`` is a no-op (returns the input unchanged).
    """
    if abs(scale - 1.0) < 1e-6:
        return video_batch
    b, t, c, h, w = video_batch.shape
    new_h = max(8, int(round(h * scale)))
    new_w = max(8, int(round(w * scale)))
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
            )
            predictions.extend(int(p) for p in logits.argmax(dim=1).cpu().tolist())
            continue

        probs_total: torch.Tensor | None = None
        n_views = 0
        for scale in tta_scales:
            scaled = _rescale_video(video_batch, scale)
            scaled_logits = _logits_for_batch(
                model,
                scaled,
                untrained_mask,
                logit_adjust=logit_adjust,
                amp_infer=amp_infer,
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
