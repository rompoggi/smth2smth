"""Training and evaluation loops.

Pure functions: no Hydra imports here. Pipelines build a ``DataLoader`` and
call these directly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from smth2smth.shared.engine.metrics import accuracy_topk


@dataclass
class EpochStats:
    """Per-epoch metrics returned by training and evaluation loops."""

    loss: float
    top1: float
    top5: float


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    num_classes: int | None = None,
    label_smoothing: float = 0.0,
    videomix_alpha: float = 0.0,
    videomix_prob: float = 1.0,
    videomix_mode: str = "cube_cutmix",
    videomix_mixup_alpha: float | None = None,
    videomix_cutmix_alpha: float | None = None,
    videomix_switch_prob: float = 0.5,
    log_interval_steps: int = 0,
    grad_accum_steps: int = 1,
    scaler: torch.amp.GradScaler | None = None,
    amp_dtype: torch.dtype = torch.float16,
    ema_model: torch.optim.swa_utils.AveragedModel | None = None,
    class_weights: torch.Tensor | None = None,
    stability_logger: Callable[..., None] | None = None,
    step_metrics_callback: Callable[[dict[str, float], int], None] | None = None,
) -> EpochStats:
    """Run one training epoch and return aggregated metrics.

    Args:
        model: Neural network to train. Set to ``train()`` mode internally.
        data_loader: Yields ``(video_batch, labels)`` tuples; videos are
            ``(B, T, C, H, W)``, labels are ``(B,)`` integer class indices.
        loss_fn: A standard classification loss (e.g. ``nn.CrossEntropyLoss``).
        optimizer: Optimizer driving the parameter updates.
        device: Target device for batches and loss computation.
        scaler: Optional :class:`torch.amp.GradScaler`. When provided and the
            device is CUDA, the forward pass runs under ``autocast(amp_dtype)``
            and gradients are scaled before ``backward``. When ``None`` (default),
            training is full-precision -- byte-for-byte identical to the legacy
            code path.
        amp_dtype: Autocast dtype (``torch.float16`` or ``torch.bfloat16``).
            Ignored when ``scaler`` is ``None``.
        ema_model: Optional :class:`torch.optim.swa_utils.AveragedModel`. When
            provided, ``ema_model.update_parameters(model)`` is called after
            every optimizer step so the EMA copy tracks the live weights.
        class_weights: Optional ``(num_classes,)`` float tensor used in the
            soft-target cross-entropy path (label-smoothing and video-mixing).
            The plain ``loss_fn`` is already class-weighted by its own
            constructor when needed. ``None`` ⇒ unweighted soft CE.

    Returns:
        :class:`EpochStats` with sample-weighted average loss, top-1, top-5.
    """
    model.train()
    running_loss = 0.0
    running_top1_correct = 0.0
    running_top5_correct = 0.0
    total = 0

    use_amp = scaler is not None and device.type == "cuda"
    accum_steps = max(1, int(grad_accum_steps))

    total_steps = len(data_loader)
    optimizer.zero_grad(set_to_none=True)
    for step_idx, (video_batch, labels) in enumerate(data_loader, start=1):
        video_batch = video_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        mixed_labels: torch.Tensor | None = None
        train_labels = labels
        use_videomix = (
            num_classes is not None
            and videomix_mode != "none"
            and torch.rand(1).item() < videomix_prob
            and (
                videomix_alpha > 0.0
                or videomix_mode == "mixup_cutmix_switch"
            )
        )
        if use_videomix:
            video_batch, train_labels, mixed_labels = apply_video_mixing(
                video_batch,
                labels,
                num_classes=num_classes,
                alpha=videomix_alpha,
                mode=videomix_mode,
                mixup_alpha=videomix_mixup_alpha,
                cutmix_alpha=videomix_cutmix_alpha,
                switch_prob=videomix_switch_prob,
            )

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model(video_batch)
            if mixed_labels is not None:
                loss = _soft_target_cross_entropy(
                    logits, mixed_labels, class_weights=class_weights
                )
            else:
                if label_smoothing > 0.0:
                    smoothed = _one_hot_targets(
                        train_labels, num_classes=int(num_classes), smoothing=label_smoothing
                    )
                    loss = _soft_target_cross_entropy(
                        logits, smoothed, class_weights=class_weights
                    )
                else:
                    loss = loss_fn(logits, train_labels)

        scaled_loss = loss / float(accum_steps)
        if use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        is_accum_step = (step_idx % accum_steps == 0) or (step_idx == total_steps)
        if is_accum_step:
            # Optional per-step stability logging. Called *before* the
            # optimizer step so gradients are still resident; unscales the
            # GradScaler in-place under fp16 AMP. Pre-clip (no clipping in
            # this recipe), so the logged norms reflect the raw optimization
            # signal.
            if stability_logger is not None:
                stability_logger(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler if use_amp else None,
                    loss_value=float(loss.item()),
                )
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema_model is not None:
                ema_model.update_parameters(model)

        batch_size = labels.size(0)
        top1, top5 = accuracy_topk(logits.detach(), train_labels, topk=(1, 5))
        running_loss += float(loss.item()) * batch_size
        running_top1_correct += float(top1.item()) * batch_size
        running_top5_correct += float(top5.item()) * batch_size
        total += batch_size

        if log_interval_steps > 0 and (
            step_idx % log_interval_steps == 0 or step_idx == total_steps
        ):
            avg_loss = running_loss / max(1, total)
            avg_top1 = running_top1_correct / max(1, total)
            avg_top5 = running_top5_correct / max(1, total)
            ts = datetime.now().strftime("%H:%M:%S")
            print(
                f"    [{ts}] step {step_idx}/{total_steps} | "
                f"avg train loss {avg_loss:.4f} top1 {avg_top1:.4f} top5 {avg_top5:.4f}"
            )
            if step_metrics_callback is not None:
                step_metrics_callback(
                    {
                        "train/loss": avg_loss,
                        "train/top1": avg_top1,
                        "train/top5": avg_top5,
                    },
                    step_idx,
                )

    return _aggregate(running_loss, running_top1_correct, running_top5_correct, total)


def _one_hot_targets(
    labels: torch.Tensor, num_classes: int, smoothing: float = 0.0
) -> torch.Tensor:
    if num_classes <= 1:
        raise ValueError(f"num_classes must be > 1, got {num_classes}.")
    with torch.no_grad():
        target = torch.zeros(labels.size(0), num_classes, device=labels.device, dtype=torch.float32)
        target.scatter_(1, labels.unsqueeze(1), 1.0)
        if smoothing > 0.0:
            target = target * (1.0 - smoothing) + (1.0 - target) * (smoothing / (num_classes - 1))
    return target


def _soft_target_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy with soft targets, optionally class-reweighted.

    When ``class_weights`` is provided, each sample's loss is scaled by
    ``sum_c (class_weights[c] * targets[i, c])``. For pure one-hot targets
    this recovers the canonical per-class weighting used by
    :class:`torch.nn.CrossEntropyLoss(weight=...)`. For soft targets
    (label smoothing, MixUp, CutMix) it is the natural extension: rare
    classes still get higher influence proportional to their mass in the
    soft target distribution.
    """
    log_probs = torch.log_softmax(logits, dim=1)
    per_sample = -(targets * log_probs).sum(dim=1)
    if class_weights is None:
        return per_sample.mean()
    w = (targets * class_weights.unsqueeze(0)).sum(dim=1)
    denom = w.sum().clamp_min(1e-12)
    return (per_sample * w).sum() / denom


VIDEOMIX_MODES: frozenset[str] = frozenset(
    {
        "none",
        "spatial_cutout",
        "frame_cutout",
        "cube_cutout",
        "spatial_cutmix",
        "frame_cutmix",
        "cube_cutmix",
        "mixup",
        "frame_mixup",
        "cube_mixup",
        "fade_mixup",
        "cutmixup",
        "frame_cutmixup",
        "cube_cutmixup",
        "mixup_cutmix_switch",
    }
)


def apply_video_mixing(
    videos: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    alpha: float,
    mode: str = "cube_cutmix",
    mixup_alpha: float | None = None,
    cutmix_alpha: float | None = None,
    switch_prob: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a video-level mixing/deleting augmentation, returning soft targets.

    Implements the data-level deleting / cut-and-pasting / blending policies
    of Kim et al. 2020 (Sec. 3.2) for the spatial, temporal (Frame*), and
    spatiotemporal (Cube*) variants of CutOut, CutMix, MixUp, and CutMixUp.

    Args:
        videos: ``(B, T, C, H, W)`` float tensor on any device.
        labels: ``(B,)`` int64 class indices, on the same device as ``videos``.
        num_classes: Total number of classes (for one-hot expansion).
        alpha: Beta(α, α) parameter. ``λ`` is sampled as ``Beta(α, α)`` for
            mixing modes, and ``1 - λ`` parametrizes the deletion size for
            CutOut modes.
        mode: One of :data:`VIDEOMIX_MODES`. Default ``"cube_cutmix"`` matches
            the legacy ``_apply_videomix`` behavior.

    Returns:
        Tuple ``(mixed_videos, labels, soft_targets)``:
            * ``mixed_videos`` -- same shape as ``videos``.
            * ``labels``       -- the original labels (unchanged; trainers may
                                  still want them for top-k accuracy).
            * ``soft_targets`` -- ``(B, num_classes)`` float, suitable for
                                  :func:`_soft_target_cross_entropy`.

    Raises:
        ValueError: On unknown ``mode``, non-5D ``videos``, or non-positive ``alpha``.
    """
    if videos.dim() != 5:
        raise ValueError(f"videos must be 5-D (B,T,C,H,W), got {tuple(videos.shape)}")
    if mode not in VIDEOMIX_MODES:
        raise ValueError(f"mode must be one of {sorted(VIDEOMIX_MODES)}, got {mode!r}")
    # The mixup_cutmix_switch mode resolves ``alpha`` from per-submode
    # ``mixup_alpha`` / ``cutmix_alpha`` further down; the outer ``alpha`` is
    # legitimately unused in that path so we skip the early check for it (and
    # the switch branch has its own post-resolution alpha check).
    if mode != "mixup_cutmix_switch" and alpha <= 0:
        raise ValueError(f"alpha must be positive for video mixing, got {alpha}")

    # SSv2 fine-tune recipe: per-batch coin flip between MixUp and (cube)CutMix
    # with independent Beta parameters (VideoMAE FINETUNE.md: mixup 0.8,
    # cutmix 1.0, switch_prob 0.5). Resolves to a concrete sub-mode + alpha so
    # the rest of this function is unchanged.
    if mode == "mixup_cutmix_switch":
        if torch.rand(1).item() < switch_prob:
            mode = "mixup"
            alpha = float(mixup_alpha) if mixup_alpha is not None else alpha
        else:
            mode = "cube_cutmix"
            alpha = float(cutmix_alpha) if cutmix_alpha is not None else alpha
        if alpha <= 0:
            raise ValueError(
                "mixup_cutmix_switch resolved to a non-positive alpha; set "
                "videomix_mixup_alpha / videomix_cutmix_alpha."
            )

    if mode == "none":
        return videos, labels, _one_hot_targets(labels, num_classes=num_classes, smoothing=0.0)

    b = videos.shape[0]
    y_a = _one_hot_targets(labels, num_classes=num_classes, smoothing=0.0)
    if b < 2 and not mode.endswith("cutout"):
        # All non-CutOut modes need a second sample.
        return videos, labels, y_a

    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())

    if mode.endswith("cutout"):
        return _apply_cutout(videos, y_a, mode=mode, lam=lam)

    perm = torch.randperm(b, device=videos.device)
    y_b = _one_hot_targets(labels[perm], num_classes=num_classes, smoothing=0.0)

    # Order matters: check the more specific suffix first ("cutmixup" ends in "mixup").
    if mode.endswith("cutmixup"):
        return _apply_cutmixup(videos, y_a, y_b, perm=perm, mode=mode, lam=lam)
    if mode.endswith("cutmix"):
        return _apply_cutmix(videos, y_a, y_b, perm=perm, mode=mode, lam=lam)
    if mode == "fade_mixup":
        return _apply_fade_mixup(videos, y_a, y_b, perm=perm, lam=lam)
    if mode.endswith("mixup"):
        return _apply_mixup(videos, y_a, y_b, perm=perm, mode=mode, lam=lam)

    # Should be unreachable since we validated `mode` above.
    raise ValueError(f"Unhandled videomix mode: {mode!r}")


def _apply_videomix(
    videos: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backwards-compatible wrapper preserving the legacy CubeCutMix behavior."""
    return apply_video_mixing(
        videos, labels, num_classes=num_classes, alpha=alpha, mode="cube_cutmix"
    )


def _sample_box(
    t: int, h: int, w: int, lam: float, mode: str, device: torch.device
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], float]:
    """Sample a (t, h, w) deletion/paste box and return ``(t-range, h-range, w-range, vol_ratio)``.

    The box dimensions follow the convention from the original CutMix paper
    (volume scales as ``1 - lam``); for the spatial-only case we collapse the
    temporal axis to the full clip, and for the frame-only case we collapse
    the spatial axes to the full frame.
    """
    if mode.startswith("spatial"):
        cut_t = t
        cut_h = max(1, int(round(h * (1.0 - lam) ** 0.5)))
        cut_w = max(1, int(round(w * (1.0 - lam) ** 0.5)))
    elif mode.startswith("frame"):
        cut_t = max(1, int(round(t * (1.0 - lam))))
        cut_h = h
        cut_w = w
    else:  # cube_*
        cut_ratio = (1.0 - lam) ** (1.0 / 3.0)
        cut_t = max(1, int(round(t * cut_ratio)))
        cut_h = max(1, int(round(h * cut_ratio)))
        cut_w = max(1, int(round(w * cut_ratio)))

    cut_t = min(cut_t, t)
    cut_h = min(cut_h, h)
    cut_w = min(cut_w, w)

    t0 = int(torch.randint(0, max(1, t - cut_t + 1), (1,), device=device).item())
    h0 = int(torch.randint(0, max(1, h - cut_h + 1), (1,), device=device).item())
    w0 = int(torch.randint(0, max(1, w - cut_w + 1), (1,), device=device).item())
    t1 = t0 + cut_t
    h1 = h0 + cut_h
    w1 = w0 + cut_w

    vol_ratio = (cut_t * cut_h * cut_w) / float(t * h * w)
    return (t0, t1), (h0, h1), (w0, w1), vol_ratio


def _apply_cutout(
    videos: torch.Tensor,
    y_a: torch.Tensor,
    *,
    mode: str,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CutOut / FrameCutOut / CubeCutOut. Zeroes a region; label is unchanged."""
    _, t, _, h, w = videos.shape
    (t0, t1), (h0, h1), (w0, w1), _ = _sample_box(
        t=t, h=h, w=w, lam=lam, mode=mode, device=videos.device
    )
    mixed = videos.clone()
    mixed[:, t0:t1, :, h0:h1, w0:w1] = 0.0
    return mixed, _argmax_labels(y_a), y_a


def _apply_cutmix(
    videos: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    *,
    perm: torch.Tensor,
    mode: str,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CutMix / FrameCutMix / CubeCutMix. Paste a box from ``videos[perm]``."""
    _, t, _, h, w = videos.shape
    (t0, t1), (h0, h1), (w0, w1), vol_ratio = _sample_box(
        t=t, h=h, w=w, lam=lam, mode=mode, device=videos.device
    )
    mixed = videos.clone()
    mixed[:, t0:t1, :, h0:h1, w0:w1] = videos[perm, t0:t1, :, h0:h1, w0:w1]
    keep_ratio = 1.0 - vol_ratio
    y_mixed = keep_ratio * y_a + vol_ratio * y_b
    return mixed, _argmax_labels(y_a), y_mixed


def _apply_mixup(
    videos: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    *,
    perm: torch.Tensor,
    mode: str,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MixUp / FrameMixUp / CubeMixUp.

    ``mixup``: uniform blend everywhere with ratio ``lam``.

    ``frame_mixup`` and ``cube_mixup``: per-pixel mask whose support is
    selected with the same box logic as CutMix; inside the mask the two clips
    are linearly blended at ``lam``, outside it the original clip is kept. The
    label proportion follows the volume-weighted mix of the two regimes.
    """
    if mode == "mixup":
        mixed = lam * videos + (1.0 - lam) * videos[perm]
        y_mixed = lam * y_a + (1.0 - lam) * y_b
        return mixed, _argmax_labels(y_a), y_mixed

    _, t, _, h, w = videos.shape
    (t0, t1), (h0, h1), (w0, w1), vol_ratio = _sample_box(
        t=t, h=h, w=w, lam=lam, mode=mode, device=videos.device
    )
    mixed = videos.clone()
    region_a = videos[:, t0:t1, :, h0:h1, w0:w1]
    region_b = videos[perm, t0:t1, :, h0:h1, w0:w1]
    mixed[:, t0:t1, :, h0:h1, w0:w1] = lam * region_a + (1.0 - lam) * region_b
    # Outside the mask: 100% sample A. Inside: lam * A + (1-lam) * B.
    weight_a = (1.0 - vol_ratio) + vol_ratio * lam
    weight_b = vol_ratio * (1.0 - lam)
    y_mixed = weight_a * y_a + weight_b * y_b
    return mixed, _argmax_labels(y_a), y_mixed


def _apply_fade_mixup(
    videos: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    *,
    perm: torch.Tensor,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FadeMixUp (Kim et al. 2020, eq. 2).

    Per-frame mixing ratio ``λ̃_t`` interpolates linearly between ``λ - γ`` and
    ``λ + γ`` where ``γ ~ Uniform(0, min(λ, 1 - λ))``. Because the endpoints
    are symmetric around ``λ``, the mean ratio is still ``λ`` and the soft
    label is identical to MixUp's.
    """
    _, t, _, _, _ = videos.shape
    gamma_max = min(lam, 1.0 - lam)
    gamma = float(torch.empty(1, device=videos.device).uniform_(0.0, gamma_max).item())
    if t == 1:
        ratios_t = torch.tensor([lam], device=videos.device, dtype=videos.dtype)
    else:
        ratios_t = torch.linspace(
            lam - gamma, lam + gamma, steps=t, device=videos.device, dtype=videos.dtype
        )
    ratios = ratios_t.view(1, t, 1, 1, 1)
    mixed = ratios * videos + (1.0 - ratios) * videos[perm]
    y_mixed = lam * y_a + (1.0 - lam) * y_b
    return mixed, _argmax_labels(y_a), y_mixed


def _apply_cutmixup(
    videos: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    *,
    perm: torch.Tensor,
    mode: str,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CutMixUp / FrameCutMixUp / CubeCutMixUp (Kim et al. 2020, eq. 1).

    Inside the box: ``lam * A + (1 - lam) * B``.
    Outside the box: ``A`` if ``lam < 0.5`` else ``B``.

    The soft label is derived from the actual pixel composition (which forces
    it to sum to 1):
        * ``lam < 0.5`` : ``y = (1 - vol*(1-lam)) y_A + vol*(1-lam) y_B``
        * ``lam >= 0.5``: ``y = lam*vol y_A + (1 - lam*vol) y_B``
    """
    _, t, _, h, w = videos.shape
    box_mode = mode.replace("cutmixup", "cutmix")
    (t0, t1), (h0, h1), (w0, w1), vol_ratio = _sample_box(
        t=t, h=h, w=w, lam=lam, mode=box_mode, device=videos.device
    )

    region_a = videos[:, t0:t1, :, h0:h1, w0:w1]
    region_b = videos[perm, t0:t1, :, h0:h1, w0:w1]
    if lam < 0.5:
        mixed = videos.clone()
        mixed[:, t0:t1, :, h0:h1, w0:w1] = lam * region_a + (1.0 - lam) * region_b
        weight_a = 1.0 - vol_ratio * (1.0 - lam)
        weight_b = vol_ratio * (1.0 - lam)
    else:
        mixed = videos[perm].clone()
        mixed[:, t0:t1, :, h0:h1, w0:w1] = lam * region_a + (1.0 - lam) * region_b
        weight_a = lam * vol_ratio
        weight_b = 1.0 - lam * vol_ratio
    y_mixed = weight_a * y_a + weight_b * y_b
    return mixed, _argmax_labels(y_a), y_mixed


def _argmax_labels(soft_targets: torch.Tensor) -> torch.Tensor:
    """Recover the integer labels of the original (un-permuted) batch."""
    return soft_targets.argmax(dim=1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> EpochStats:
    """Run one evaluation pass over ``data_loader``.

    Args:
        model: Neural network to evaluate. Set to ``eval()`` mode internally.
        data_loader: Same contract as in :func:`train_one_epoch`.
        loss_fn: Loss used for monitoring (no gradients flow).
        device: Target device for batches and loss computation.
        amp_enabled: When ``True`` and ``device`` is CUDA, run the forward
            pass under :func:`torch.amp.autocast`. Default ``False`` keeps the
            legacy full-precision path bit-for-bit identical.
        amp_dtype: Autocast dtype. Ignored when ``amp_enabled`` is ``False``.

    Returns:
        :class:`EpochStats` with sample-weighted average loss, top-1, top-5.
    """
    model.eval()
    running_loss = 0.0
    running_top1_correct = 0.0
    running_top5_correct = 0.0
    total = 0

    use_amp = amp_enabled and device.type == "cuda"

    for video_batch, labels in data_loader:
        video_batch = video_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model(video_batch)
            loss = loss_fn(logits, labels)

        batch_size = labels.size(0)
        top1, top5 = accuracy_topk(logits, labels, topk=(1, 5))
        running_loss += float(loss.item()) * batch_size
        running_top1_correct += float(top1.item()) * batch_size
        running_top5_correct += float(top5.item()) * batch_size
        total += batch_size

    return _aggregate(running_loss, running_top1_correct, running_top5_correct, total)


def _aggregate(loss_sum: float, top1_sum: float, top5_sum: float, total: int) -> EpochStats:
    if total == 0:
        return EpochStats(loss=0.0, top1=0.0, top5=0.0)
    return EpochStats(
        loss=loss_sum / total,
        top1=top1_sum / total,
        top5=top5_sum / total,
    )


def predict_argmax(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Run inference and return top-1 predictions plus the labels seen.

    Args:
        model: Network. Set to ``eval()`` mode internally.
        data_loader: Yields ``(video_batch, labels)`` tuples.
        device: Target device.

    Returns:
        Tuple ``(predictions, labels)``. Both are Python lists of ``int``.
    """
    model.eval()
    predictions: list[int] = []
    labels_seen: list[int] = []
    with torch.no_grad():
        for video_batch, labels in data_loader:
            video_batch = video_batch.to(device, non_blocking=True)
            logits = model(video_batch)
            predictions.extend(int(p) for p in logits.argmax(dim=1).cpu().tolist())
            labels_seen.extend(int(label) for label in labels.cpu().tolist())
    return predictions, labels_seen
