"""Training and evaluation loops.

Pure functions: no Hydra imports here. Pipelines build a ``DataLoader`` and
call these directly.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    log_interval_steps: int = 0,
) -> EpochStats:
    """Run one training epoch and return aggregated metrics.

    Args:
        model: Neural network to train. Set to ``train()`` mode internally.
        data_loader: Yields ``(video_batch, labels)`` tuples; videos are
            ``(B, T, C, H, W)``, labels are ``(B,)`` integer class indices.
        loss_fn: A standard classification loss (e.g. ``nn.CrossEntropyLoss``).
        optimizer: Optimizer driving the parameter updates.
        device: Target device for batches and loss computation.

    Returns:
        :class:`EpochStats` with sample-weighted average loss, top-1, top-5.
    """
    model.train()
    running_loss = 0.0
    running_top1_correct = 0.0
    running_top5_correct = 0.0
    total = 0

    total_steps = len(data_loader)
    for step_idx, (video_batch, labels) in enumerate(data_loader, start=1):
        video_batch = video_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        mixed_labels: torch.Tensor | None = None
        train_labels = labels
        if (
            num_classes is not None
            and videomix_alpha > 0.0
            and videomix_mode != "none"
            and torch.rand(1).item() < videomix_prob
        ):
            video_batch, train_labels, mixed_labels = apply_video_mixing(
                video_batch,
                labels,
                num_classes=num_classes,
                alpha=videomix_alpha,
                mode=videomix_mode,
            )

        logits = model(video_batch)
        if mixed_labels is not None:
            loss = _soft_target_cross_entropy(logits, mixed_labels)
        else:
            if label_smoothing > 0.0:
                smoothed = _one_hot_targets(
                    train_labels, num_classes=int(num_classes), smoothing=label_smoothing
                )
                loss = _soft_target_cross_entropy(logits, smoothed)
            else:
                loss = loss_fn(logits, train_labels)
        loss.backward()
        optimizer.step()

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
            print(
                f"    step {step_idx}/{total_steps} | "
                f"avg train loss {avg_loss:.4f} top1 {avg_top1:.4f} top5 {avg_top5:.4f}"
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


def _soft_target_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


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
    }
)


def apply_video_mixing(
    videos: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    alpha: float,
    mode: str = "cube_cutmix",
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
    if alpha <= 0:
        raise ValueError(f"alpha must be positive for video mixing, got {alpha}")
    if mode not in VIDEOMIX_MODES:
        raise ValueError(f"mode must be one of {sorted(VIDEOMIX_MODES)}, got {mode!r}")
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
) -> EpochStats:
    """Run one evaluation pass over ``data_loader``.

    Args:
        model: Neural network to evaluate. Set to ``eval()`` mode internally.
        data_loader: Same contract as in :func:`train_one_epoch`.
        loss_fn: Loss used for monitoring (no gradients flow).
        device: Target device for batches and loss computation.

    Returns:
        :class:`EpochStats` with sample-weighted average loss, top-1, top-5.
    """
    model.eval()
    running_loss = 0.0
    running_top1_correct = 0.0
    running_top5_correct = 0.0
    total = 0

    for video_batch, labels in data_loader:
        video_batch = video_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
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
