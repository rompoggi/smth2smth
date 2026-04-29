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
            and torch.rand(1).item() < videomix_prob
        ):
            video_batch, train_labels, mixed_labels = _apply_videomix(
                video_batch, labels, num_classes=num_classes, alpha=videomix_alpha
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

        if log_interval_steps > 0 and (step_idx % log_interval_steps == 0 or step_idx == total_steps):
            avg_loss = running_loss / max(1, total)
            avg_top1 = running_top1_correct / max(1, total)
            avg_top5 = running_top5_correct / max(1, total)
            print(
                f"    step {step_idx}/{total_steps} | "
                f"avg train loss {avg_loss:.4f} top1 {avg_top1:.4f} top5 {avg_top5:.4f}"
            )

    return _aggregate(running_loss, running_top1_correct, running_top5_correct, total)


def _one_hot_targets(labels: torch.Tensor, num_classes: int, smoothing: float = 0.0) -> torch.Tensor:
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


def _apply_videomix(
    videos: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if videos.dim() != 5:
        raise ValueError(f"videos must be 5-D (B,T,C,H,W), got {tuple(videos.shape)}")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive for VideoMix, got {alpha}")

    b, t, _, h, w = videos.shape
    if b < 2:
        return videos, labels, _one_hot_targets(labels, num_classes=num_classes, smoothing=0.0)

    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    perm = torch.randperm(b, device=videos.device)

    cut_ratio = (1.0 - lam) ** (1.0 / 3.0)
    cut_t = max(1, int(t * cut_ratio))
    cut_h = max(1, int(h * cut_ratio))
    cut_w = max(1, int(w * cut_ratio))
    t0 = int(torch.randint(0, max(1, t - cut_t + 1), (1,), device=videos.device).item())
    h0 = int(torch.randint(0, max(1, h - cut_h + 1), (1,), device=videos.device).item())
    w0 = int(torch.randint(0, max(1, w - cut_w + 1), (1,), device=videos.device).item())
    t1, h1, w1 = t0 + cut_t, h0 + cut_h, w0 + cut_w

    mixed = videos.clone()
    mixed[:, t0:t1, :, h0:h1, w0:w1] = videos[perm, t0:t1, :, h0:h1, w0:w1]
    keep_ratio = 1.0 - ((t1 - t0) * (h1 - h0) * (w1 - w0)) / float(t * h * w)
    y_a = _one_hot_targets(labels, num_classes=num_classes, smoothing=0.0)
    y_b = _one_hot_targets(labels[perm], num_classes=num_classes, smoothing=0.0)
    y_mixed = keep_ratio * y_a + (1.0 - keep_ratio) * y_b
    return mixed, labels, y_mixed


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
