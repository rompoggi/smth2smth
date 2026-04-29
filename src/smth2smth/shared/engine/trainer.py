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

    for video_batch, labels in data_loader:
        video_batch = video_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(video_batch)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        top1, top5 = accuracy_topk(logits.detach(), labels, topk=(1, 5))
        running_loss += float(loss.item()) * batch_size
        running_top1_correct += float(top1.item()) * batch_size
        running_top5_correct += float(top5.item()) * batch_size
        total += batch_size

    return _aggregate(running_loss, running_top1_correct, running_top5_correct, total)


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
