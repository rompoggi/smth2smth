"""Tests for the engine's training and evaluation loops.

Uses a tiny synthetic linear classifier (no CNN) so the loops can be exercised
quickly on CPU without touching torchvision or the dataset.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from smth2smth.shared.engine import (
    EpochStats,
    evaluate_epoch,
    predict_argmax,
    train_one_epoch,
)


class _SyntheticBatchDataset(Dataset):
    """Returns ``(video, label)`` tuples with a fixed deterministic mapping."""

    def __init__(self, num_samples: int, num_classes: int) -> None:
        self.num_samples = num_samples
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Produce a one-hot-ish video tensor whose mean reveals the class.
        label = idx % self.num_classes
        video = torch.zeros(2, 3, 4, 4)  # (T, C, H, W)
        video[..., label % 4] = float(label + 1)  # easy-to-learn signal
        return video, torch.tensor(label, dtype=torch.long)


class _MeanThenLinear(nn.Module):
    """Mean over (T, H, W) then a linear layer; matches the (B, T, C, H, W) input contract."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        pooled = video_batch.mean(dim=(1, 3, 4))  # (B, C)
        return self.fc(pooled)


def _build(num_classes: int = 3) -> tuple[DataLoader, _MeanThenLinear]:
    ds = _SyntheticBatchDataset(num_samples=12, num_classes=num_classes)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    model = _MeanThenLinear(in_channels=3, num_classes=num_classes)
    return loader, model


class TestTrainAndEvaluate:
    def test_train_one_epoch_returns_finite_stats(self) -> None:
        loader, model = _build()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        stats = train_one_epoch(model, loader, loss_fn, optimizer, torch.device("cpu"))
        assert isinstance(stats, EpochStats)
        assert 0.0 <= stats.top1 <= 1.0
        assert 0.0 <= stats.top5 <= 1.0
        assert stats.loss >= 0.0

    def test_evaluate_epoch_no_grad_computation(self) -> None:
        loader, model = _build()
        loss_fn = nn.CrossEntropyLoss()
        for p in model.parameters():
            p.requires_grad_(True)
        stats = evaluate_epoch(model, loader, loss_fn, torch.device("cpu"))
        assert isinstance(stats, EpochStats)
        # Make sure no gradients were accumulated by the eval pass.
        for p in model.parameters():
            assert p.grad is None or torch.all(p.grad == 0)

    def test_predict_argmax_returns_aligned_lists(self) -> None:
        loader, model = _build(num_classes=3)
        preds, labels = predict_argmax(model, loader, torch.device("cpu"))
        assert len(preds) == len(labels) == len(loader.dataset)  # type: ignore[arg-type]
        assert all(0 <= p < 3 for p in preds)
        assert all(0 <= label < 3 for label in labels)

    def test_training_reduces_loss(self) -> None:
        # Run a few epochs and confirm loss strictly decreases on this trivial task.
        loader, model = _build()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        first = train_one_epoch(model, loader, loss_fn, optimizer, torch.device("cpu"))
        for _ in range(5):
            train_one_epoch(model, loader, loss_fn, optimizer, torch.device("cpu"))
        last = evaluate_epoch(model, loader, loss_fn, torch.device("cpu"))
        assert last.loss < first.loss

    def test_train_one_epoch_supports_label_smoothing_and_videomix(self) -> None:
        loader, model = _build(num_classes=3)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        stats = train_one_epoch(
            model,
            loader,
            loss_fn,
            optimizer,
            torch.device("cpu"),
            num_classes=3,
            label_smoothing=0.1,
            videomix_alpha=1.0,
            videomix_prob=1.0,
        )
        assert isinstance(stats, EpochStats)
        assert 0.0 <= stats.top1 <= 1.0
        assert 0.0 <= stats.top5 <= 1.0
        assert stats.loss >= 0.0


def _drain(it: Iterator) -> list:
    return list(it)
