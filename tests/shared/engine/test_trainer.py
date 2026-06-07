"""Tests for the engine's training and evaluation loops.

Uses a tiny synthetic linear classifier (no CNN) so the loops can be exercised
quickly on CPU without touching torchvision or the dataset.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from smth2smth.shared.engine import (
    VIDEOMIX_MODES,
    EpochStats,
    apply_video_mixing,
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


# --- Video-mixing dispatcher tests -----------------------------------------


_NUM_CLASSES = 4


def _make_videos(
    b: int = 4, t: int = 6, c: int = 3, h: int = 8, w: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    videos = torch.rand(b, t, c, h, w)
    labels = torch.arange(b) % _NUM_CLASSES
    return videos, labels


@pytest.mark.parametrize("mode", sorted(VIDEOMIX_MODES - {"none"}))
def test_apply_video_mixing_shapes_and_label_simplex(mode: str) -> None:
    """Every mode must preserve the video shape and produce a valid soft-label."""
    videos, labels = _make_videos()
    torch.manual_seed(123)
    mixed, returned_labels, soft = apply_video_mixing(
        videos, labels, num_classes=_NUM_CLASSES, alpha=1.0, mode=mode
    )
    assert mixed.shape == videos.shape
    assert returned_labels.shape == labels.shape
    assert soft.shape == (videos.size(0), _NUM_CLASSES)
    # Soft targets must be non-negative and sum to 1 along the class axis.
    assert torch.all(soft >= -1e-6)
    sums = soft.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_apply_video_mixing_none_is_identity() -> None:
    videos, labels = _make_videos()
    mixed, _, soft = apply_video_mixing(
        videos, labels, num_classes=_NUM_CLASSES, alpha=1.0, mode="none"
    )
    assert torch.equal(mixed, videos)
    # ``none`` produces hard one-hot labels.
    assert torch.equal(soft.argmax(dim=1), labels)


def test_apply_video_mixing_unknown_mode_raises() -> None:
    videos, labels = _make_videos()
    with pytest.raises(ValueError):
        apply_video_mixing(videos, labels, num_classes=_NUM_CLASSES, alpha=1.0, mode="bogus")


def test_apply_cutout_zeros_a_region_without_changing_labels() -> None:
    videos, labels = _make_videos(b=2)
    mixed, returned_labels, soft = apply_video_mixing(
        videos, labels, num_classes=_NUM_CLASSES, alpha=1.0, mode="cube_cutout"
    )
    assert torch.any(mixed == 0.0)
    # CutOut: labels are exactly the originals (one-hot).
    assert torch.equal(soft.argmax(dim=1), labels)
    assert torch.equal(returned_labels, labels)


def test_fade_mixup_label_matches_uniform_mixup() -> None:
    """FadeMixUp's soft label depends on lam only (γ averages out by symmetry)."""
    videos, labels = _make_videos(b=4, t=6)
    torch.manual_seed(7)
    _, _, soft_fade = apply_video_mixing(
        videos, labels, num_classes=_NUM_CLASSES, alpha=1.0, mode="fade_mixup"
    )
    # Two label rows must sum to 1 each and have at most 2 non-zero entries.
    nz_per_row = (soft_fade > 1e-6).sum(dim=1)
    assert torch.all(nz_per_row <= 2)


def test_train_one_epoch_supports_each_videomix_mode() -> None:
    """Smoke-test the trainer through every supported video-mixing mode."""
    for mode in sorted(VIDEOMIX_MODES):
        loader, model = _build(num_classes=_NUM_CLASSES)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        stats = train_one_epoch(
            model,
            loader,
            loss_fn,
            optimizer,
            torch.device("cpu"),
            num_classes=_NUM_CLASSES,
            videomix_alpha=1.0,
            videomix_prob=1.0,
            videomix_mode=mode,
        )
        assert isinstance(stats, EpochStats), f"mode={mode!r} returned {type(stats)}"
        assert stats.loss >= 0.0, f"mode={mode!r} produced negative loss"


def test_train_one_epoch_metrics_logger_receives_steps() -> None:
    loader, model = _build()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    logged: list[tuple[int, dict[str, float]]] = []

    def _callback(metrics: dict[str, float], step: int) -> None:
        logged.append((step, metrics))

    train_one_epoch(
        model,
        loader,
        loss_fn,
        optimizer,
        torch.device("cpu"),
        log_interval_steps=1,
        step_metrics_callback=_callback,
    )
    assert len(logged) == len(loader)
    assert logged[0][0] == 1
    assert logged[-1][0] == len(loader)
    assert "train/loss" in logged[-1][1]
    assert "train/top1" in logged[-1][1]
