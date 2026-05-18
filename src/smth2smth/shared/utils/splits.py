"""Train/val split helpers."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

VideoSample = tuple[Path, int]


def split_train_val(
    samples: list[VideoSample],
    val_ratio: float,
    seed: int,
) -> tuple[list[VideoSample], list[VideoSample]]:
    """Shuffle ``samples`` deterministically and split into train and val.

    Args:
        samples: Full list of ``(video_dir, label)`` pairs.
        val_ratio: Fraction reserved for validation. Use ``0.0`` to skip splitting.
        seed: RNG seed for the in-place shuffle.

    Returns:
        Tuple ``(train_samples, val_samples)``. The split mirrors the helper used
        in the professor baseline so train.py and evaluate.py stay consistent.
    """
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    if val_ratio <= 0.0:
        return shuffled, []

    n_val = int(round(len(shuffled) * val_ratio))
    n_val = max(1, n_val) if len(shuffled) > 1 else 0

    val_samples = shuffled[:n_val]
    train_samples = shuffled[n_val:]
    if len(train_samples) == 0:
        train_samples = val_samples[:-1]
        val_samples = val_samples[-1:]

    return train_samples, val_samples


def label_counts(samples: list[VideoSample]) -> dict[int, int]:
    """Count clips per class label in ``samples``."""
    counts: dict[int, int] = defaultdict(int)
    for _, label in samples:
        counts[int(label)] += 1
    return dict(counts)


def split_train_val_stratified(
    samples: list[VideoSample],
    val_ratio: float,
    seed: int,
) -> tuple[list[VideoSample], list[VideoSample]]:
    """Split ``samples`` into train and val with per-class holdout fractions.

    For each class, roughly ``val_ratio`` of its clips are assigned to validation.
    Classes with a single clip are kept entirely in train so holdout never drops
    a class from the eval set when more than one clip exists elsewhere.

    Args:
        samples: Full list of ``(video_dir, label)`` pairs.
        val_ratio: Fraction of each class reserved for validation.
        seed: RNG seed for shuffling within each class.

    Returns:
        Tuple ``(train_samples, val_samples)``.

    Raises:
        ValueError: If ``val_ratio`` is not in ``(0, 1)``.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}.")

    rng = random.Random(seed)
    by_label: dict[int, list[VideoSample]] = defaultdict(list)
    for item in samples:
        by_label[int(item[1])].append(item)

    train_out: list[VideoSample] = []
    val_out: list[VideoSample] = []

    for label in sorted(by_label):
        group = list(by_label[label])
        rng.shuffle(group)
        n = len(group)
        if n <= 1:
            train_out.extend(group)
            continue
        n_val = int(round(n * val_ratio))
        n_val = min(max(1, n_val), n - 1)
        val_out.extend(group[:n_val])
        train_out.extend(group[n_val:])

    rng.shuffle(train_out)
    rng.shuffle(val_out)
    return train_out, val_out
