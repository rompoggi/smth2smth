"""Train/val split helpers."""

from __future__ import annotations

import random
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
