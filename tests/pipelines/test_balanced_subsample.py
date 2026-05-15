"""Unit tests for :func:`smth2smth.pipelines.train._balanced_per_class_subsample`."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from smth2smth.pipelines.train import _balanced_per_class_subsample


def _fake_samples(per_class: dict[int, int]) -> list[tuple[Path, int]]:
    """Build a flat ``(video_dir, class_index)`` list from a per-class size map."""
    out: list[tuple[Path, int]] = []
    for cls, n in per_class.items():
        for i in range(n):
            out.append((Path(f"/tmp/class_{cls:03d}/video_{i:05d}"), cls))
    return out


def test_keeps_all_when_under_threshold() -> None:
    """If every class has fewer than ``max_per_class`` samples, nothing is dropped."""
    samples = _fake_samples({0: 3, 1: 2, 2: 5})
    out = _balanced_per_class_subsample(samples, max_per_class=10, seed=0)
    assert sorted(out) == sorted(samples)


def test_caps_each_class_at_max() -> None:
    """Per-class counts are clamped to ``max_per_class`` exactly."""
    samples = _fake_samples({0: 50, 1: 20, 2: 5, 3: 100})
    out = _balanced_per_class_subsample(samples, max_per_class=10, seed=0)
    counts = Counter(lbl for _, lbl in out)
    assert counts[0] == 10
    assert counts[1] == 10
    assert counts[2] == 5
    assert counts[3] == 10


def test_deterministic_with_same_seed() -> None:
    """Same seed -> same subset (order preserved)."""
    samples = _fake_samples({0: 50, 1: 50})
    a = _balanced_per_class_subsample(samples, max_per_class=10, seed=42)
    b = _balanced_per_class_subsample(samples, max_per_class=10, seed=42)
    assert a == b


def test_different_seed_gives_different_subset() -> None:
    """Different seeds should usually produce different selections."""
    samples = _fake_samples({0: 100})
    a = _balanced_per_class_subsample(samples, max_per_class=10, seed=0)
    b = _balanced_per_class_subsample(samples, max_per_class=10, seed=1)
    assert set(p for p, _ in a) != set(p for p, _ in b)


def test_zero_or_negative_max_returns_empty() -> None:
    """Non-positive ``max_per_class`` produces an empty list (no surprises)."""
    samples = _fake_samples({0: 10, 1: 10})
    assert _balanced_per_class_subsample(samples, max_per_class=0, seed=0) == []
    assert _balanced_per_class_subsample(samples, max_per_class=-5, seed=0) == []


def test_preserves_original_ordering() -> None:
    """Surviving samples appear in the same order they had in the input list."""
    samples = _fake_samples({0: 30, 1: 30})
    out = _balanced_per_class_subsample(samples, max_per_class=5, seed=7)
    original_index = {sample: idx for idx, sample in enumerate(samples)}
    indices = [original_index[s] for s in out]
    assert indices == sorted(indices)
