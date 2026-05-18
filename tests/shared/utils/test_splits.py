"""Tests for dataset split helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from smth2smth.shared.utils.splits import label_counts, split_train_val_stratified


def _make_samples(n_per_class: int, n_classes: int = 5) -> list[tuple[Path, int]]:
    out: list[tuple[Path, int]] = []
    for label in range(n_classes):
        for i in range(n_per_class):
            out.append((Path(f"c{label}/v{i}"), label))
    return out


def test_split_train_val_stratified_ratio_and_disjoint() -> None:
    samples = _make_samples(n_per_class=20, n_classes=8)
    train, val = split_train_val_stratified(samples, val_ratio=0.1, seed=42)
    assert len(train) + len(val) == len(samples)
    train_set = set(train)
    val_set = set(val)
    assert train_set.isdisjoint(val_set)
    assert len(val) == pytest.approx(0.1 * len(samples), rel=0.05)


def test_split_train_val_stratified_per_class_fraction() -> None:
    samples = _make_samples(n_per_class=100, n_classes=33)
    train, val = split_train_val_stratified(samples, val_ratio=0.1, seed=7)
    full = label_counts(samples)
    hold = label_counts(val)
    for label, n_full in full.items():
        n_hold = hold.get(label, 0)
        assert n_hold >= 1
        assert n_hold <= n_full - 1
        assert n_hold == pytest.approx(0.1 * n_full, rel=0.25)


def test_split_train_val_stratified_reproducible() -> None:
    samples = _make_samples(n_per_class=10, n_classes=4)
    a_train, a_val = split_train_val_stratified(samples, val_ratio=0.2, seed=0)
    b_train, b_val = split_train_val_stratified(samples, val_ratio=0.2, seed=0)
    assert a_train == b_train
    assert a_val == b_val


def test_split_train_val_stratified_invalid_ratio() -> None:
    with pytest.raises(ValueError, match="val_ratio"):
        split_train_val_stratified([], val_ratio=0.0, seed=0)
