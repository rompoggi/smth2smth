"""Tests for frozen holdout_clean manifest."""

from __future__ import annotations

from pathlib import Path

from smth2smth.ensemble.holdout import (
    apply_holdout_manifest,
    build_and_write_holdout_clean,
    read_holdout_keys,
)
from smth2smth.shared.data import collect_video_samples
from smth2smth.shared.utils.splits import split_train_val_stratified


def test_build_holdout_clean_reproducible(tmp_path: Path) -> None:
    """Manifest round-trip matches stratified split with seed 42."""
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    for split, n_classes, per_class in (("train", 3, 5), ("val", 3, 4)):
        root = tmp_path / split
        for c in range(n_classes):
            cls = root / f"{c:03d}_Class_{c}"
            cls.mkdir(parents=True)
            for i in range(per_class):
                clip = cls / f"clip_{c}_{i}"
                clip.mkdir()
                (clip / "frame_000.jpg").write_bytes(b"x")

    out = tmp_path / "holdout_clean.json"
    report = build_and_write_holdout_clean(
        train_dir=train_dir,
        val_dir=val_dir,
        output_path=out,
        holdout_ratio=0.25,
        split_seed=42,
    )
    assert report["n_holdout"] > 0
    val_all = collect_video_samples(val_dir)
    _, holdout2 = split_train_val_stratified(val_all, val_ratio=0.25, seed=42)
    keys_manifest = read_holdout_keys(out)
    keys_split = {p.name for p, _ in holdout2}
    assert keys_manifest == keys_split

    val_for_train, holdout = apply_holdout_manifest(val_all, out)
    assert len(val_for_train) + len(holdout) == len(val_all)
    assert {p.name for p, _ in holdout} == keys_manifest
