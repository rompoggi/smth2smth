"""Official-val stratified holdout split (matches ``train.py``)."""

from __future__ import annotations

import json
from pathlib import Path

from smth2smth.shared.data import collect_video_samples
from smth2smth.shared.utils.splits import VideoSample, split_train_val_stratified


def build_official_val_holdout(
    val_dir: Path,
    *,
    holdout_ratio: float = 0.1,
    split_seed: int = 42,
) -> list[VideoSample]:
    """Return the stratified holdout val clips used for Mix fitting.

    Uses the same ``split_train_val_stratified`` call as ``train.py`` when
    ``official_val_holdout_ratio > 0``. The split RNG seed is ``split_seed``
    (typically 42), **not** the per-run training seed.

    Args:
        val_dir: Official validation root (``data/val``).
        holdout_ratio: Fraction held out per class (default 0.1).
        split_seed: RNG seed for the stratified split.

    Returns:
        List of ``(video_dir, label)`` holdout samples.
    """
    val_dir = val_dir.resolve()
    all_val = collect_video_samples(val_dir)
    _train_part, holdout = split_train_val_stratified(
        all_val, val_ratio=holdout_ratio, seed=split_seed
    )
    return holdout


def sample_key(video_dir: Path) -> str:
    """Stable string id for a clip (folder name under val)."""
    return video_dir.name


def write_holdout_manifest(holdout: list[VideoSample], path: Path) -> None:
    """Persist holdout clip ids and labels for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "keys": [sample_key(vd) for vd, _ in holdout],
        "labels": [int(lab) for _, lab in holdout],
        "paths": [str(vd.resolve()) for vd, _ in holdout],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_holdout_manifest(path: Path) -> tuple[list[str], list[int]]:
    """Load keys and labels written by :func:`write_holdout_manifest`."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["keys"]), [int(x) for x in data["labels"]]
