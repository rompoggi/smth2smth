"""Tests for E6 extra-training-data collection in :mod:`smth2smth.pipelines.train`."""

from __future__ import annotations

from pathlib import Path

import pytest

from smth2smth.pipelines.train import _collect_extra_train_samples


def _make_clip(root: Path, class_name: str, video: str, n_frames: int = 4) -> None:
    """Create a ``root/class_name/video/frame_*.jpg`` clip (empty frame files)."""
    video_dir = root / class_name / video
    video_dir.mkdir(parents=True)
    for i in range(n_frames):
        (video_dir / f"frame_{i:03d}.jpg").touch()


def test_disabled_returns_empty(tmp_path):
    # Even with a valid dir present, use_extra=false must collect nothing.
    _make_clip(tmp_path, "000_Closing_something", "video_1")
    assert _collect_extra_train_samples(False, tmp_path) == []


def test_enabled_collects_clips(tmp_path):
    _make_clip(tmp_path, "000_Closing_something", "video_1")
    _make_clip(tmp_path, "002_Dropping_something_into_something", "video_2")
    samples = _collect_extra_train_samples(True, tmp_path)
    labels = sorted(label for _, label in samples)
    assert len(samples) == 2
    assert labels == [0, 2]  # class index parsed from the NNN_ prefix


def test_enabled_without_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="train_extra_dir"):
        _collect_extra_train_samples(True, None)


def test_enabled_missing_dir_raises(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        _collect_extra_train_samples(True, missing)
