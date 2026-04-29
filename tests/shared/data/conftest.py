"""Shared fixtures for data-layer tests."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest
from PIL import Image


def _write_dummy_jpeg(path: Path, color: tuple[int, int, int] = (128, 64, 32)) -> None:
    """Write a tiny solid-color JPEG to ``path`` (parent must already exist)."""
    image = Image.new("RGB", (32, 32), color)
    image.save(path, format="JPEG", quality=70)


def _make_video_dir(
    parent: Path,
    video_name: str,
    num_frames: int,
    color: tuple[int, int, int] = (128, 64, 32),
) -> Path:
    """Create a video directory with ``num_frames`` JPG frames."""
    video_dir = parent / video_name
    video_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_frames):
        _write_dummy_jpeg(video_dir / f"frame_{i:03d}.jpg", color=color)
    return video_dir


@pytest.fixture
def numeric_prefix_dataset(tmp_path: Path) -> Path:
    """Build a tiny dataset with class folders that have numeric prefixes.

    Layout::

        root/
          017_ClassA/
            video_1/  (5 frames)
            video_2/  (5 frames)
          042_ClassB/
            video_1/  (5 frames)
    """
    root = tmp_path / "root_numeric"
    class_a = root / "017_ClassA"
    class_b = root / "042_ClassB"
    _make_video_dir(class_a, "video_1", num_frames=5)
    _make_video_dir(class_a, "video_2", num_frames=5)
    _make_video_dir(class_b, "video_1", num_frames=5)
    return root


@pytest.fixture
def alphabetical_dataset(tmp_path: Path) -> Path:
    """Build a tiny dataset whose class folders have no numeric prefix."""
    root = tmp_path / "root_alpha"
    _make_video_dir(root / "alpha", "video_1", num_frames=4)
    _make_video_dir(root / "beta", "video_1", num_frames=4)
    return root


@pytest.fixture
def short_video_dataset(tmp_path: Path) -> Path:
    """Dataset with one video that has fewer frames than typical T."""
    root = tmp_path / "root_short"
    _make_video_dir(root / "000_OnlyClass", "video_short", num_frames=2)
    return root


@pytest.fixture
def empty_class_dataset(tmp_path: Path) -> Path:
    """Dataset where one class folder has no usable video sub-folders."""
    root = tmp_path / "root_with_empty_class"
    # Useful class with frames.
    _make_video_dir(root / "000_Useful", "video_1", num_frames=3)
    # Empty class (no video subdirs at all).
    (root / "001_Empty").mkdir(parents=True)
    return root


def _all_dataset_roots(*roots: Path) -> Iterable[Path]:
    """Helper used in parametrized tests."""
    return roots
