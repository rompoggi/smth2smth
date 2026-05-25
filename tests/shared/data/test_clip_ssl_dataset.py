"""Tests for the clip-level SSL dataset used by V-JEPA pretraining."""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

from smth2smth.shared.data import ClipSSLDataset, collect_all_video_dirs


def _make_split_with_classes(
    root: Path, n_classes: int = 2, n_videos: int = 2, n_frames: int = 4
) -> None:
    for c in range(n_classes):
        for v in range(n_videos):
            video_dir = root / f"{c:03d}_class" / f"video_{c}_{v}"
            video_dir.mkdir(parents=True)
            for f in range(n_frames):
                img = Image.new("RGB", (16, 16), color=(f * 30, c * 30, v * 30))
                img.save(video_dir / f"frame_{f:03d}.jpg", "JPEG")


def _make_test_split(root: Path, n_videos: int = 2, n_frames: int = 4) -> None:
    for v in range(n_videos):
        video_dir = root / f"video_t_{v}"
        video_dir.mkdir(parents=True)
        for f in range(n_frames):
            img = Image.new("RGB", (16, 16), color=(f * 30, 50, v * 30))
            img.save(video_dir / f"frame_{f:03d}.jpg", "JPEG")


class TestCollectAllVideoDirs:
    def test_handles_class_split(self, tmp_path: Path) -> None:
        train = tmp_path / "train"
        _make_split_with_classes(train, n_classes=2, n_videos=2, n_frames=4)
        dirs = collect_all_video_dirs([train])
        assert len(dirs) == 4
        for p in dirs:
            assert p.is_dir()

    def test_handles_flat_test_split(self, tmp_path: Path) -> None:
        test = tmp_path / "test"
        _make_test_split(test, n_videos=3, n_frames=4)
        dirs = collect_all_video_dirs([test])
        assert len(dirs) == 3

    def test_dedupes_across_roots(self, tmp_path: Path) -> None:
        train = tmp_path / "train"
        _make_split_with_classes(train, n_classes=1, n_videos=2, n_frames=4)
        # Pass the same root twice -- should still return 2 unique entries.
        dirs = collect_all_video_dirs([train, train])
        assert len(dirs) == 2

    def test_missing_root_is_skipped(self, tmp_path: Path) -> None:
        train = tmp_path / "train"
        _make_split_with_classes(train, n_classes=1, n_videos=1, n_frames=4)
        dirs = collect_all_video_dirs([train, tmp_path / "missing"])
        assert len(dirs) == 1


class TestClipSSLDataset:
    def _make(self, tmp_path: Path, n_frames: int = 4) -> ClipSSLDataset:
        train = tmp_path / "train"
        _make_split_with_classes(train, n_classes=1, n_videos=2, n_frames=n_frames)
        dirs = collect_all_video_dirs([train])
        transform = T.Compose([T.Resize((16, 16)), T.ToTensor()])
        return ClipSSLDataset(video_dirs=dirs, num_frames=n_frames, transform=transform)

    def test_returns_correct_shape(self, tmp_path: Path) -> None:
        ds = self._make(tmp_path)
        sample = ds[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (4, 3, 16, 16)

    def test_length(self, tmp_path: Path) -> None:
        ds = self._make(tmp_path)
        assert len(ds) == 2

    def test_temporal_expand_interpolation(self, tmp_path: Path) -> None:
        train = tmp_path / "train"
        _make_split_with_classes(train, n_classes=1, n_videos=1, n_frames=8)
        dirs = collect_all_video_dirs([train])
        transform = T.Compose([T.Resize((16, 16)), T.ToTensor()])
        ds = ClipSSLDataset(
            video_dirs=dirs,
            num_frames=16,
            source_num_frames=4,
            temporal_expand_mode="interpolation",
            transform=transform,
        )
        sample = ds[0]
        assert sample.shape == (16, 3, 16, 16)

    def test_temporal_jitter_keeps_shape(self, tmp_path: Path) -> None:
        train = tmp_path / "train"
        _make_split_with_classes(train, n_classes=1, n_videos=1, n_frames=10)
        dirs = collect_all_video_dirs([train])
        transform = T.Compose([T.Resize((16, 16)), T.ToTensor()])
        ds = ClipSSLDataset(
            video_dirs=dirs, num_frames=4, transform=transform, temporal_jitter=1.0
        )
        for _ in range(5):
            sample = ds[0]
            assert sample.shape == (4, 3, 16, 16)
