"""Tests for the still-frames dataset used by SSL pretraining."""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

from smth2smth.shared.data import (
    MultiViewStillFramesDataset,
    collect_all_frame_paths,
)


def _make_split_with_classes(root: Path, n_classes: int = 2, n_videos: int = 2, n_frames: int = 3) -> None:
    for c in range(n_classes):
        for v in range(n_videos):
            video_dir = root / f"{c:03d}_class" / f"video_{c}_{v}"
            video_dir.mkdir(parents=True)
            for f in range(n_frames):
                img = Image.new("RGB", (16, 16), color=(f * 30, c * 30, v * 30))
                img.save(video_dir / f"frame_{f:03d}.jpg", "JPEG")


def _make_test_split(root: Path, n_videos: int = 2, n_frames: int = 3) -> None:
    for v in range(n_videos):
        video_dir = root / f"video_t_{v}"
        video_dir.mkdir(parents=True)
        for f in range(n_frames):
            img = Image.new("RGB", (16, 16), color=(f * 30, 50, v * 30))
            img.save(video_dir / f"frame_{f:03d}.jpg", "JPEG")


class TestCollectAllFramePaths:
    """Frame discovery across train/val/test layouts."""

    def test_handles_class_split(self, tmp_path: Path) -> None:
        train = tmp_path / "train"
        _make_split_with_classes(train, n_classes=2, n_videos=2, n_frames=3)
        paths = collect_all_frame_paths([train])
        # 2 classes * 2 videos * 3 frames = 12.
        assert len(paths) == 12

    def test_handles_test_layout(self, tmp_path: Path) -> None:
        test = tmp_path / "test"
        _make_test_split(test, n_videos=3, n_frames=4)
        paths = collect_all_frame_paths([test])
        assert len(paths) == 12

    def test_combines_multiple_roots_and_dedups(self, tmp_path: Path) -> None:
        train = tmp_path / "train"
        val = tmp_path / "val"
        _make_split_with_classes(train, n_classes=1, n_videos=1, n_frames=2)
        _make_split_with_classes(val, n_classes=1, n_videos=1, n_frames=2)
        paths = collect_all_frame_paths([train, val])
        assert len(paths) == 4

    def test_missing_root_is_skipped(self, tmp_path: Path) -> None:
        paths = collect_all_frame_paths([tmp_path / "does_not_exist"])
        assert paths == []


class TestMultiViewStillFramesDataset:
    """Per-sample multi-view sampling for DINO."""

    def _sample_paths(self, tmp_path: Path) -> list[Path]:
        train = tmp_path / "train"
        _make_split_with_classes(train, n_classes=2, n_videos=2, n_frames=2)
        return collect_all_frame_paths([train])

    def test_two_globals_no_locals(self, tmp_path: Path) -> None:
        paths = self._sample_paths(tmp_path)
        ds = MultiViewStillFramesDataset(
            frame_paths=paths,
            global_transform=T.Compose([T.Resize(8), T.ToTensor()]),
            local_transform=None,
            num_local_views=0,
        )
        assert len(ds) == len(paths)
        item = ds[0]
        assert len(item["global_views"]) == 2
        assert len(item["local_views"]) == 0
        assert isinstance(item["global_views"][0], torch.Tensor)
        assert item["global_views"][0].shape[0] == 3  # CHW

    def test_two_globals_plus_locals(self, tmp_path: Path) -> None:
        paths = self._sample_paths(tmp_path)
        ds = MultiViewStillFramesDataset(
            frame_paths=paths,
            global_transform=T.Compose([T.Resize(8), T.ToTensor()]),
            local_transform=T.Compose([T.Resize(4), T.ToTensor()]),
            num_local_views=3,
        )
        item = ds[0]
        assert len(item["global_views"]) == 2
        assert len(item["local_views"]) == 3
        assert item["global_views"][0].shape[-1] == 8
        assert item["local_views"][0].shape[-1] == 4
