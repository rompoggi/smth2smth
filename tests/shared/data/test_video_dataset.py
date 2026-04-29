"""Tests for ``smth2smth.shared.data.video_dataset``."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torchvision import transforms

from smth2smth.shared.data.video_dataset import (
    VideoFrameDataset,
    collect_video_samples,
    parse_class_index,
    pick_frame_indices,
)


def _identity_transform() -> transforms.Compose:
    """A minimal PIL -> tensor transform with a fixed output spatial size."""
    return transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()])


class TestParseClassIndex:
    """parse_class_index extracts numeric prefixes only when present."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("000_FooBar", 0),
            ("017_PutSomething", 17),
            ("9_Misc", 9),
            ("042_Class", 42),
        ],
    )
    def test_returns_prefix_when_present(self, name: str, expected: int) -> None:
        assert parse_class_index(name) == expected

    @pytest.mark.parametrize(
        "name",
        ["alpha", "no_prefix", "_underscore_first", "abc_123_x"],
    )
    def test_returns_none_when_no_numeric_prefix(self, name: str) -> None:
        assert parse_class_index(name) is None


class TestCollectVideoSamples:
    """collect_video_samples enumerates samples and resolves class indices."""

    def test_numeric_prefix_class_indices(self, numeric_prefix_dataset: Path) -> None:
        samples = collect_video_samples(numeric_prefix_dataset)
        labels = sorted({label for _, label in samples})
        assert labels == [17, 42]
        assert len(samples) == 3  # 2 videos in 017, 1 in 042

    def test_alphabetical_fallback_indices(self, alphabetical_dataset: Path) -> None:
        samples = collect_video_samples(alphabetical_dataset)
        # alpha -> 0, beta -> 1 (alphabetical order, no numeric prefix)
        assert sorted({label for _, label in samples}) == [0, 1]
        assert len(samples) == 2

    def test_skips_empty_class_folders(self, empty_class_dataset: Path) -> None:
        samples = collect_video_samples(empty_class_dataset)
        # Only one usable video; the empty class folder must be silently skipped.
        assert len(samples) == 1
        _, label = samples[0]
        assert label == 0

    def test_missing_root_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            collect_video_samples(tmp_path / "does_not_exist")

    def test_root_with_no_videos_raises(self, tmp_path: Path) -> None:
        empty_root = tmp_path / "empty_root"
        empty_root.mkdir()
        with pytest.raises(RuntimeError):
            collect_video_samples(empty_root)


class TestPickFrameIndices:
    """pick_frame_indices selects evenly spaced indices and handles edge cases."""

    def test_basic_evenly_spaced(self) -> None:
        # linspace(0, 9, 5) -> [0, 2.25, 4.5, 6.75, 9]; Python rounds 4.5 to 4 (banker's rounding).
        indices = pick_frame_indices(num_available=10, num_frames=5)
        assert indices[0] == 0
        assert indices[-1] == 9
        assert len(indices) == 5
        assert indices == sorted(indices)

    def test_short_video_repeats_last_frame(self) -> None:
        # 2 frames available, want 8 -> indices repeat with rounding.
        indices = pick_frame_indices(num_available=2, num_frames=8)
        assert len(indices) == 8
        assert min(indices) == 0
        assert max(indices) == 1
        # Must be non-decreasing because linspace is monotone.
        assert indices == sorted(indices)

    def test_single_frame_repeats(self) -> None:
        assert pick_frame_indices(num_available=1, num_frames=4) == [0, 0, 0, 0]

    @pytest.mark.parametrize("num_available,num_frames", [(0, 4), (5, 0), (-1, 4), (5, -2)])
    def test_invalid_inputs_raise(self, num_available: int, num_frames: int) -> None:
        with pytest.raises(ValueError):
            pick_frame_indices(num_available=num_available, num_frames=num_frames)


class TestVideoFrameDataset:
    """End-to-end behavior of the dataset class."""

    def test_getitem_returns_expected_tensor_shape(self, numeric_prefix_dataset: Path) -> None:
        num_frames = 4
        dataset = VideoFrameDataset(
            root_dir=numeric_prefix_dataset,
            num_frames=num_frames,
            transform=_identity_transform(),
        )
        assert len(dataset) == 3
        video_tensor, label = dataset[0]
        assert isinstance(video_tensor, torch.Tensor)
        assert video_tensor.shape == (num_frames, 3, 16, 16)
        assert video_tensor.dtype == torch.float32
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.long
        assert label.ndim == 0

    def test_getitem_handles_short_videos(self, short_video_dataset: Path) -> None:
        num_frames = 8
        dataset = VideoFrameDataset(
            root_dir=short_video_dataset,
            num_frames=num_frames,
            transform=_identity_transform(),
        )
        video_tensor, _ = dataset[0]
        assert video_tensor.shape == (num_frames, 3, 16, 16)

    def test_explicit_sample_list_overrides_scan(self, numeric_prefix_dataset: Path) -> None:
        all_samples = collect_video_samples(numeric_prefix_dataset)
        # Keep only the first sample to verify sample_list takes precedence.
        subset = all_samples[:1]
        dataset = VideoFrameDataset(
            root_dir=numeric_prefix_dataset,
            num_frames=4,
            transform=_identity_transform(),
            sample_list=subset,
        )
        assert len(dataset) == 1
