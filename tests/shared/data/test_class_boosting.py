"""Tests for deterministic class boosting (paired-verb duplicate train rows)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torchvision import transforms

from smth2smth.shared.data.temporal_pair_augment import (
    build_track_a_temporal_reversal_map,
    expand_train_samples_for_class_boosting,
)
from smth2smth.shared.data.video_dataset import VideoFrameDataset


def _jpeg(path: Path, rgb: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), rgb).save(path, format="JPEG", quality=90)


def test_expand_doubles_only_paired_classes() -> None:
    pair = build_track_a_temporal_reversal_map()
    samples = [
        (Path("/a/v1"), 8),
        (Path("/a/v2"), 17),
    ]
    out = expand_train_samples_for_class_boosting(samples, pair)
    assert len(out) == 3
    assert out[0] == (Path("/a/v1"), 8, False)
    assert out[1] == (Path("/a/v1"), 9, True)
    assert out[2] == (Path("/a/v2"), 17)


def test_video_frame_dataset_class_boost_row_reverses_frames(tmp_path: Path) -> None:
    video_dir = tmp_path / "008_Moving_something_down" / "v0"
    _jpeg(video_dir / "frame_000.jpg", (255, 0, 0))
    _jpeg(video_dir / "frame_001.jpg", (0, 255, 0))
    _jpeg(video_dir / "frame_002.jpg", (0, 0, 255))

    tfm = transforms.Compose([transforms.Resize((4, 4)), transforms.ToTensor()])
    sample_list = [
        (video_dir, 8, False),
        (video_dir, 9, True),
    ]
    ds = VideoFrameDataset(
        root_dir=tmp_path,
        num_frames=3,
        transform=tfm,
        sample_list=sample_list,
    )
    x0, y0 = ds[0]
    x1, y1 = ds[1]
    assert int(y0) == 8 and int(y1) == 9
    assert x0[0, 0].mean() > 0.9 and x0[2, 0].mean() < 0.1
    assert x1[0, 0].mean() < 0.1 and x1[2, 0].mean() > 0.9
