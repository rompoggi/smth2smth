"""Tests for Track A temporal reversal + opposite-label augmentation."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torchvision import transforms

from smth2smth.shared.data.temporal_pair_augment import (
    TRACK_A_TEMPORAL_REVERSAL_PAIRS,
    build_track_a_temporal_reversal_map,
)
from smth2smth.shared.data.video_dataset import VideoFrameDataset


def _solid_jpeg(path: Path, rgb: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), rgb).save(path, format="JPEG", quality=90)


def test_pair_map_is_symmetric() -> None:
    m = build_track_a_temporal_reversal_map()
    for a, b in TRACK_A_TEMPORAL_REVERSAL_PAIRS:
        assert m[int(a)] == int(b)
        assert m[int(b)] == int(a)
    assert len(m) == 2 * len(TRACK_A_TEMPORAL_REVERSAL_PAIRS)


def test_temporal_reversal_reorders_frames_and_swaps_label(tmp_path: Path) -> None:
    video_dir = tmp_path / "008_Moving_something_down" / "video_0"
    _solid_jpeg(video_dir / "frame_000.jpg", (255, 0, 0))
    _solid_jpeg(video_dir / "frame_001.jpg", (0, 255, 0))
    _solid_jpeg(video_dir / "frame_002.jpg", (0, 0, 255))

    tfm = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])
    pair_map = build_track_a_temporal_reversal_map()
    sample_list = [(video_dir, 8)]

    ds_on = VideoFrameDataset(
        root_dir=tmp_path,
        num_frames=3,
        transform=tfm,
        sample_list=sample_list,
        temporal_reversal_pair_to_opposite=pair_map,
        temporal_reversal_prob=1.0,
    )
    vid_on, y_on = ds_on[0]
    assert int(y_on) == 9
    assert vid_on.shape == (3, 3, 8, 8)
    assert vid_on[0, 2].mean() > 0.95 and vid_on[0, 0].mean() < 0.1 and vid_on[0, 1].mean() < 0.1

    ds_off = VideoFrameDataset(
        root_dir=tmp_path,
        num_frames=3,
        transform=tfm,
        sample_list=sample_list,
        temporal_reversal_pair_to_opposite=pair_map,
        temporal_reversal_prob=0.0,
    )
    vid_off, y_off = ds_off[0]
    assert int(y_off) == 8
    assert vid_off[0, 0].mean() > 0.95


def test_temporal_reversal_skips_unpaired_labels_even_at_prob_one(tmp_path: Path) -> None:
    video_dir = tmp_path / "017_Unpaired" / "video_0"
    _solid_jpeg(video_dir / "frame_000.jpg", (255, 0, 0))
    _solid_jpeg(video_dir / "frame_001.jpg", (0, 255, 0))

    tfm = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])
    pair_map = build_track_a_temporal_reversal_map()
    sample_list = [(video_dir, 17)]

    ds = VideoFrameDataset(
        root_dir=tmp_path,
        num_frames=2,
        transform=tfm,
        sample_list=sample_list,
        temporal_reversal_pair_to_opposite=pair_map,
        temporal_reversal_prob=1.0,
    )
    _, y = ds[0]
    assert int(y) == 17
