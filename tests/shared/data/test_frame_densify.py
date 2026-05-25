"""Tests for 4->16 frame densification helpers."""

from __future__ import annotations

from PIL import Image

from smth2smth.shared.data.frame_densify import (
    ANCHOR_COUNT,
    DENSE_FRAME_COUNT,
    BlendInterpolator,
    duplicate_anchors_to_dense,
    interpolate_anchors_to_dense,
)
from smth2smth.shared.data.video_dataset import pick_frame_indices


def _solid(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (32, 32), color)


def test_duplicate_matches_pick_frame_indices() -> None:
    anchors = [_solid((i, 0, 0)) for i in range(ANCHOR_COUNT)]
    dense = duplicate_anchors_to_dense(anchors)
    indices = pick_frame_indices(ANCHOR_COUNT, DENSE_FRAME_COUNT)
    assert len(dense) == DENSE_FRAME_COUNT
    for col, anchor_idx in enumerate(indices):
        assert dense[col].getpixel((0, 0)) == anchors[anchor_idx].getpixel((0, 0))


def test_interpolate_produces_sixteen_frames() -> None:
    anchors = [_solid((0, i, 0)) for i in range(ANCHOR_COUNT)]
    dense = interpolate_anchors_to_dense(anchors, BlendInterpolator())
    assert len(dense) == DENSE_FRAME_COUNT
    assert dense[0].getpixel((0, 0)) == (0, 0, 0)
    assert dense[-1].getpixel((0, 0)) == (0, 3, 0)
