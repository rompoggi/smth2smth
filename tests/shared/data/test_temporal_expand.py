"""Tests for 4->16 temporal frame expansion."""

from __future__ import annotations

import pytest
from PIL import Image

from smth2smth.shared.data.temporal_expand import expand_temporal_frames


def _solid(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (8, 8), color=color)


class TestExpandTemporalFrames:
    def test_replication_four_to_sixteen(self) -> None:
        frames = [_solid((i, 0, 0)) for i in range(4)]
        out = expand_temporal_frames(frames, target_num_frames=16, mode="replication")
        assert len(out) == 16
        assert out[0] is frames[0]
        assert out[3] is frames[0]
        assert out[4] is frames[1]
        assert out[15] is frames[3]

    def test_interpolation_endpoints(self) -> None:
        frames = [_solid((0, 0, 0)), _solid((255, 0, 0))]
        out = expand_temporal_frames(frames, target_num_frames=4, mode="interpolation")
        assert len(out) == 4
        assert out[0] is frames[0]
        assert out[-1] is frames[-1]

    def test_noop_when_same_length(self) -> None:
        frames = [_solid((1, 2, 3)), _solid((4, 5, 6))]
        out = expand_temporal_frames(frames, target_num_frames=2, mode="interpolation")
        assert out == frames

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            expand_temporal_frames([_solid((0, 0, 0))], target_num_frames=4, mode="bad")
