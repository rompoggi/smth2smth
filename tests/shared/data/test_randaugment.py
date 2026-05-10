"""Tests for ``smth2smth.shared.data.randaugment``."""

from __future__ import annotations

import random

import pytest
from PIL import Image

from smth2smth.shared.data.randaugment import (
    RA_FREE_OPS,
    RA_OPS,
    RandAugment,
)
from smth2smth.shared.data.transforms import build_transforms


def _seed(value: int = 0) -> None:
    random.seed(value)


def test_all_ops_preserve_image_size_and_mode() -> None:
    """Every op must return a PIL ``Image`` with the same size and 'RGB' mode."""
    img = Image.new("RGB", (40, 30), color=(120, 50, 200))
    for name, op in RA_OPS.items():
        out = op(img, level=15.0, magnitude_max=30.0)
        assert isinstance(out, Image.Image), f"{name} returned {type(out)}"
        assert out.size == img.size, f"{name} changed size: {out.size}"
        assert out.mode == "RGB", f"{name} changed mode to {out.mode}"


def test_n_zero_is_identity() -> None:
    """``n=0`` must return the input unchanged."""
    pipe = RandAugment(n=0, m=10.0, mode="spatial")
    img = Image.new("RGB", (24, 24), color=(10, 200, 30))
    out = pipe(img)
    assert isinstance(out, Image.Image)
    assert img.tobytes() == out.tobytes()


def test_clip_call_returns_list_of_correct_length() -> None:
    pipe = RandAugment(n=2, m=10.0, mode="temporal_plus")
    frames = [Image.new("RGB", (24, 24), color=(c, c, c)) for c in (10, 20, 30, 40)]
    out = pipe(frames)
    assert isinstance(out, list)
    assert len(out) == len(frames)
    assert all(isinstance(f, Image.Image) for f in out)
    assert all(f.size == (24, 24) for f in out)


@pytest.mark.parametrize("mode", ["spatial", "temporal", "temporal_plus", "mix"])
def test_magnitudes_endpoint_invariants(mode: str) -> None:
    pipe = RandAugment(n=1, m=10.0, mode=mode, magnitude_max=30.0)
    levels = pipe.magnitudes(num_frames=8)
    assert len(levels) == 8
    for lv in levels:
        assert -1e-6 <= lv <= 30.0 + 1e-6


def test_temporal_plus_is_symmetric_around_m() -> None:
    """For ``temporal_plus``, the per-frame magnitudes are symmetric around M."""
    _seed(123)
    pipe = RandAugment(n=1, m=10.0, mode="temporal_plus", magnitude_max=30.0)
    levels = pipe.magnitudes(num_frames=9)
    midpoint = 0.5 * (levels[0] + levels[-1])
    assert abs(midpoint - 10.0) < 1e-5
    diffs = [levels[i + 1] - levels[i] for i in range(len(levels) - 1)]
    assert max(diffs) - min(diffs) < 1e-5  # constant step (linear)


def test_spatial_mode_is_constant_across_frames() -> None:
    pipe = RandAugment(n=1, m=12.0, mode="spatial")
    levels = pipe.magnitudes(num_frames=6)
    assert all(abs(lv - 12.0) < 1e-9 for lv in levels)


def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError):
        RandAugment(n=1, m=10.0, mode="not_a_mode")


def test_invalid_op_raises() -> None:
    with pytest.raises(ValueError):
        RandAugment(n=1, m=10.0, ops=["rotate", "totally_made_up_op"])


def test_free_ops_are_magnitude_independent() -> None:
    """``identity``/``autocontrast``/``equalize`` ignore the magnitude argument."""
    img = Image.new("RGB", (40, 30), color=(100, 50, 25))
    for name in RA_FREE_OPS:
        op = RA_OPS[name]
        out_low = op(img, level=0.0, magnitude_max=30.0)
        out_high = op(img, level=30.0, magnitude_max=30.0)
        assert out_low.tobytes() == out_high.tobytes()


def test_build_transforms_with_randaugment_train_shape() -> None:
    """End-to-end: build_transforms with randaugment yields the expected tensor shape."""
    augment = {
        "name": "randaugment_t",
        "random_horizontal_flip": True,
        "random_crop": True,
        "crop_padding": 16,
        "color_jitter": False,
        "sync_across_frames": True,
        "randaugment": {
            "enabled": True,
            "n": 2,
            "m": 9.0,
            "mode": "temporal_plus",
            "magnitude_max": 30.0,
            "ops": None,
        },
    }
    pipe = build_transforms(
        image_size=32, is_training=True, use_imagenet_norm=True, augment=augment
    )
    frames = [Image.new("RGB", (60, 80), color=(c, c, c)) for c in (10, 50, 90, 130)]
    out = pipe(frames)
    assert isinstance(out, list)
    assert len(out) == 4
    for tensor in out:
        assert tensor.shape == (3, 32, 32)


def test_build_transforms_eval_skips_randaugment() -> None:
    """Eval pipeline must be deterministic even when randaugment is enabled."""
    augment = {
        "random_horizontal_flip": True,
        "random_crop": True,
        "crop_padding": 16,
        "sync_across_frames": True,
        "randaugment": {
            "enabled": True,
            "n": 3,
            "m": 15.0,
            "mode": "temporal_plus",
        },
    }
    pipe = build_transforms(
        image_size=32, is_training=False, use_imagenet_norm=True, augment=augment
    )
    img = Image.new("RGB", (60, 60), color=(40, 80, 120))
    import torch

    assert torch.allclose(pipe(img), pipe(img))


def test_disabled_randaugment_matches_no_randaugment() -> None:
    """``randaugment.enabled=False`` must produce the same tensor as a config without it."""
    import torch

    base = {
        "random_horizontal_flip": False,
        "random_crop": False,
        "color_jitter": False,
    }
    with_disabled_ra = dict(
        base, randaugment={"enabled": False, "n": 2, "m": 9.0, "mode": "spatial"}
    )

    pipe_a = build_transforms(image_size=32, is_training=True, use_imagenet_norm=True, augment=base)
    pipe_b = build_transforms(
        image_size=32, is_training=True, use_imagenet_norm=True, augment=with_disabled_ra
    )
    img = Image.new("RGB", (32, 32), color=(123, 45, 67))
    assert torch.allclose(pipe_a(img), pipe_b(img))
