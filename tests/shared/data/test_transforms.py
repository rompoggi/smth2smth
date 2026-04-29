"""Tests for ``smth2smth.shared.data.transforms``."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from smth2smth.shared.data.transforms import build_transforms


@pytest.mark.parametrize("is_training", [True, False])
@pytest.mark.parametrize("use_imagenet_norm", [True, False])
def test_build_transforms_output_shape_and_type(is_training: bool, use_imagenet_norm: bool) -> None:
    image_size = 64
    pipeline = build_transforms(
        image_size=image_size,
        is_training=is_training,
        use_imagenet_norm=use_imagenet_norm,
    )
    image = Image.new("RGB", (100, 80), color=(10, 20, 30))
    tensor = pipeline(image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, image_size, image_size)
    assert tensor.dtype == torch.float32


def test_eval_pipeline_is_deterministic() -> None:
    pipeline = build_transforms(image_size=32, is_training=False, use_imagenet_norm=True)
    image = Image.new("RGB", (50, 50), color=(64, 64, 64))
    out1 = pipeline(image)
    out2 = pipeline(image)
    assert torch.allclose(out1, out2)


def test_imagenet_norm_changes_output_distribution() -> None:
    """Sanity check: ImageNet norm and symmetric norm are not identical."""
    image = Image.new("RGB", (32, 32), color=(128, 128, 128))
    imagenet = build_transforms(image_size=32, is_training=False, use_imagenet_norm=True)
    symmetric = build_transforms(image_size=32, is_training=False, use_imagenet_norm=False)
    assert not torch.allclose(imagenet(image), symmetric(image))


def test_strong_augment_train_and_eval_shapes_match() -> None:
    """The strong augment policy must produce identical tensor shapes for train/eval."""
    image_size = 32
    augment = {
        "name": "strong",
        "random_horizontal_flip": True,
        "random_crop": True,
        "crop_padding": 16,
        "color_jitter": True,
        "color_jitter_brightness": 0.2,
        "color_jitter_contrast": 0.2,
        "color_jitter_saturation": 0.2,
        "color_jitter_hue": 0.0,
    }
    train_pipe = build_transforms(
        image_size=image_size, is_training=True, use_imagenet_norm=True, augment=augment
    )
    eval_pipe = build_transforms(
        image_size=image_size, is_training=False, use_imagenet_norm=True, augment=augment
    )
    image = Image.new("RGB", (80, 60), color=(123, 45, 67))
    train_tensor = train_pipe(image)
    eval_tensor = eval_pipe(image)
    assert train_tensor.shape == (3, image_size, image_size)
    assert eval_tensor.shape == (3, image_size, image_size)


def test_eval_pipeline_with_strong_augment_is_deterministic() -> None:
    """The eval branch of the strong policy uses CenterCrop, so it must be deterministic."""
    augment = {"random_crop": True, "crop_padding": 16, "color_jitter": True}
    pipe = build_transforms(
        image_size=32, is_training=False, use_imagenet_norm=True, augment=augment
    )
    image = Image.new("RGB", (80, 60), color=(40, 80, 120))
    assert torch.allclose(pipe(image), pipe(image))


def test_augment_none_matches_legacy_default() -> None:
    """The 'none' augment policy should match the legacy (no-augment) call."""
    augment_none = {
        "random_horizontal_flip": True,
        "random_crop": False,
        "color_jitter": False,
    }
    legacy = build_transforms(image_size=32, is_training=False, use_imagenet_norm=True)
    with_none = build_transforms(
        image_size=32, is_training=False, use_imagenet_norm=True, augment=augment_none
    )
    image = Image.new("RGB", (40, 40), color=(10, 10, 10))
    assert torch.allclose(legacy(image), with_none(image))
