"""Per-frame image transforms.

Adapted from the professor baseline at
``external/prof_baseline/src/utils.py::build_transforms``.

The augmentation policy is configurable via the ``augment`` Hydra config group
(see ``configs/augment/{none,strong}.yaml``). Calls without an ``augment``
argument fall back to the legacy behavior (Resize + optional RandomHorizontalFlip)
so that existing call sites and the smoke test keep working.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torchvision import transforms

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_SYMMETRIC_MEAN = (0.5, 0.5, 0.5)
_SYMMETRIC_STD = (0.5, 0.5, 0.5)


def _augment_get(augment: Mapping[str, Any] | None, key: str, default: Any) -> Any:
    """Fetch ``key`` from a dict-like augment config, returning ``default`` if missing.

    Tolerates both plain dicts and OmegaConf ``DictConfig`` objects (which both
    implement ``Mapping`` but ``DictConfig`` may also raise on missing keys when
    struct mode is enabled).
    """
    if augment is None:
        return default
    try:
        value = augment.get(key, default)  # type: ignore[union-attr]
    except Exception:
        return default
    return default if value is None else value


def build_transforms(
    image_size: int = 224,
    is_training: bool = True,
    use_imagenet_norm: bool = True,
    augment: Mapping[str, Any] | None = None,
) -> transforms.Compose:
    """Build a per-frame torchvision transform pipeline.

    The returned pipeline maps a PIL ``Image`` in RGB to a normalized
    ``(C, H, W)`` float tensor of shape ``(3, image_size, image_size)``.

    Args:
        image_size: Target square size for the resized frame.
        is_training: When ``True`` enables the random augmentations specified
            in ``augment`` (or the legacy hflip-only default when ``augment``
            is ``None``).
        use_imagenet_norm: When ``True`` uses ImageNet mean/std (recommended
            for pretrained backbones). Otherwise uses symmetric ``(0.5, 0.5, 0.5)``
            normalization.
        augment: Optional mapping with augmentation knobs:

            - ``random_horizontal_flip`` (bool, default ``True``)
            - ``random_crop`` (bool, default ``False``) -- when ``True``, the
              image is first resized to ``image_size + crop_padding`` and then
              cropped to ``image_size`` (random for training, center for eval).
            - ``crop_padding`` (int, default ``0``) -- padding added before
              cropping; only used when ``random_crop`` is ``True``.
            - ``color_jitter`` (bool, default ``False``) -- training only.
            - ``color_jitter_{brightness,contrast,saturation,hue}`` (float).

            When ``None``, the legacy behavior is used: Resize + (optional)
            RandomHorizontalFlip.

    Returns:
        A ``torchvision.transforms.Compose`` pipeline.
    """
    normalize = (
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
        if use_imagenet_norm
        else transforms.Normalize(mean=_SYMMETRIC_MEAN, std=_SYMMETRIC_STD)
    )

    use_random_crop = bool(_augment_get(augment, "random_crop", False))
    crop_padding = int(_augment_get(augment, "crop_padding", 0))
    use_hflip = bool(_augment_get(augment, "random_horizontal_flip", True))
    use_color_jitter = bool(_augment_get(augment, "color_jitter", False))

    if use_random_crop and crop_padding > 0:
        resize_size = image_size + crop_padding
    else:
        resize_size = image_size
    steps: list = [transforms.Resize((resize_size, resize_size))]

    if is_training:
        if use_random_crop:
            steps.append(transforms.RandomCrop((image_size, image_size)))
        if use_hflip:
            steps.append(transforms.RandomHorizontalFlip())
        if use_color_jitter:
            steps.append(
                transforms.ColorJitter(
                    brightness=float(_augment_get(augment, "color_jitter_brightness", 0.0)),
                    contrast=float(_augment_get(augment, "color_jitter_contrast", 0.0)),
                    saturation=float(_augment_get(augment, "color_jitter_saturation", 0.0)),
                    hue=float(_augment_get(augment, "color_jitter_hue", 0.0)),
                )
            )
    else:
        if use_random_crop:
            steps.append(transforms.CenterCrop((image_size, image_size)))

    steps.extend([transforms.ToTensor(), normalize])
    return transforms.Compose(steps)
