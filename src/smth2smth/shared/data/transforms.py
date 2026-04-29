"""Frame/clip transforms.

Adapted from the professor baseline at
``external/prof_baseline/src/utils.py::build_transforms``.

The augmentation policy is configurable via the ``augment`` Hydra config group
(see ``configs/augment/{none,strong}.yaml``). Calls without an ``augment``
argument fall back to the legacy behavior (Resize + optional RandomHorizontalFlip)
so that existing call sites and the smoke test keep working.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import ColorJitter, Normalize
from torchvision.transforms import functional as F

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
) -> _FrameOrClipTransform:
    """Build a transform callable for either one frame or a frame sequence.

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
        A callable that accepts either a single PIL image or a sequence of PIL
        images. When a sequence is provided, augmentation randomness can be
        synchronized across all frames.
    """
    normalize = (
        Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
        if use_imagenet_norm
        else Normalize(mean=_SYMMETRIC_MEAN, std=_SYMMETRIC_STD)
    )

    use_random_crop = bool(_augment_get(augment, "random_crop", False))
    crop_padding = int(_augment_get(augment, "crop_padding", 0))
    use_hflip = bool(_augment_get(augment, "random_horizontal_flip", True))
    use_color_jitter = bool(_augment_get(augment, "color_jitter", False))

    resize_size = image_size + crop_padding if (use_random_crop and crop_padding > 0) else image_size
    sync_across_frames = bool(_augment_get(augment, "sync_across_frames", False))
    color_jitter = ColorJitter(
        brightness=float(_augment_get(augment, "color_jitter_brightness", 0.0)),
        contrast=float(_augment_get(augment, "color_jitter_contrast", 0.0)),
        saturation=float(_augment_get(augment, "color_jitter_saturation", 0.0)),
        hue=float(_augment_get(augment, "color_jitter_hue", 0.0)),
    )
    return _FrameOrClipTransform(
        image_size=image_size,
        resize_size=resize_size,
        normalize=normalize,
        is_training=is_training,
        use_random_crop=use_random_crop,
        use_hflip=use_hflip,
        use_color_jitter=use_color_jitter,
        color_jitter=color_jitter,
        sync_across_frames=sync_across_frames,
    )


class _FrameOrClipTransform:
    def __init__(
        self,
        image_size: int,
        resize_size: int,
        normalize: Normalize,
        is_training: bool,
        use_random_crop: bool,
        use_hflip: bool,
        use_color_jitter: bool,
        color_jitter: ColorJitter,
        sync_across_frames: bool,
    ) -> None:
        self.image_size = image_size
        self.resize_size = resize_size
        self.normalize = normalize
        self.is_training = is_training
        self.use_random_crop = use_random_crop
        self.use_hflip = use_hflip
        self.use_color_jitter = use_color_jitter
        self.color_jitter = color_jitter
        self.sync_across_frames = sync_across_frames

    def __call__(self, image_or_images: Image.Image | Sequence[Image.Image]) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(image_or_images, Image.Image):
            return self._apply_single(image_or_images, self._sample_params())
        if len(image_or_images) == 0:
            return []

        if self.sync_across_frames:
            params = self._sample_params()
            return [self._apply_single(image, params) for image in image_or_images]
        return [self._apply_single(image, self._sample_params()) for image in image_or_images]

    def _sample_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"flip": False, "crop_ijhw": None, "jitter_fn": None}
        if self.use_random_crop:
            if self.is_training:
                top = random.randint(0, self.resize_size - self.image_size)
                left = random.randint(0, self.resize_size - self.image_size)
                params["crop_ijhw"] = (top, left, self.image_size, self.image_size)
            else:
                top = (self.resize_size - self.image_size) // 2
                left = (self.resize_size - self.image_size) // 2
                params["crop_ijhw"] = (top, left, self.image_size, self.image_size)
        if self.is_training and self.use_hflip:
            params["flip"] = bool(torch.rand(1).item() < 0.5)
        if self.is_training and self.use_color_jitter:
            params["jitter_fn"] = self.color_jitter.get_params(
                self.color_jitter.brightness,
                self.color_jitter.contrast,
                self.color_jitter.saturation,
                self.color_jitter.hue,
            )
        return params

    def _apply_single(self, image: Image.Image, params: Mapping[str, Any]) -> torch.Tensor:
        x = F.resize(image, [self.resize_size, self.resize_size])
        crop = params.get("crop_ijhw")
        if crop is not None:
            x = F.crop(x, *crop)
        if bool(params.get("flip", False)):
            x = F.hflip(x)
        jitter_fn = params.get("jitter_fn")
        if jitter_fn is not None:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = jitter_fn
            for fn_id in fn_idx.tolist():
                if fn_id == 0 and brightness_factor is not None:
                    x = F.adjust_brightness(x, float(brightness_factor))
                elif fn_id == 1 and contrast_factor is not None:
                    x = F.adjust_contrast(x, float(contrast_factor))
                elif fn_id == 2 and saturation_factor is not None:
                    x = F.adjust_saturation(x, float(saturation_factor))
                elif fn_id == 3 and hue_factor is not None:
                    x = F.adjust_hue(x, float(hue_factor))
        x = F.to_tensor(x)
        return self.normalize(x)
