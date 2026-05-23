"""Frame/clip transforms.

Adapted from the professor baseline at
``external/prof_baseline/src/utils.py::build_transforms``.

The augmentation policy is configurable via the ``augment`` Hydra config group
(see ``configs/augment/{none,strong}.yaml``). Calls without an ``augment``
argument fall back to the legacy behavior (Resize + optional RandomHorizontalFlip)
so that existing call sites and the smoke test keep working.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from PIL import Image, ImageFilter, ImageOps
from torchvision.transforms import ColorJitter, Normalize
from torchvision.transforms import functional as F

from smth2smth.shared.data.randaugment import DEFAULT_MAGNITUDE_MAX, RandAugment

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


def _sample_multiscale_crop_fractions(
    scale: tuple[float, float],
    aspect: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Sample a random resized crop as fractions of image height/width.

    Matches torchvision ``RandomResizedCrop`` area/aspect sampling used in the
    MCG-NJU VideoMAE SSv2 recipe (scale in [0.08, 1.0], aspect in [0.75, 1.33]).

    Returns:
        ``(top_frac, left_frac, height_frac, width_frac)`` in ``[0, 1]``.
    """
    scale_lo, scale_hi = float(scale[0]), float(scale[1])
    aspect_lo, aspect_hi = float(aspect[0]), float(aspect[1])
    for _ in range(10):
        target_area = random.uniform(scale_lo, scale_hi)
        log_aspect = random.uniform(math.log(aspect_lo), math.log(aspect_hi))
        aspect_ratio = math.exp(log_aspect)
        height_frac = math.sqrt(target_area / aspect_ratio)
        width_frac = math.sqrt(target_area * aspect_ratio)
        if height_frac <= 1.0 and width_frac <= 1.0 and min(height_frac, width_frac) > 0.05:
            top_frac = random.uniform(0.0, 1.0 - height_frac)
            left_frac = random.uniform(0.0, 1.0 - width_frac)
            return top_frac, left_frac, height_frac, width_frac
    height_frac = width_frac = 0.9
    top_frac = left_frac = 0.05
    return top_frac, left_frac, height_frac, width_frac


def _random_erase_tensor(
    tensor: torch.Tensor,
    *,
    prob: float,
    mode: str,
    count: int,
    sl: float = 0.02,
    sh: float = 0.4,
    r1: float = 0.3,
) -> torch.Tensor:
    """Apply DeiT-style random erasing on a ``(C, H, W)`` tensor in ``[0, 1]``."""
    if prob <= 0.0 or random.random() > prob:
        return tensor
    out = tensor.clone()
    _, height, width = out.shape
    area = height * width
    for _ in range(max(1, int(count))):
        for _attempt in range(100):
            target_area = random.uniform(sl, sh) * area
            aspect = random.uniform(r1, 1.0 / r1)
            erase_h = int(round(math.sqrt(target_area * aspect)))
            erase_w = int(round(math.sqrt(target_area / aspect)))
            if erase_h < height and erase_w < width:
                top = random.randint(0, height - erase_h)
                left = random.randint(0, width - erase_w)
                if mode == "pixel":
                    out[:, top : top + erase_h, left : left + erase_w] = torch.rand(
                        out.size(0), erase_h, erase_w, device=out.device, dtype=out.dtype
                    )
                else:
                    out[:, top : top + erase_h, left : left + erase_w] = 0.0
                break
    return out


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
            - ``random_grayscale`` (bool) and ``random_grayscale_prob`` (float)
              -- training only; converts to grayscale then back to 3 channels.
            - ``gaussian_blur`` (bool), ``gaussian_blur_prob`` (float), and
              ``gaussian_blur_radius_{min,max}`` -- PIL Gaussian blur (training).

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
    hflip_prob = float(_augment_get(augment, "random_horizontal_flip_prob", 0.5))
    use_color_jitter = bool(_augment_get(augment, "color_jitter", False))
    color_jitter_prob = float(_augment_get(augment, "color_jitter_prob", 1.0))

    resize_size = (
        image_size + crop_padding if (use_random_crop and crop_padding > 0) else image_size
    )
    sync_across_frames = bool(_augment_get(augment, "sync_across_frames", False))
    color_jitter = ColorJitter(
        brightness=float(_augment_get(augment, "color_jitter_brightness", 0.0)),
        contrast=float(_augment_get(augment, "color_jitter_contrast", 0.0)),
        saturation=float(_augment_get(augment, "color_jitter_saturation", 0.0)),
        hue=float(_augment_get(augment, "color_jitter_hue", 0.0)),
    )

    randaugment = _build_randaugment(augment) if is_training else None

    random_grayscale = bool(_augment_get(augment, "random_grayscale", False))
    random_grayscale_prob = float(_augment_get(augment, "random_grayscale_prob", 0.0))
    gaussian_blur = bool(_augment_get(augment, "gaussian_blur", False))
    gaussian_blur_prob = float(_augment_get(augment, "gaussian_blur_prob", 0.0))
    gaussian_blur_radius_min = float(_augment_get(augment, "gaussian_blur_radius_min", 0.1))
    gaussian_blur_radius_max = float(_augment_get(augment, "gaussian_blur_radius_max", 2.0))

    use_multiscale_crop = bool(_augment_get(augment, "multiscale_crop", False))
    ms_scale_cfg = _augment_get(augment, "multiscale_scale", [0.08, 1.0])
    ms_aspect_cfg = _augment_get(augment, "multiscale_aspect", [0.75, 1.3333])
    multiscale_scale = (float(ms_scale_cfg[0]), float(ms_scale_cfg[1]))
    multiscale_aspect = (float(ms_aspect_cfg[0]), float(ms_aspect_cfg[1]))

    re_cfg = _augment_get(augment, "random_erase", None)
    random_erase_enabled = bool(_augment_get(re_cfg, "enabled", False)) and is_training
    random_erase_prob = float(_augment_get(re_cfg, "prob", 0.0))
    random_erase_mode = str(_augment_get(re_cfg, "mode", "pixel"))
    random_erase_count = int(_augment_get(re_cfg, "count", 1))

    return _FrameOrClipTransform(
        image_size=image_size,
        resize_size=resize_size,
        normalize=normalize,
        is_training=is_training,
        use_random_crop=use_random_crop,
        use_hflip=use_hflip,
        hflip_prob=hflip_prob,
        use_color_jitter=use_color_jitter,
        color_jitter_prob=color_jitter_prob,
        color_jitter=color_jitter,
        sync_across_frames=sync_across_frames,
        randaugment=randaugment,
        random_grayscale=random_grayscale,
        random_grayscale_prob=random_grayscale_prob,
        gaussian_blur=gaussian_blur,
        gaussian_blur_prob=gaussian_blur_prob,
        gaussian_blur_radius_min=gaussian_blur_radius_min,
        gaussian_blur_radius_max=gaussian_blur_radius_max,
        use_multiscale_crop=use_multiscale_crop,
        multiscale_scale=multiscale_scale,
        multiscale_aspect=multiscale_aspect,
        random_erase_enabled=random_erase_enabled,
        random_erase_prob=random_erase_prob,
        random_erase_mode=random_erase_mode,
        random_erase_count=random_erase_count,
    )


def _build_randaugment(augment: Mapping[str, Any] | None) -> RandAugment | None:
    """Build a :class:`RandAugment` instance from an augment cfg, or ``None``.

    Reads the optional ``augment.randaugment`` sub-config. If it's missing, or
    ``enabled`` is falsy, returns ``None`` and downstream code skips it.
    """
    ra_cfg = _augment_get(augment, "randaugment", None)
    if ra_cfg is None:
        return None
    if not bool(_augment_get(ra_cfg, "enabled", False)):
        return None
    n = int(_augment_get(ra_cfg, "n", 2))
    m = float(_augment_get(ra_cfg, "m", 9.0))
    mode = str(_augment_get(ra_cfg, "mode", "temporal_plus"))
    magnitude_max = float(_augment_get(ra_cfg, "magnitude_max", DEFAULT_MAGNITUDE_MAX))
    ops = _augment_get(ra_cfg, "ops", None)
    if ops is not None:
        ops = list(ops)
    return RandAugment(n=n, m=m, mode=mode, magnitude_max=magnitude_max, ops=ops)


class _FrameOrClipTransform:
    def __init__(
        self,
        image_size: int,
        resize_size: int,
        normalize: Normalize,
        is_training: bool,
        use_random_crop: bool,
        use_hflip: bool,
        hflip_prob: float,
        use_color_jitter: bool,
        color_jitter_prob: float,
        color_jitter: ColorJitter,
        sync_across_frames: bool,
        randaugment: RandAugment | None = None,
        *,
        random_grayscale: bool = False,
        random_grayscale_prob: float = 0.0,
        gaussian_blur: bool = False,
        gaussian_blur_prob: float = 0.0,
        gaussian_blur_radius_min: float = 0.1,
        gaussian_blur_radius_max: float = 2.0,
        use_multiscale_crop: bool = False,
        multiscale_scale: tuple[float, float] = (0.08, 1.0),
        multiscale_aspect: tuple[float, float] = (0.75, 1.3333),
        random_erase_enabled: bool = False,
        random_erase_prob: float = 0.0,
        random_erase_mode: str = "pixel",
        random_erase_count: int = 1,
    ) -> None:
        self.image_size = image_size
        self.resize_size = resize_size
        self.normalize = normalize
        self.is_training = is_training
        self.use_random_crop = use_random_crop
        self.use_hflip = use_hflip
        self.hflip_prob = float(hflip_prob)
        self.use_color_jitter = use_color_jitter
        self.color_jitter_prob = float(color_jitter_prob)
        self.color_jitter = color_jitter
        self.sync_across_frames = sync_across_frames
        self.randaugment = randaugment
        self.random_grayscale = random_grayscale
        self.random_grayscale_prob = random_grayscale_prob
        self.gaussian_blur = gaussian_blur
        self.gaussian_blur_prob = gaussian_blur_prob
        self.gaussian_blur_radius_min = gaussian_blur_radius_min
        self.gaussian_blur_radius_max = gaussian_blur_radius_max
        self.use_multiscale_crop = use_multiscale_crop
        self.multiscale_scale = multiscale_scale
        self.multiscale_aspect = multiscale_aspect
        self.random_erase_enabled = random_erase_enabled
        self.random_erase_prob = random_erase_prob
        self.random_erase_mode = random_erase_mode
        self.random_erase_count = random_erase_count

    def __call__(
        self, image_or_images: Image.Image | Sequence[Image.Image]
    ) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(image_or_images, Image.Image):
            params = self._sample_params(num_frames=1)
            return self._apply_single(image_or_images, params, frame_index=0)
        if len(image_or_images) == 0:
            return []

        num_frames = len(image_or_images)
        if self.sync_across_frames:
            params = self._sample_params(num_frames=num_frames)
            return [
                self._apply_single(image, params, frame_index=t)
                for t, image in enumerate(image_or_images)
            ]
        # Independent geometric/photometric randomness per frame, but RandAugment
        # always uses the same op identities and magnitude schedule across the
        # whole clip (otherwise temporal interpolation is meaningless).
        ra_params = self._sample_randaugment_params(num_frames=num_frames)
        outputs: list[torch.Tensor] = []
        for t, image in enumerate(image_or_images):
            params = self._sample_params(num_frames=num_frames, randaugment_params=ra_params)
            outputs.append(self._apply_single(image, params, frame_index=t))
        return outputs

    def _sample_randaugment_params(self, num_frames: int) -> dict[str, Any] | None:
        """Sample RandAugment op identities and per-frame magnitudes for one clip."""
        if self.randaugment is None or not self.is_training or self.randaugment.n == 0:
            return None
        op_names = self.randaugment._sample_op_names()
        levels = self.randaugment.magnitudes(num_frames=num_frames)
        return {"op_names": op_names, "levels": levels}

    def _sample_params(
        self,
        num_frames: int = 1,
        randaugment_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "flip": False,
            "crop_ijhw": None,
            "jitter_fn": None,
            "randaugment": randaugment_params
            if randaugment_params is not None
            else self._sample_randaugment_params(num_frames=num_frames),
        }
        if self.use_multiscale_crop:
            if self.is_training:
                params["multiscale_frac"] = _sample_multiscale_crop_fractions(
                    self.multiscale_scale, self.multiscale_aspect
                )
            else:
                params["multiscale_frac"] = None
        elif self.use_random_crop:
            if self.is_training:
                top = random.randint(0, self.resize_size - self.image_size)
                left = random.randint(0, self.resize_size - self.image_size)
                params["crop_ijhw"] = (top, left, self.image_size, self.image_size)
            else:
                top = (self.resize_size - self.image_size) // 2
                left = (self.resize_size - self.image_size) // 2
                params["crop_ijhw"] = (top, left, self.image_size, self.image_size)
        if self.random_erase_enabled:
            params["random_erase"] = random.random() < self.random_erase_prob
        else:
            params["random_erase"] = False
        if self.is_training and self.use_hflip:
            params["flip"] = bool(torch.rand(1).item() < self.hflip_prob)
        if (
            self.is_training
            and self.use_color_jitter
            and self.color_jitter_prob > 0.0
            and random.random() < self.color_jitter_prob
        ):
            params["jitter_fn"] = self.color_jitter.get_params(
                self.color_jitter.brightness,
                self.color_jitter.contrast,
                self.color_jitter.saturation,
                self.color_jitter.hue,
            )
        if self.is_training and self.random_grayscale and self.random_grayscale_prob > 0.0:
            params["grayscale"] = random.random() < self.random_grayscale_prob
        else:
            params["grayscale"] = False
        if self.is_training and self.gaussian_blur and self.gaussian_blur_prob > 0.0:
            if random.random() < self.gaussian_blur_prob:
                r_lo = min(self.gaussian_blur_radius_min, self.gaussian_blur_radius_max)
                r_hi = max(self.gaussian_blur_radius_min, self.gaussian_blur_radius_max)
                params["blur_radius"] = random.uniform(r_lo, r_hi)
            else:
                params["blur_radius"] = None
        else:
            params["blur_radius"] = None
        return params

    def _apply_single(
        self,
        image: Image.Image,
        params: Mapping[str, Any],
        frame_index: int = 0,
    ) -> torch.Tensor:
        ra = params.get("randaugment")
        if ra is not None and self.randaugment is not None:
            op_names = ra["op_names"]
            levels = ra["levels"]
            if len(levels) > 0:
                level = levels[min(frame_index, len(levels) - 1)]
                image = self.randaugment._apply_ops_to_frame(image, op_names, level)
        ms_frac = params.get("multiscale_frac")
        if ms_frac is not None:
            top_f, left_f, h_f, w_f = ms_frac
            width, height = image.size
            crop_h = max(1, int(round(h_f * height)))
            crop_w = max(1, int(round(w_f * width)))
            top = min(int(round(top_f * height)), height - crop_h)
            left = min(int(round(left_f * width)), width - crop_w)
            x = F.crop(image, top, left, crop_h, crop_w)
            x = F.resize(x, [self.image_size, self.image_size])
        elif self.use_multiscale_crop and not self.is_training:
            x = F.resize(image, [self.image_size, self.image_size])
        else:
            x = F.resize(image, [self.resize_size, self.resize_size])
            crop = params.get("crop_ijhw")
            if crop is not None:
                x = F.crop(x, *crop)
        if bool(params.get("flip", False)):
            x = F.hflip(x)
        if bool(params.get("grayscale", False)):
            x = ImageOps.grayscale(x).convert("RGB")
        blur_r = params.get("blur_radius")
        if blur_r is not None:
            x = x.filter(ImageFilter.GaussianBlur(radius=float(blur_r)))
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
        if bool(params.get("random_erase", False)):
            x = _random_erase_tensor(
                x,
                prob=1.0,
                mode=self.random_erase_mode,
                count=self.random_erase_count,
            )
        return self.normalize(x)
