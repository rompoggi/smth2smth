"""RandAugment and RandAugment-T for video clips.

Implements the per-frame data-level augmentation policies described in
Kim et al. 2020, "Learning Temporally Invariant and Localizable Features via
Data Augmentation for Video Recognition" (Sec. 3.1, Fig. 2). The 14-op set is
the standard RandAugment palette (Cubuk et al. 2019).

This module is intentionally Hydra-free: the public API takes plain Python
arguments (``int``/``float``/``str``) and operates on PIL images (or sequences
of PIL images for the temporal extension). The Hydra-aware glue lives in
``transforms.py``.

Magnitudes are normalized to a [0, ``magnitude_max``] integer scale (default 30,
following the original RandAugment paper). Each op converts the level to its
own physical units (e.g. degrees of rotation, fraction of width for translate,
0..256 threshold for solarize).

Modes (Sec. 4.2):
    * ``"spatial"``       : single magnitude ``M`` shared across the clip
                              (this is the original image-domain RandAugment
                              applied to each frame, with synced randomness).
    * ``"temporal"``      : ``M1 ~ Uniform(0.1, M)``, ``M2 = M``.
    * ``"temporal_plus"`` : ``M1 = M - delta``, ``M2 = M + delta``,
                              ``delta ~ Uniform(0, 0.5 * M)``. Best in Tab. 2.
    * ``"mix"``           : per-clip, randomly pick ``"spatial"`` or
                              ``"temporal_plus"``.

Geometric ops (rotate, shear-x, shear-y, translate-x, translate-y) and
photometric ops (solarize, color, posterize, contrast, brightness, sharpness)
honor the per-frame magnitude. ``identity``, ``autocontrast``, and ``equalize``
are magnitude-free and apply uniformly across frames.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

from PIL import Image, ImageOps
from torchvision.transforms import functional as F

# -- Op registry ------------------------------------------------------------

# Each op maps (image, level_in_[0, magnitude_max], magnitude_max) -> image.
OpFn = Callable[[Image.Image, float, float], Image.Image]

# Maximum physical magnitudes at level == magnitude_max. Tuned to roughly match
# the values in the original RandAugment reference implementation.
_MAX_ROTATE_DEG = 30.0
_MAX_SHEAR = 0.3  # tan(angle), unitless
_MAX_TRANSLATE_FRAC = 0.45  # fraction of image width / height
_MAX_ENHANCE = 0.9  # color/contrast/brightness/sharpness factor offset
_MAX_POSTERIZE_REDUCE = 4  # bits removed (8 -> 4)
_MAX_SOLARIZE_REDUCE = 256  # threshold floor (256 -> 0)


def _scale(level: float, magnitude_max: float, max_value: float) -> float:
    """Linear scale ``level / magnitude_max * max_value`` clamped to [0, max_value]."""
    if magnitude_max <= 0:
        return 0.0
    return max(0.0, min(1.0, level / magnitude_max)) * max_value


def _signed(level: float, magnitude_max: float, max_value: float) -> float:
    """Same as ``_scale`` but with a random sign (used by symmetric ops like rotate)."""
    value = _scale(level, magnitude_max, max_value)
    return value if random.random() < 0.5 else -value


def _identity(img: Image.Image, level: float, magnitude_max: float) -> Image.Image:  # noqa: ARG001
    return img


def _autocontrast(img: Image.Image, level: float, magnitude_max: float) -> Image.Image:  # noqa: ARG001
    return ImageOps.autocontrast(img)


def _equalize(img: Image.Image, level: float, magnitude_max: float) -> Image.Image:  # noqa: ARG001
    return ImageOps.equalize(img)


def _rotate(img: Image.Image, level: float, magnitude_max: float) -> Image.Image:
    return F.rotate(img, _signed(level, magnitude_max, _MAX_ROTATE_DEG))


def _shear_x(img: Image.Image, level: float, magnitude_max: float) -> Image.Image:
    s = _signed(level, magnitude_max, _MAX_SHEAR)
    return F.affine(
        img, angle=0.0, translate=(0, 0), scale=1.0, shear=[s * 180.0 / 3.141592653589793, 0.0]
    )


def _shear_y(img: Image.Image, level: float, magnitude_max: float) -> Image.Image:
    s = _signed(level, magnitude_max, _MAX_SHEAR)
    return F.affine(
        img, angle=0.0, translate=(0, 0), scale=1.0, shear=[0.0, s * 180.0 / 3.141592653589793]
    )


def _translate_x(img: Image.Image, level: float, magnitude_max: float) -> Image.Image:
    frac = _signed(level, magnitude_max, _MAX_TRANSLATE_FRAC)
    px = int(round(frac * img.size[0]))
    return F.affine(img, angle=0.0, translate=(px, 0), scale=1.0, shear=[0.0, 0.0])


def _translate_y(img: Image.Image, level: float, magnitude_max: float) -> Image.Image:
    frac = _signed(level, magnitude_max, _MAX_TRANSLATE_FRAC)
    px = int(round(frac * img.size[1]))
    return F.affine(img, angle=0.0, translate=(0, px), scale=1.0, shear=[0.0, 0.0])


def _solarize(img: Image.Image, level: float, magnitude_max: float) -> Image.Image:
    threshold = int(
        round(_MAX_SOLARIZE_REDUCE - _scale(level, magnitude_max, _MAX_SOLARIZE_REDUCE))
    )
    threshold = max(0, min(256, threshold))
    return ImageOps.solarize(img, threshold=threshold)


def _posterize(img: Image.Image, level: float, magnitude_max: float) -> Image.Image:
    bits = int(round(8 - _scale(level, magnitude_max, _MAX_POSTERIZE_REDUCE)))
    bits = max(1, min(8, bits))
    return ImageOps.posterize(img, bits=bits)


def _color(img: Image.Image, level: float, magnitude_max: float) -> Image.Image:
    return F.adjust_saturation(img, 1.0 + _signed(level, magnitude_max, _MAX_ENHANCE))


def _contrast(img: Image.Image, level: float, magnitude_max: float) -> Image.Image:
    return F.adjust_contrast(img, 1.0 + _signed(level, magnitude_max, _MAX_ENHANCE))


def _brightness(img: Image.Image, level: float, magnitude_max: float) -> Image.Image:
    return F.adjust_brightness(img, 1.0 + _signed(level, magnitude_max, _MAX_ENHANCE))


def _sharpness(img: Image.Image, level: float, magnitude_max: float) -> Image.Image:
    return F.adjust_sharpness(img, 1.0 + _signed(level, magnitude_max, _MAX_ENHANCE))


RA_OPS: dict[str, OpFn] = {
    "identity": _identity,
    "autocontrast": _autocontrast,
    "equalize": _equalize,
    "rotate": _rotate,
    "shear_x": _shear_x,
    "shear_y": _shear_y,
    "translate_x": _translate_x,
    "translate_y": _translate_y,
    "solarize": _solarize,
    "posterize": _posterize,
    "color": _color,
    "contrast": _contrast,
    "brightness": _brightness,
    "sharpness": _sharpness,
}

# Ops whose effect is independent of magnitude (paper Sec. 3.1).
RA_FREE_OPS: frozenset[str] = frozenset({"identity", "autocontrast", "equalize"})

DEFAULT_MAGNITUDE_MAX: float = 30.0
VALID_MODES: frozenset[str] = frozenset({"spatial", "temporal", "temporal_plus", "mix"})


class RandAugment:
    """RandAugment / RandAugment-T applied to a clip (sequence of PIL frames).

    Args:
        n: Number of augmentation operations to apply sequentially per frame
            (paper symbol ``N``). Sampled once per clip; the *same* op
            identities are reused across all frames so that motion within a
            clip is only modulated by the magnitude.
        m: Base magnitude in [0, ``magnitude_max``] (paper symbol ``M``).
        mode: One of ``{"spatial", "temporal", "temporal_plus", "mix"}``.
            See the module docstring.
        magnitude_max: Upper bound of the magnitude scale. The reference
            RandAugment uses 30.
        ops: Optional iterable of op names to restrict the candidate pool to.
            Defaults to all 14 ops in :data:`RA_OPS`.

    Notes:
        * Calling on a single PIL ``Image`` is a no-op for the temporal
          interpolation (only one frame, so M1 == M2 == M).
        * Calling on a sequence of frames returns a list of PIL frames of the
          same length.
        * This class does **not** convert to tensors; it composes naturally
          with ``F.to_tensor`` / ``Normalize`` downstream.
    """

    def __init__(
        self,
        n: int,
        m: float,
        mode: str = "temporal_plus",
        magnitude_max: float = DEFAULT_MAGNITUDE_MAX,
        ops: Sequence[str] | None = None,
    ) -> None:
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        if m < 0:
            raise ValueError(f"m must be >= 0, got {m}")
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}")

        self.n = int(n)
        self.m = float(m)
        self.mode = mode
        self.magnitude_max = float(magnitude_max)

        if ops is None:
            self.op_names: list[str] = list(RA_OPS.keys())
        else:
            unknown = [name for name in ops if name not in RA_OPS]
            if unknown:
                raise ValueError(f"Unknown RandAugment ops: {unknown}")
            self.op_names = list(ops)
        if len(self.op_names) == 0:
            raise ValueError("ops must contain at least one operation")

    def _resolve_endpoints(self) -> tuple[float, float]:
        """Return ``(M1, M2)`` for the current call, per the active mode."""
        m = self.m
        if self.mode == "spatial":
            return m, m
        if self.mode == "temporal":
            m1_low = 0.1
            m1_high = max(m1_low, m)
            m1 = random.uniform(m1_low, m1_high)
            return m1, m
        if self.mode == "temporal_plus":
            delta = random.uniform(0.0, 0.5 * m)
            return max(0.0, m - delta), min(self.magnitude_max, m + delta)
        # "mix"
        if random.random() < 0.5:
            return m, m
        delta = random.uniform(0.0, 0.5 * m)
        return max(0.0, m - delta), min(self.magnitude_max, m + delta)

    def magnitudes(self, num_frames: int) -> list[float]:
        """Return the per-frame magnitudes for a clip of ``num_frames``."""
        if num_frames <= 0:
            return []
        m1, m2 = self._resolve_endpoints()
        if num_frames == 1:
            return [0.5 * (m1 + m2)]
        step = (m2 - m1) / (num_frames - 1)
        return [m1 + step * t for t in range(num_frames)]

    def _sample_op_names(self) -> list[str]:
        """Sample ``n`` ops *with replacement* (matches the paper's Numpy pseudocode)."""
        return [random.choice(self.op_names) for _ in range(self.n)]

    def _apply_ops_to_frame(
        self,
        image: Image.Image,
        op_names: Sequence[str],
        level: float,
    ) -> Image.Image:
        out = image
        for name in op_names:
            fn = RA_OPS[name]
            applied_level = 0.0 if name in RA_FREE_OPS else level
            out = fn(out, applied_level, self.magnitude_max)
        return out

    def __call__(
        self,
        image_or_images: Image.Image | Sequence[Image.Image],
    ) -> Image.Image | list[Image.Image]:
        if self.n == 0:
            if isinstance(image_or_images, Image.Image):
                return image_or_images
            return list(image_or_images)

        if isinstance(image_or_images, Image.Image):
            op_names = self._sample_op_names()
            return self._apply_ops_to_frame(image_or_images, op_names, self.m)

        frames = list(image_or_images)
        if len(frames) == 0:
            return frames
        op_names = self._sample_op_names()
        levels = self.magnitudes(len(frames))
        return [
            self._apply_ops_to_frame(frame, op_names, levels[t]) for t, frame in enumerate(frames)
        ]


__all__ = [
    "DEFAULT_MAGNITUDE_MAX",
    "RA_FREE_OPS",
    "RA_OPS",
    "VALID_MODES",
    "RandAugment",
]
