"""Extract only active (enabled) augmentation knobs for logging."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _get(augment: Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    if augment is None:
        return default
    try:
        value = augment.get(key, default)  # type: ignore[union-attr]
    except Exception:
        return default
    return default if value is None else value


def active_augment_summary(augment: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a flat dict of augmentation settings that are actually in use.

    Omits disabled flags and nested blocks that are off, so W&B / stdout only
    show augmentations that affect training.

    Args:
        augment: Hydra ``augment`` config mapping (or ``None``).

    Returns:
        Plain dict suitable for ``wandb.config`` or structured logging.
    """
    if augment is None:
        return {}

    out: dict[str, Any] = {}
    name = _get(augment, "name")
    if name is not None:
        out["augment_name"] = str(name)

    if bool(_get(augment, "random_crop", False)):
        out["random_crop"] = True
        padding = int(_get(augment, "crop_padding", 0))
        if padding > 0:
            out["crop_padding"] = padding

    if bool(_get(augment, "random_horizontal_flip", False)):
        out["random_horizontal_flip"] = True
        prob = float(_get(augment, "random_horizontal_flip_prob", 0.5))
        out["random_horizontal_flip_prob"] = prob

    if bool(_get(augment, "sync_across_frames", False)):
        out["sync_across_frames"] = True

    if bool(_get(augment, "color_jitter", False)):
        out["color_jitter"] = True
        for key in (
            "color_jitter_brightness",
            "color_jitter_contrast",
            "color_jitter_saturation",
            "color_jitter_hue",
            "color_jitter_prob",
        ):
            val = _get(augment, key)
            if val is not None and float(val) != 0.0:
                out[key] = float(val)

    if bool(_get(augment, "random_grayscale", False)):
        out["random_grayscale"] = True
        out["random_grayscale_prob"] = float(_get(augment, "random_grayscale_prob", 0.0))

    if bool(_get(augment, "gaussian_blur", False)):
        out["gaussian_blur"] = True
        out["gaussian_blur_prob"] = float(_get(augment, "gaussian_blur_prob", 0.0))
        out["gaussian_blur_radius_min"] = float(_get(augment, "gaussian_blur_radius_min", 0.1))
        out["gaussian_blur_radius_max"] = float(_get(augment, "gaussian_blur_radius_max", 2.0))

    ra_cfg = _get(augment, "randaugment", None)
    if ra_cfg is not None and bool(_get(ra_cfg, "enabled", False)):
        out["randaugment"] = {
            "enabled": True,
            "n": int(_get(ra_cfg, "n", 2)),
            "m": float(_get(ra_cfg, "m", 9.0)),
            "mode": str(_get(ra_cfg, "mode", "temporal_plus")),
        }
        mag_max = _get(ra_cfg, "magnitude_max")
        if mag_max is not None:
            out["randaugment"]["magnitude_max"] = float(mag_max)
        ops = _get(ra_cfg, "ops")
        if ops is not None:
            out["randaugment"]["ops"] = list(ops)

    return out


__all__ = ["active_augment_summary"]
