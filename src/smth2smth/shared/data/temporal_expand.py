"""Expand a short source clip (e.g. T=4) to a longer model input (e.g. T=16).

Two modes mirror the professor / Track-B data regimes:

* **replication** ("percolation"): repeat each source frame ``target // source``
  times (V-JEPA fpc16 style — zero motion delta between repeated slots).
* **interpolation**: place ``target`` samples along the source timeline and
  linearly blend adjacent RGB frames in PIL space.
"""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image

VALID_MODES = frozenset({"replication", "interpolation"})


def expand_temporal_frames(
    frames: Sequence[Image.Image],
    *,
    target_num_frames: int,
    mode: str = "interpolation",
) -> list[Image.Image]:
    """Upsample a short frame list to ``target_num_frames`` outputs.

    Args:
        frames: Source RGB PIL images in temporal order (length ``source_T``).
        target_num_frames: Desired output length (e.g. 16).
        mode: ``"replication"`` or ``"interpolation"``.

    Returns:
        List of length ``target_num_frames``.

    Raises:
        ValueError: On invalid ``mode``, empty input, or incompatible sizes.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}.")
    source_t = len(frames)
    if source_t == 0:
        raise ValueError("frames must not be empty.")
    target_t = int(target_num_frames)
    if target_t <= 0:
        raise ValueError(f"target_num_frames must be > 0, got {target_t}.")
    if source_t == target_t:
        return list(frames)
    if source_t == 1:
        return [frames[0]] * target_t
    if mode == "replication":
        return _expand_replication(frames, target_t=target_t)
    return _expand_interpolation(frames, target_t=target_t)


def _expand_replication(frames: Sequence[Image.Image], *, target_t: int) -> list[Image.Image]:
    source_t = len(frames)
    base = target_t // source_t
    remainder = target_t % source_t
    out: list[Image.Image] = []
    for i, frame in enumerate(frames):
        repeats = base + (1 if i < remainder else 0)
        out.extend([frame] * repeats)
    return out


def _expand_interpolation(frames: Sequence[Image.Image], *, target_t: int) -> list[Image.Image]:
    source_t = len(frames)
    if target_t == 1:
        return [frames[0]]
    out: list[Image.Image] = []
    max_src = float(source_t - 1)
    for out_i in range(target_t):
        pos = out_i * max_src / float(target_t - 1)
        lo = int(pos)
        hi = min(lo + 1, source_t - 1)
        alpha = pos - float(lo)
        if lo == hi or alpha <= 0.0:
            out.append(frames[lo])
        elif alpha >= 1.0:
            out.append(frames[hi])
        else:
            out.append(Image.blend(frames[lo], frames[hi], alpha))
    return out


__all__ = ["VALID_MODES", "expand_temporal_frames"]
