"""Temporal densification: 4 anchor frames -> 16-frame clips.

Two ingestion modes aligned with Track B / V-JEPA fpc16:

* **duplicate** — same rule as :func:`pick_frame_indices` when ``num_available=4``
  and ``num_frames=16`` (linspace indices with repeats).
* **interpolate** — four uniformly spaced mids between each anchor pair
  (``t in {0.2, 0.4, 0.6, 0.8}`` per gap), using a pluggable
  :class:`FrameInterpolator` backend.

The default backend uses OpenCV optical-flow warping (no extra model weights).
Swap in FILM / RIFE later by subclassing :class:`FrameInterpolator`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from smth2smth.shared.data.video_dataset import _list_frame_paths, pick_frame_indices

ANCHOR_COUNT = 4
DENSE_FRAME_COUNT = 16
MIDS_PER_GAP = 4  # (16 - 4) // 3


def load_anchor_frames(video_dir: Path) -> list[Image.Image]:
    """Load all on-disk frames for a clip, sorted by file name.

    Args:
        video_dir: Folder with ``frame_*.jpg`` (typically exactly four frames).

    Returns:
        RGB PIL images in temporal order.

    Raises:
        RuntimeError: If the folder has no frame images.
    """
    paths = _list_frame_paths(video_dir)
    if not paths:
        raise RuntimeError(f"No frames under {video_dir}")
    frames: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(image.convert("RGB"))
    return frames


def duplicate_anchors_to_dense(anchors: Sequence[Image.Image], num_frames: int = DENSE_FRAME_COUNT) -> list[Image.Image]:
    """Repeat anchors to ``num_frames`` using the dataloader linspace rule.

    Args:
        anchors: Short clip (expected length 4).
        num_frames: Target length (default 16).

    Returns:
        Length-``num_frames`` list of PIL images (may repeat the same anchor).
    """
    n = len(anchors)
    if n <= 0:
        raise ValueError("anchors must be non-empty.")
    indices = pick_frame_indices(n, num_frames)
    return [anchors[i] for i in indices]


def _gap_fractions(n_mid: int) -> list[float]:
    """Return ``n_mid`` times in (0, 1) evenly spaced, excluding endpoints."""
    if n_mid <= 0:
        return []
    step = 1.0 / (n_mid + 1)
    return [step * (k + 1) for k in range(n_mid)]


def densify_gap(
    frame_a: Image.Image,
    frame_b: Image.Image,
    interpolator: FrameInterpolator,
    *,
    n_mid: int = MIDS_PER_GAP,
) -> list[Image.Image]:
    """Build ``[I_a, mid_1, ..., mid_n, I_b]`` along one temporal segment.

    Args:
        frame_a: Start anchor.
        frame_b: End anchor.
        interpolator: Backend that synthesizes in-between frames.
        n_mid: Number of frames strictly between ``frame_a`` and ``frame_b``.

    Returns:
        List of length ``n_mid + 2``.
    """
    mids = [interpolator.interpolate(frame_a, frame_b, t) for t in _gap_fractions(n_mid)]
    return [frame_a, *mids, frame_b]


def interpolate_anchors_to_dense(
    anchors: Sequence[Image.Image],
    interpolator: FrameInterpolator,
    *,
    num_frames: int = DENSE_FRAME_COUNT,
) -> list[Image.Image]:
    """Chain per-gap interpolation across consecutive anchors.

    Args:
        anchors: Ordered keyframes (expected length 4).
        interpolator: Frame synthesis backend.
        num_frames: Expected output length (default 16); must match
            ``len(anchors) + (len(anchors) - 1) * n_mid``.

    Returns:
        Dense clip of PIL images.

    Raises:
        ValueError: If anchor count or resulting length does not match ``num_frames``.
    """
    anchor_list = list(anchors)
    if len(anchor_list) < 2:
        raise ValueError("Need at least two anchors to interpolate.")
    n_gaps = len(anchor_list) - 1
    n_mid = (num_frames - len(anchor_list)) // n_gaps
    if len(anchor_list) + n_gaps * n_mid != num_frames:
        raise ValueError(
            f"Cannot build num_frames={num_frames} from {len(anchor_list)} anchors "
            f"with {n_mid} mids per gap."
        )

    dense: list[Image.Image] = []
    for gap_idx in range(n_gaps):
        segment = densify_gap(anchor_list[gap_idx], anchor_list[gap_idx + 1], interpolator, n_mid=n_mid)
        if gap_idx < n_gaps - 1:
            dense.extend(segment[:-1])
        else:
            dense.extend(segment)
    if len(dense) != num_frames:
        raise ValueError(f"Expected {num_frames} frames, got {len(dense)}.")
    return dense


class FrameInterpolator(ABC):
    """Synthesize one frame between two RGB images at fractional time ``t``."""

    @abstractmethod
    def interpolate(self, frame_a: Image.Image, frame_b: Image.Image, t: float) -> Image.Image:
        """Return the frame at ``t=0`` -> ``frame_a``, ``t=1`` -> ``frame_b``.

        Args:
            frame_a: Start image.
            frame_b: End image.
            t: Interpolation time in ``(0, 1)``.

        Returns:
            RGB PIL image.
        """


class BlendInterpolator(FrameInterpolator):
    """Linear cross-fade (fast baseline, no motion model)."""

    def interpolate(self, frame_a: Image.Image, frame_b: Image.Image, t: float) -> Image.Image:
        if not 0.0 < t < 1.0:
            raise ValueError(f"t must be in (0, 1), got {t}.")
        a = np.asarray(frame_a, dtype=np.float32)
        b = np.asarray(frame_b, dtype=np.float32)
        blended = (1.0 - t) * a + t * b
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


class FlowWarpInterpolator(FrameInterpolator):
    """Bidirectional optical-flow warp + blend (OpenCV Farneback).

    Requires ``opencv-python`` or ``opencv-python-headless``.
    """

    def __init__(self) -> None:
        try:
            import cv2  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "FlowWarpInterpolator requires opencv-python-headless. "
                "Install with: uv pip install opencv-python-headless"
            ) from exc

    def interpolate(self, frame_a: Image.Image, frame_b: Image.Image, t: float) -> Image.Image:
        import cv2

        if not 0.0 < t < 1.0:
            raise ValueError(f"t must be in (0, 1), got {t}.")
        a = np.asarray(frame_a.convert("RGB"))
        b = np.asarray(frame_b.convert("RGB"))
        if a.shape != b.shape:
            b = np.asarray(
                frame_b.convert("RGB").resize((a.shape[1], a.shape[0]), Image.Resampling.BILINEAR)
            )

        gray_a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        flow_ab = cv2.calcOpticalFlowFarneback(
            gray_a, gray_b, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        flow_ba = cv2.calcOpticalFlowFarneback(
            gray_b, gray_a, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        h, w = a.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

        map_a_x = grid_x + t * flow_ab[..., 0]
        map_a_y = grid_y + t * flow_ab[..., 1]
        warped_a = cv2.remap(a, map_a_x, map_a_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        map_b_x = grid_x + (1.0 - t) * flow_ba[..., 0]
        map_b_y = grid_y + (1.0 - t) * flow_ba[..., 1]
        warped_b = cv2.remap(b, map_b_x, map_b_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        out = (1.0 - t) * warped_a.astype(np.float32) + t * warped_b.astype(np.float32)
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def build_interpolator(backend: str) -> FrameInterpolator:
    """Construct an interpolator from a short backend name.

    Args:
        backend: ``"blend"`` or ``"flow"``.

    Returns:
        :class:`FrameInterpolator` instance.

    Raises:
        ValueError: If ``backend`` is unknown.
    """
    key = backend.strip().lower()
    if key == "blend":
        return BlendInterpolator()
    if key == "flow":
        return FlowWarpInterpolator()
    raise ValueError(f"Unknown interpolator backend {backend!r}. Choose 'blend' or 'flow'.")
