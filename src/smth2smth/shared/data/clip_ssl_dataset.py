"""Unlabeled video-clip dataset for V-JEPA-style SSL.

Walks the same on-disk layout used by :class:`VideoFrameDataset` -- one
folder per video, frames inside -- but is label-free and supports a mixed
roots argument so we can union ``train/``, ``val/`` and ``test/`` (a
Track-A-compliant unlabeled corpus). Each ``__getitem__`` returns a single
clip tensor ``(T, C, H, W)``; SSL pretraining does not consume labels.

Compared to :class:`VideoFrameDataset`:

* No class subfolders are required. We accept *either* a class-bucketed
  root (``root/<class>/<video>/<frame>.jpg``) -- in which case we ignore
  the class index -- *or* a flat root (``root/<video>/<frame>.jpg``, as
  on the ``test/`` split).
* Frame indices are still picked deterministically (linspace) by default,
  but an optional ``temporal_jitter`` knob nudges the start offset by a
  uniform-random amount so the same video looks slightly different epoch
  over epoch -- useful when a video has more frames than ``num_frames``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from smth2smth.shared.data.temporal_expand import VALID_MODES, expand_temporal_frames
from smth2smth.shared.data.video_dataset import (
    _FRAME_EXTENSIONS,
    _list_frame_paths,
    collect_video_samples,
    pick_frame_indices,
)


def collect_all_video_dirs(root_dirs: Sequence[Path]) -> list[Path]:
    """Walk ``root_dirs`` and return every video directory found.

    Each entry is the *parent* of frame files. Handles both layouts:
    class-bucketed (``root/<class>/<video>/<frame>``) and flat
    (``root/<video>/<frame>``). Duplicates across roots are removed.

    Args:
        root_dirs: One or more split roots (e.g. ``[train, val, test]``).

    Returns:
        Sorted list of video directories. Empty if no videos are found.
    """
    out: list[Path] = []
    for root in root_dirs:
        root = Path(root).resolve()
        if not root.is_dir():
            continue
        try:
            samples = collect_video_samples(root)
            out.extend(video_dir for video_dir, _label in samples)
        except RuntimeError:
            # Flat layout (e.g. test/<video>/<frame>): no class subfolders.
            for video_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                if any(video_dir.glob(ext) for ext in _FRAME_EXTENSIONS):
                    out.append(video_dir)
    return sorted({p for p in out}, key=lambda p: str(p))


class ClipSSLDataset(Dataset):
    """Yield a single ``(T, C, H, W)`` clip per sample, no labels.

    Args:
        video_dirs: Pre-built list of video directories (one per sample).
            Use :func:`collect_all_video_dirs` to build it from raw paths.
        num_frames: Number of frames to sample per clip (``T``). When a
            video has fewer frames than ``num_frames`` the linspace
            sampling repeats existing indices, exactly as in
            :class:`VideoFrameDataset`.
        transform: Per-frame or per-clip transform mapping a PIL ``Image``
            (or a list of them) to a ``(C, H, W)`` tensor. Typically the
            same ``build_transforms(is_training=True, ...)`` used for
            supervised training, so SSL sees the same input distribution.
        temporal_jitter: Probability of perturbing the linspace start
            offset by ±1 frame. Default ``0.0`` (deterministic). With T=4
            frames available, only ``0.0`` makes sense; the field exists
            for forward-compatibility when more frames are extracted.
        source_num_frames: Frames sampled from disk before temporal expansion.
            When ``None``, equals ``num_frames`` (no expansion).
        temporal_expand_mode: When ``source_num_frames < num_frames``, how to
            upsample: ``"replication"`` (percolation / repeat each frame) or
            ``"interpolation"`` (linear blend between neighbours).
    """

    def __init__(
        self,
        video_dirs: list[Path],
        num_frames: int,
        transform: Callable[[Image.Image | Sequence[Image.Image]], torch.Tensor | list[torch.Tensor]],
        temporal_jitter: float = 0.0,
        source_num_frames: int | None = None,
        temporal_expand_mode: str = "interpolation",
    ) -> None:
        if len(video_dirs) == 0:
            raise ValueError("video_dirs is empty.")
        if num_frames <= 0:
            raise ValueError(f"num_frames must be > 0, got {num_frames}.")
        if not 0.0 <= temporal_jitter <= 1.0:
            raise ValueError(
                f"temporal_jitter must be in [0, 1], got {temporal_jitter}."
            )
        self.video_dirs = [Path(p) for p in video_dirs]
        self.num_frames = int(num_frames)
        self.source_num_frames = int(source_num_frames) if source_num_frames is not None else self.num_frames
        if self.source_num_frames <= 0:
            raise ValueError(f"source_num_frames must be > 0, got {self.source_num_frames}.")
        if self.num_frames < self.source_num_frames:
            raise ValueError(
                f"num_frames ({self.num_frames}) must be >= source_num_frames "
                f"({self.source_num_frames})."
            )
        if temporal_expand_mode not in VALID_MODES:
            raise ValueError(
                f"temporal_expand_mode must be one of {sorted(VALID_MODES)}, "
                f"got {temporal_expand_mode!r}."
            )
        self.temporal_expand_mode = str(temporal_expand_mode)
        self.transform = transform
        self.temporal_jitter = float(temporal_jitter)

    def __len__(self) -> int:
        return len(self.video_dirs)

    def __getitem__(self, index: int) -> torch.Tensor:
        video_dir = self.video_dirs[index]
        frame_paths = _list_frame_paths(video_dir)
        if len(frame_paths) == 0:
            raise RuntimeError(f"Video folder {video_dir} has no frames.")
        indices = pick_frame_indices(len(frame_paths), self.source_num_frames)

        if self.temporal_jitter > 0.0 and len(frame_paths) > self.source_num_frames:
            if bool(torch.rand(1).item() < self.temporal_jitter):
                shift = int(torch.randint(low=-1, high=2, size=(1,)).item())
                if shift != 0:
                    indices = [
                        min(max(i + shift, 0), len(frame_paths) - 1) for i in indices
                    ]

        raw_frames: list[Image.Image] = []
        for frame_index in indices:
            path = frame_paths[frame_index]
            with Image.open(path) as image:
                raw_frames.append(image.convert("RGB"))

        if self.source_num_frames < self.num_frames:
            raw_frames = expand_temporal_frames(
                raw_frames,
                target_num_frames=self.num_frames,
                mode=self.temporal_expand_mode,
            )

        try:
            transformed = self.transform(raw_frames)
            if isinstance(transformed, list):
                frames = transformed
            else:
                frames = [self.transform(frame) for frame in raw_frames]  # type: ignore[arg-type]
        except Exception:
            frames = [self.transform(frame) for frame in raw_frames]  # type: ignore[arg-type]

        return torch.stack(frames, dim=0)  # (T, C, H, W)


__all__ = ["ClipSSLDataset", "collect_all_video_dirs"]
