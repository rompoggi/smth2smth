"""Still-frame dataset for self-supervised pretraining.

The supervised pipeline operates on whole videos (``T`` frames per sample).
Self-supervised pretraining (DINO, MoCo, etc.) is most effective when each
sample is a *single* still frame with two strongly augmented views: this gives
the trunk many millions of training pairs even from a few thousand videos.

This module does NOT touch the existing :class:`VideoFrameDataset` -- it adds
a parallel, image-only entry point that walks the same disk layout.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from smth2smth.shared.data.video_dataset import (
    _FRAME_EXTENSIONS,
    collect_video_samples,
)


def collect_all_frame_paths(root_dirs: list[Path], include_test: bool = True) -> list[Path]:
    """Walk ``root_dirs`` and return every individual frame file path.

    The function handles both layouts used by the project:

    * ``train/<class>/<video>/<frame>.jpg`` -- via :func:`collect_video_samples`.
    * ``test/<video>/<frame>.jpg`` -- when no class subdirectory exists, walk
      one level shallower.

    Args:
        root_dirs: List of split roots to scan (e.g. ``[train, val, test]``).
        include_test: If False, scans only the directories that look like
            class-bucketed splits. Mostly here for unit tests.

    Returns:
        Sorted list of frame paths. Empty if no frames are found.
    """
    paths: list[Path] = []
    for root in root_dirs:
        root = root.resolve()
        if not root.is_dir():
            continue
        try:
            samples = collect_video_samples(root)
            for video_dir, _label in samples:
                for ext in _FRAME_EXTENSIONS:
                    paths.extend(video_dir.glob(ext))
        except RuntimeError:
            if not include_test:
                continue
            # No class subfolders -> walk one level: ``root/<video>/<frame>.jpg``.
            for video_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                for ext in _FRAME_EXTENSIONS:
                    paths.extend(video_dir.glob(ext))
    return sorted(set(paths), key=lambda p: str(p))


class MultiViewStillFramesDataset(Dataset):
    """Yields several augmented views of a single still frame per sample.

    Designed for DINO-style multi-crop SSL: ``view_transform`` should be
    callable with no arguments returning a fresh transform that applies a
    random augmentation pipeline to a PIL image; we sample one global pair
    plus ``num_local_views`` local crops per item.

    Args:
        frame_paths: Pre-built list of frame file paths.
        global_transform: Transform applied to produce the two global views.
        local_transform: Transform applied to produce each of the local views.
            ``None`` disables local crops (only the two globals are returned).
        num_local_views: Number of local crops per sample. ``0`` ⇒ none.
    """

    def __init__(
        self,
        frame_paths: list[Path],
        global_transform: Callable[[Image.Image], torch.Tensor],
        local_transform: Callable[[Image.Image], torch.Tensor] | None = None,
        num_local_views: int = 0,
    ) -> None:
        if len(frame_paths) == 0:
            raise ValueError("frame_paths is empty.")
        if num_local_views < 0:
            raise ValueError(f"num_local_views must be >= 0, got {num_local_views}.")
        if num_local_views > 0 and local_transform is None:
            raise ValueError("local_transform is required when num_local_views > 0.")
        self.frame_paths = list(frame_paths)
        self.global_transform = global_transform
        self.local_transform = local_transform
        self.num_local_views = int(num_local_views)

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.frame_paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            global_views = [self.global_transform(image), self.global_transform(image)]
            local_views: list[torch.Tensor] = []
            if self.num_local_views > 0 and self.local_transform is not None:
                for _ in range(self.num_local_views):
                    local_views.append(self.local_transform(image))
        return {
            "global_views": global_views,
            "local_views": local_views,
        }
