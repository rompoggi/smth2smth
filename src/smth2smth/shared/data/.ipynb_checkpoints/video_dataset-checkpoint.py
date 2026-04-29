"""Frame-folder video dataset.

Adapted from the professor baseline at
``external/prof_baseline/src/dataset/video_dataset.py``.

Expected folder layout under ``root_dir``::

    root_dir/
        000_SomeClass/
            video_12345/
                frame_000.jpg
                frame_001.jpg
                ...
        001_AnotherClass/
            ...

Each ``__getitem__`` returns:
    video_tensor: float tensor of shape ``(T, C, H, W)``
    label: int64 scalar class index
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

_FRAME_EXTENSIONS: tuple[str, ...] = ("*.jpg", "*.jpeg", "*.png", "*.webp")

VideoSample = tuple[Path, int]


def _list_frame_paths(video_dir: Path) -> list[Path]:
    """Return all frame image paths in ``video_dir`` sorted by file name.

    Args:
        video_dir: Directory containing frame images for a single video.

    Returns:
        Sorted list of frame file paths. Empty list if no frames are found.
    """
    paths: list[Path] = []
    for extension in _FRAME_EXTENSIONS:
        paths.extend(video_dir.glob(extension))
    return sorted(paths, key=lambda p: p.name)


def parse_class_index(class_dir_name: str) -> int | None:
    """Parse the leading numeric prefix from a class folder name.

    Args:
        class_dir_name: Folder name, e.g. ``"017_PutSomethingNextToSomething"``.

    Returns:
        The integer prefix (e.g. ``17``) or ``None`` if the folder name does
        not start with ``<digits>_``.
    """
    match = re.match(r"^(\d+)_", class_dir_name)
    if match is None:
        return None
    return int(match.group(1))


def collect_video_samples(root_dir: Path) -> list[VideoSample]:
    """Walk ``root_dir`` and collect ``(video_dir, class_index)`` pairs.

    The function expects ``root_dir`` to contain class subfolders. Each class
    folder contains video subfolders, each holding the frame images.

    Class index resolution:
        * If the class folder name starts with a numeric prefix
          (e.g. ``"017_..."``), that prefix is used.
        * Otherwise, classes are indexed by their alphabetical order.

    Args:
        root_dir: Split root directory (e.g. ``processed_data/train``).

    Returns:
        List of ``(video_dir, class_index)`` tuples. Sorted deterministically
        by class folder, then by video folder name.

    Raises:
        FileNotFoundError: If ``root_dir`` does not exist.
        RuntimeError: If no video folders with frames were discovered.
    """
    root_dir = root_dir.resolve()
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root_dir}")

    samples: list[VideoSample] = []
    class_dirs = [p for p in sorted(root_dir.iterdir()) if p.is_dir()]

    fallback_index = {p.name: i for i, p in enumerate(class_dirs)}

    for class_dir in class_dirs:
        parsed = parse_class_index(class_dir.name)
        class_index = parsed if parsed is not None else fallback_index[class_dir.name]

        for video_dir in sorted(class_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            frame_paths = _list_frame_paths(video_dir)
            if len(frame_paths) == 0:
                continue
            samples.append((video_dir, class_index))

    if len(samples) == 0:
        raise RuntimeError(f"No video folders with frames under {root_dir}")

    return samples


def pick_frame_indices(num_available: int, num_frames: int) -> list[int]:
    """Pick ``num_frames`` evenly spaced frame indices in ``[0, num_available - 1]``.

    Behavior matches the professor baseline:
        * ``num_available == 1`` -> the single frame is repeated ``num_frames`` times.
        * ``num_available < num_frames`` -> indices may repeat (linspace rounding).

    Args:
        num_available: Number of frames available for the video.
        num_frames: Target number of frames to sample.

    Returns:
        List of integer frame indices, length ``num_frames``.

    Raises:
        ValueError: If ``num_available <= 0`` or ``num_frames <= 0``.
    """
    if num_available <= 0:
        raise ValueError("Video has no frames.")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")

    if num_available == 1:
        return [0] * num_frames

    positions = torch.linspace(0, num_available - 1, steps=num_frames)
    return [int(round(float(x))) for x in positions]


class VideoFrameDataset(Dataset):
    """Lazy dataset of fixed-length frame tensors per video folder.

    Args:
        root_dir: Split root containing class folders. Used only when
            ``sample_list`` is ``None``.
        num_frames: Number of frames ``T`` to sample per video.
        transform: Per-frame transform mapping a PIL ``Image`` to a
            ``(C, H, W)`` tensor (typically ``Resize`` + ``ToTensor`` + ``Normalize``).
        sample_list: Optional pre-built list of ``(video_dir, label)`` pairs.
            Useful for train/val splits.
    """

    def __init__(
        self,
        root_dir: str | Path,
        num_frames: int,
        transform: Callable[[Image.Image | Sequence[Image.Image]], torch.Tensor | list[torch.Tensor]],
        sample_list: list[VideoSample] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.transform = transform

        if sample_list is None:
            self.samples: list[VideoSample] = collect_video_samples(self.root_dir)
        else:
            self.samples = list(sample_list)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        video_dir, label = self.samples[index]
        frame_paths = _list_frame_paths(video_dir)
        indices = pick_frame_indices(len(frame_paths), self.num_frames)

        raw_frames: list[Image.Image] = []
        for frame_index in indices:
            path = frame_paths[frame_index]
            with Image.open(path) as image:
                raw_frames.append(image.convert("RGB"))

        try:
            transformed = self.transform(raw_frames)
            if isinstance(transformed, list):
                frames = transformed
            else:
                frames = [self.transform(frame) for frame in raw_frames]  # type: ignore[arg-type]
        except Exception:
            frames = [self.transform(frame) for frame in raw_frames]  # type: ignore[arg-type]

        video_tensor = torch.stack(frames, dim=0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return video_tensor, label_tensor
