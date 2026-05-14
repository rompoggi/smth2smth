"""Zero-shot evaluation/submission helpers for V-JEPA 2 SSv2 checkpoints.

The Something-Something v2 finetuned V-JEPA 2 checkpoints on HuggingFace
Hub ship with a full 174-class SSv2 classifier head. Our challenge subset
is exactly 33 of those 174 SSv2 classes (with class 27 missing on disk),
so we can run zero-shot inference -- *no training* -- by:

    1. Loading ``transformers.VJEPA2ForVideoClassification.from_pretrained``.
    2. Auto-aligning our class folder names against the model's
       ``config.id2label`` to derive ``our_idx -> ssv2_idx``.
    3. Restricting the 174-d logits to the matched SSv2 indices (in our
       index order) and argmax/top-k in the resulting 32-d slice.

This module owns the *reusable* pieces of that pipeline:

* :func:`normalize_class_name` -- lower-case, strip leading ``<digits>_``,
  collapse to space-joined alphanumeric tokens.
* :func:`build_label_mapping` -- name-match our folder names to SSv2
  ``id2label`` entries, with a token-aligned unique-prefix fallback for
  filesystem-truncated names.
* :class:`VideoFramesDataset` -- read N frames per video folder, resize,
  ImageNet-normalize, stack to ``(N, 3, H, W)``.

Both the eval and submit scripts under ``scripts/`` import from here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from smth2smth.shared.data.video_dataset import pick_frame_indices

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

_WORD_RE = re.compile(r"[a-z0-9]+")
_LEADING_DIGITS_RE = re.compile(r"^\d+_")


def normalize_class_name(name: str) -> str:
    """Lower-case, drop leading ``<digits>_``, collapse to alphanumeric tokens.

    Example::

        "018_Pulling_something_from_left_to_right"
            -> "pulling something from left to right"
        "Pulling [Something] from Left to Right"
            -> "pulling something from left to right"
    """
    name = _LEADING_DIGITS_RE.sub("", name)
    return " ".join(_WORD_RE.findall(name.lower()))


def build_label_mapping(
    our_class_dirs: Iterable[Path],
    id2label: dict[int, str],
    *,
    log_fn: Any = print,
) -> tuple[dict[int, int], list[str]]:
    """Map our class indices to SSv2 indices via normalized-name matching.

    Matching strategy (in order, first hit wins):

    1. **Exact normalized match** -- both names lower-cased, leading
       ``<digits>_`` stripped, collapsed to alphanumeric tokens joined by
       single spaces.
    2. **Token-aligned unique-prefix match** -- our normalized name is a
       prefix of exactly one SSv2 normalized name with a space boundary.
       This handles filesystem-truncated folder names such as
       ``015_Pretending_to_pour_something_out_of_something_but_something_``
       which is a truncation of the SSv2 label
       ``'Pretending to pour [something] out of [something], but [something]
       is empty'``. Ambiguous prefix matches are rejected so we never
       silently align to the wrong class.

    Args:
        our_class_dirs: Sorted iterable of class folder paths (typically
            under ``data/train/`` or ``data/val/``). Class index is parsed
            from the leading numeric prefix.
        id2label: HuggingFace ``model.config.id2label``.
        log_fn: Callable used for diagnostic messages. Defaults to
            :func:`print`; tests pass a noop.

    Returns:
        ``(mapping, unmatched_names)`` where ``mapping`` is
        ``{our_idx: ssv2_idx}`` (only for matched classes) and
        ``unmatched_names`` is the list of our class folder names that
        could not be aligned.
    """
    ssv2_norm_to_idx: dict[str, int] = {}
    for ssv2_idx, label in id2label.items():
        norm = normalize_class_name(str(label))
        if norm in ssv2_norm_to_idx:
            log_fn(
                f"[map] WARNING: duplicate normalized SSv2 label {norm!r} "
                f"(kept idx {ssv2_norm_to_idx[norm]}, ignoring {int(ssv2_idx)})"
            )
            continue
        ssv2_norm_to_idx[norm] = int(ssv2_idx)

    def _unique_prefix(needle: str) -> int | None:
        """Return the SSv2 index whose norm uniquely starts with ``needle``."""
        candidates = [
            ssv2_idx
            for ssv2_norm, ssv2_idx in ssv2_norm_to_idx.items()
            if ssv2_norm == needle or ssv2_norm.startswith(needle + " ")
        ]
        return candidates[0] if len(candidates) == 1 else None

    mapping: dict[int, int] = {}
    unmatched: list[str] = []
    for class_dir in our_class_dirs:
        match = re.match(r"^(\d+)_(.+)$", class_dir.name)
        if match is None:
            unmatched.append(class_dir.name)
            continue
        our_idx = int(match.group(1))
        our_norm = normalize_class_name(class_dir.name)
        ssv2_idx = ssv2_norm_to_idx.get(our_norm)
        if ssv2_idx is None:
            ssv2_idx = _unique_prefix(our_norm)
            if ssv2_idx is not None:
                log_fn(
                    f"[map] prefix-matched our class {our_idx:03d} "
                    f"({class_dir.name!r}) to SSv2 idx {ssv2_idx} "
                    f"({id2label[ssv2_idx]!r})"
                )
        if ssv2_idx is None:
            unmatched.append(class_dir.name)
            continue
        mapping[our_idx] = ssv2_idx
    return mapping, unmatched


def _load_clip(
    video_dir: Path,
    num_frames: int,
    image_size: int,
) -> torch.Tensor:
    """Read ``num_frames`` frames from ``video_dir``, normalize, stack.

    Uses :func:`pick_frame_indices` for the same evenly-spaced sampling
    convention as the supervised pipeline.
    """
    frame_paths = sorted(video_dir.glob("*.jpg"), key=lambda p: p.name)
    if not frame_paths:
        raise RuntimeError(f"No JPG frames in {video_dir}")
    indices = pick_frame_indices(len(frame_paths), num_frames)
    frames: list[torch.Tensor] = []
    for frame_idx in indices:
        with Image.open(frame_paths[frame_idx]) as image:
            img = image.convert("RGB")
        img = TF.resize(img, [image_size, image_size])
        t = TF.to_tensor(img)
        t = TF.normalize(t, mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))
        frames.append(t)
    return torch.stack(frames, dim=0)


class VideoFramesDataset(Dataset):
    """Dataset that returns ``(clip_tensor, metadata)`` for each video.

    ``metadata`` is whatever the caller stores in ``meta_list[i]`` (typically
    an integer label for val, or a ``str`` video name for the test split).
    """

    def __init__(
        self,
        video_dirs: list[Path],
        meta_list: list[Any],
        num_frames: int,
        image_size: int,
    ) -> None:
        if len(video_dirs) != len(meta_list):
            raise ValueError(
                f"video_dirs ({len(video_dirs)}) and meta_list "
                f"({len(meta_list)}) must have the same length."
            )
        self.video_dirs = video_dirs
        self.meta_list = meta_list
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.video_dirs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Any]:
        clip = _load_clip(
            self.video_dirs[index],
            num_frames=self.num_frames,
            image_size=self.image_size,
        )
        return clip, self.meta_list[index]
