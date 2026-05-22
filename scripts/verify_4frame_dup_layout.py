#!/usr/bin/env python3
"""E6 open-verification gate: report the real per-clip temporal layout.

The Track B novelty hypothesis (see ``experiments_track_b_20052026.md`` §6) is
that the local dataset feeds *4 real frames duplicated into a 16-slot tensor*,
a zero-motion-delta token structure the SSv2-finetuned V-JEPA 2 head never saw.
Before building the re-downsampling pipeline (download_ssv2_subset_4frame.py)
we must confirm what the existing dataloader actually produces, because the
shared sampler (:func:`pick_frame_indices`) uses ``linspace`` rounding, not a
literal 4x duplication — so the assumption may or may not hold per clip.

This script loads one clip through the *real* ``VideoFrameDataset`` (same
sampler + transforms as training), then reports:

  * the source folder and number of frames on disk,
  * the frame indices the sampler picked,
  * the byte-identical duplication groups in the output tensor,
  * a verdict on whether the layout matches "K unique frames duplicated to T".

Usage::

    PYTHONPATH=src .venv/bin/python scripts/verify_4frame_dup_layout.py
    PYTHONPATH=src .venv/bin/python scripts/verify_4frame_dup_layout.py \\
        --data-dir data/train --num-frames 16 --image-size 256 --video <clip_dir>
    # Hard-fail unless the layout is exactly 4 unique frames duplicated to 16:
    PYTHONPATH=src .venv/bin/python scripts/verify_4frame_dup_layout.py \\
        --expect-unique 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smth2smth.shared.data.transforms import build_transforms  # noqa: E402
from smth2smth.shared.data.video_dataset import (  # noqa: E402
    VideoFrameDataset,
    collect_video_samples,
    pick_frame_indices,
)


def duplication_groups(clip: torch.Tensor) -> list[list[int]]:
    """Group output frame slots whose tensors are byte-identical.

    Args:
        clip: ``(T, C, H, W)`` clip tensor.

    Returns:
        Groups of frame-slot indices; each group is one set of equal frames,
        in first-seen order. ``[[0,1,2,3],[4,5,6,7],...]`` means 4x duplication.
    """
    groups: list[list[int]] = []
    for t in range(clip.size(0)):
        placed = False
        for group in groups:
            if torch.equal(clip[t], clip[group[0]]):
                group.append(t)
                placed = True
                break
        if not placed:
            groups.append([t])
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train", type=Path)
    parser.add_argument("--num-frames", default=16, type=int)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument(
        "--video",
        default=None,
        type=Path,
        help="Specific video folder. Defaults to the first collected clip.",
    )
    parser.add_argument(
        "--expect-unique",
        default=None,
        type=int,
        help="If set, exit non-zero unless the clip has exactly this many "
        "unique frames duplicated evenly to --num-frames.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    samples = collect_video_samples(data_dir)
    if args.video is not None:
        target = args.video.resolve()
        sample_list = [(target, 0)]
    else:
        sample_list = [samples[0]]
    video_dir = sample_list[0][0]

    transform = build_transforms(
        image_size=int(args.image_size),
        is_training=False,
        use_imagenet_norm=True,
    )
    dataset = VideoFrameDataset(
        root_dir=data_dir,
        num_frames=int(args.num_frames),
        transform=transform,
        sample_list=sample_list,
    )
    clip, _ = dataset[0]

    n_available = len(list(video_dir.glob("*.jpg"))) or len(
        [p for ext in ("*.jpeg", "*.png", "*.webp") for p in video_dir.glob(ext)]
    )
    picked = pick_frame_indices(max(1, n_available), int(args.num_frames))
    groups = duplication_groups(clip)
    n_unique = len(groups)
    group_sizes = sorted({len(g) for g in groups})

    print(f"clip folder       : {video_dir}")
    print(f"frames on disk    : {n_available}")
    print(f"requested frames T: {args.num_frames}")
    print(f"sampler indices   : {picked}")
    print(f"output tensor     : {tuple(clip.shape)}")
    print(f"unique frames     : {n_unique}")
    print(f"duplication groups: {groups}")
    print(f"group sizes seen  : {group_sizes}")

    even_dup = len(group_sizes) == 1 and n_unique * group_sizes[0] == int(args.num_frames)
    if even_dup:
        print(
            f"VERDICT: {n_unique} unique frames each duplicated "
            f"{group_sizes[0]}x into {args.num_frames} slots."
        )
    else:
        print(
            "VERDICT: NOT a clean even duplication — the linspace sampler "
            "produced uneven repeats (expected when frames-on-disk does not "
            "divide T)."
        )

    if args.expect_unique is not None:
        ok = even_dup and n_unique == int(args.expect_unique)
        print(
            f"CHECK --expect-unique={args.expect_unique}: "
            f"{'PASS' if ok else 'FAIL'}"
        )
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
