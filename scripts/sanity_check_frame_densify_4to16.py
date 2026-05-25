#!/usr/bin/env python3
"""Visual sanity check: 4 anchor frames -> 16 (duplicate vs interpolated).

Writes per-clip comparison PNGs and side-by-side MP4s under ``logs/``.

Example::

    PYTHONPATH=src .venv/bin/python scripts/sanity_check_frame_densify_4to16.py \\
        --data-root data/val --num-clips 6 --backend flow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, PillowWriter
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smth2smth.shared.data.frame_densify import (  # noqa: E402
    build_interpolator,
    duplicate_anchors_to_dense,
    interpolate_anchors_to_dense,
    load_anchor_frames,
)
from smth2smth.shared.data.video_dataset import collect_video_samples, pick_frame_indices  # noqa: E402

# Diverse classes for a quick qualitative pass (motion + pretend/pulling).
_DEFAULT_CLASS_PREFIXES: tuple[str, ...] = (
    "018_Pulling",
    "016_Pretending",
    "022_Putting",
    "011_Picking",
    "000_Closing",
    "028_Taking",
)


def _anchor_column_starts(indices: list[int]) -> set[int]:
    """Return columns where a new anchor index begins (first frame of each block)."""
    starts: set[int] = set()
    prev: int | None = None
    for col, idx in enumerate(indices):
        if idx != prev:
            starts.add(col)
            prev = idx
    return starts


def _pick_default_clips(data_root: Path, num_clips: int) -> list[Path]:
    """Return one video_dir per requested class prefix when possible."""
    samples = collect_video_samples(data_root)
    by_class: dict[str, list[Path]] = {}
    for video_dir, _label in samples:
        class_name = video_dir.parent.name
        by_class.setdefault(class_name, []).append(video_dir)

    chosen: list[Path] = []
    for prefix in _DEFAULT_CLASS_PREFIXES:
        if len(chosen) >= num_clips:
            break
        for class_name, dirs in sorted(by_class.items()):
            if class_name.startswith(prefix):
                chosen.append(dirs[0])
                break

    if len(chosen) < num_clips:
        for video_dir, _ in samples:
            if video_dir not in chosen:
                chosen.append(video_dir)
            if len(chosen) >= num_clips:
                break
    return chosen[:num_clips]


def _save_comparison_png(
    out_path: Path,
    *,
    duplicate: list[Image.Image],
    interpolated: list[Image.Image],
    duplicate_anchor_cols: set[int],
    interp_anchor_cols: set[int],
    title: str,
) -> None:
    """Two-row grid: top = duplicate-to-16, bottom = flow/blend interpolate-to-16."""
    n = len(duplicate)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.15, 4.2))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, (dup_img, interp_img) in enumerate(zip(duplicate, interpolated, strict=True)):
        for row, (img, row_label, anchor_cols) in enumerate(
            (
                (dup_img, "duplicate (dataloader rule)", duplicate_anchor_cols),
                (interp_img, "interpolated", interp_anchor_cols),
            ),
        ):
            ax = axes[row, col]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            is_anchor = col in anchor_cols
            border_color = "#2ca02c" if is_anchor else "#1f77b4"
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2.5 if is_anchor else 0.8)
            if col == 0:
                ax.set_ylabel(row_label, fontsize=8)
            ax.set_title(f"f{col}", fontsize=7)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_playback_mp4(out_path: Path, frames: list[Image.Image], *, fps: float, label: str) -> None:
    """Animate frames sequentially (easier to judge motion than the strip view)."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_axis_off()
    im = ax.imshow(frames[0])
    ax.set_title(label, fontsize=9)

    writer: FFMpegWriter | PillowWriter
    if FFMpegWriter.isAvailable():
        writer = FFMpegWriter(fps=fps, bitrate=1200)
        suffix = ".mp4"
    else:
        writer = PillowWriter(fps=fps)
        suffix = ".gif"

    path = out_path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(path), dpi=100):
        for frame in frames:
            im.set_data(np.asarray(frame.convert("RGB")))
            writer.grab_frame()
    plt.close(fig)
    print(f"  wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data" / "val",
        help="Split root (train or val) with class/video folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "logs" / "frame_densify_sanity_4to16",
        help="Directory for PNG / MP4 outputs.",
    )
    parser.add_argument("--num-clips", type=int, default=6, help="Number of clips to visualize.")
    parser.add_argument(
        "--backend",
        choices=("flow", "blend"),
        default="flow",
        help="Interpolator for the bottom row (flow = OpenCV Farneback warp).",
    )
    parser.add_argument("--fps", type=float, default=4.0, help="Playback FPS for MP4/GIF.")
    parser.add_argument(
        "--video-dirs",
        type=Path,
        nargs="*",
        default=None,
        help="Explicit video folders; overrides automatic class sampling.",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise SystemExit(f"data root not found: {data_root}")

    if args.video_dirs:
        clip_dirs = [p.resolve() for p in args.video_dirs]
    else:
        clip_dirs = _pick_default_clips(data_root, args.num_clips)

    interpolator = build_interpolator(args.backend)
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Backend: {args.backend}")
    print(f"Clips: {len(clip_dirs)}")
    print(f"Output: {out_dir}")

    for video_dir in clip_dirs:
        class_name = video_dir.parent.name
        clip_id = video_dir.name
        stem = f"{class_name}__{clip_id}"

        anchors = load_anchor_frames(video_dir)
        if len(anchors) != 4:
            print(f"  skip {stem}: expected 4 frames, got {len(anchors)}")
            continue

        duplicate = duplicate_anchors_to_dense(anchors)
        interpolated = interpolate_anchors_to_dense(anchors, interpolator)
        dup_indices = pick_frame_indices(len(anchors), len(duplicate))
        # Interpolated layout: I0, 4 mids, I1, 4 mids, I2, 4 mids, I3 -> anchors at 0,5,10,15.
        interp_anchor_cols = {0, 5, 10, 15}

        title = f"{class_name} / {clip_id}  (4 anchors -> 16)"
        png_path = out_dir / f"{stem}_compare.png"
        _save_comparison_png(
            png_path,
            duplicate=duplicate,
            interpolated=interpolated,
            duplicate_anchor_cols=_anchor_column_starts(dup_indices),
            interp_anchor_cols=interp_anchor_cols,
            title=title,
        )
        print(f"  wrote {png_path}")

        _save_playback_mp4(
            out_dir / f"{stem}_duplicate_playback",
            duplicate,
            fps=args.fps,
            label=f"{clip_id} — duplicate x16",
        )
        _save_playback_mp4(
            out_dir / f"{stem}_{args.backend}_playback",
            interpolated,
            fps=args.fps,
            label=f"{clip_id} — {args.backend} interpolate x16",
        )

    readme = out_dir / "README.txt"
    readme.write_text(
        "4-to-16 frame densify sanity check\n"
        "==================================\n"
        "Top row (*_compare.png): duplicate via pick_frame_indices(4, 16) — same as\n"
        "  dataset.num_frames=16 on 4-frame folders.\n"
        "Bottom row: 4 uniformly spaced mids per gap (OpenCV flow warp by default).\n"
        "Green border = original anchor frame.\n"
        "*_duplicate_playback / *_flow_playback: sequential animations.\n",
        encoding="utf-8",
    )
    print(f"Done. Artifacts in {out_dir}")


if __name__ == "__main__":
    main()
