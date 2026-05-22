#!/usr/bin/env python3
"""E6: build a Track B ``train_extra/`` set from full-SSv2 source clips.

Pulls Something-Something-v2 source videos whose *template* class matches one
of our local 32-class folders, re-downsamples each to the project's on-disk
format (the same N real frames the local ``train/`` clips store), and writes
them under ``data/train_extra/<NNN_Class>/<video_id>/frame_000.jpg ...``.

Why store N (=4) real frames and not 16: ``scripts/verify_4frame_dup_layout.py``
confirms the local ``train/`` clips store **4 real frames on disk**, and the
shared dataloader expands them to 16 via ``pick_frame_indices(4, 16)`` (linspace
rounding, *not* an even 4x duplication). Storing 4 frames here makes the extra
clips pass through the identical expansion at load time, so they land in exactly
the test-time distribution. ``--frames`` and ``--source-fraction`` are exposed
so the extraction can be re-aligned if the verification script reports a
different layout for your data.

This script is the I/O shell around three pure, unit-tested helpers
(:func:`select_class_targets`, :func:`extract_real_frames`,
:func:`write_clip_frames`). The webm decode step requires PyAV
(``uv add av``) and the SSv2 source archive (registration required); both are
intentionally absent from the default environment, so the decode path raises a
clear error if invoked without them. Defaults to ``--dry-run`` (print the plan,
write nothing).

Usage::

    # 1. confirm the local layout the extra clips must match
    PYTHONPATH=src .venv/bin/python scripts/verify_4frame_dup_layout.py

    # 2. dry-run the plan (no decode, no writes)
    PYTHONPATH=src .venv/bin/python scripts/download_ssv2_subset_4frame.py \\
        --ssv2-labels-json /path/to/ssv2/train.json \\
        --local-classes-dir data/train

    # 3. extract for real (needs PyAV + the SSv2 .webm directory)
    PYTHONPATH=src .venv/bin/python scripts/download_ssv2_subset_4frame.py \\
        --ssv2-labels-json /path/to/ssv2/train.json \\
        --ssv2-videos-dir /path/to/ssv2/20bn-something-something-v2 \\
        --local-classes-dir data/train --out-dir data/train_extra \\
        --limit-per-class 200 --no-dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smth2smth.shared.data.video_dataset import pick_frame_indices  # noqa: E402
from smth2smth.track_b.zero_shot import normalize_class_name  # noqa: E402


def select_class_targets(
    local_class_dirs: list[Path],
    ssv2_records: list[tuple[str, str]],
) -> dict[str, str]:
    """Map SSv2 video id -> local class folder name via normalized template match.

    Args:
        local_class_dirs: Our ``NNN_Class`` folders (e.g. under ``data/train``).
        ssv2_records: ``(video_id, template)`` pairs from the SSv2 label json,
            where ``template`` is the class template (e.g. "Closing [something]").

    Returns:
        ``{video_id: local_folder_name}`` for every SSv2 clip whose normalized
        template matches one of our local class folders. Clips that match no
        local class are dropped (we only want our 32-class subset).
    """
    local_by_norm: dict[str, str] = {
        normalize_class_name(d.name): d.name for d in local_class_dirs
    }
    targets: dict[str, str] = {}
    for video_id, template in ssv2_records:
        local_name = local_by_norm.get(normalize_class_name(template))
        if local_name is not None:
            targets[str(video_id)] = local_name
    return targets


def extract_real_frames(
    decoded_frames: list,
    num_frames: int,
    source_fraction: float = 0.6,
) -> list:
    """Pick ``num_frames`` real frames from the first ``source_fraction`` of a clip.

    Mirrors the professor's preprocessing (first ~60% of the source frames,
    subsampled to a fixed count) using the project's :func:`pick_frame_indices`
    over the windowed region so the sampling convention matches ``train/``.

    Args:
        decoded_frames: All decoded frames of the source video (any sequence;
            PIL images or arrays — only indexing is used here).
        num_frames: Number of real frames to keep (the local on-disk count).
        source_fraction: Fraction of the leading frames to sample from, in
            ``(0, 1]``.

    Returns:
        A list of ``num_frames`` frames selected from the window.

    Raises:
        ValueError: If ``decoded_frames`` is empty or ``source_fraction`` is
            out of range.
    """
    n = len(decoded_frames)
    if n == 0:
        raise ValueError("decoded_frames is empty.")
    if not 0.0 < source_fraction <= 1.0:
        raise ValueError(f"source_fraction must be in (0, 1], got {source_fraction}.")
    window = max(1, int(round(n * source_fraction)))
    indices = pick_frame_indices(window, num_frames)
    return [decoded_frames[i] for i in indices]


def write_clip_frames(frames: list, out_video_dir: Path) -> int:
    """Write ``frames`` as zero-padded ``frame_NNN.jpg`` files.

    Args:
        frames: PIL ``Image`` objects (RGB).
        out_video_dir: Destination folder (created if absent).

    Returns:
        Number of frames written.
    """
    out_video_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.convert("RGB").save(out_video_dir / f"frame_{i:03d}.jpg", quality=95)
    return len(frames)


def load_ssv2_records(labels_json: Path) -> list[tuple[str, str]]:
    """Load ``(video_id, template)`` pairs from an SSv2 label json.

    Accepts the official SSv2 ``train.json`` / ``validation.json`` schema: a
    list of objects with ``id`` and ``template`` (falling back to ``label``).
    """
    data = json.loads(labels_json.read_text())
    records: list[tuple[str, str]] = []
    for entry in data:
        vid = str(entry["id"])
        template = str(entry.get("template", entry.get("label", "")))
        records.append((vid, template))
    return records


def decode_video_frames(video_path: Path) -> list:
    """Decode all frames of a video to a list of PIL RGB images (lazy PyAV).

    Raises:
        ImportError: If PyAV is not installed (``uv add av``).
        FileNotFoundError: If ``video_path`` does not exist.
    """
    if not video_path.is_file():
        raise FileNotFoundError(f"SSv2 source video not found: {video_path}")
    try:
        import av  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "Decoding SSv2 .webm clips requires PyAV. Install it with "
            "`uv add av` (or `pip install av`), then re-run with --no-dry-run."
        ) from exc
    from PIL import Image

    frames: list = []
    with av.open(str(video_path)) as container:  # pragma: no cover - needs source data
        for frame in container.decode(video=0):
            frames.append(Image.fromarray(frame.to_ndarray(format="rgb24")))
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssv2-labels-json", required=True, type=Path)
    parser.add_argument("--local-classes-dir", default=Path("data/train"), type=Path)
    parser.add_argument(
        "--ssv2-videos-dir",
        default=None,
        type=Path,
        help="Directory of SSv2 source clips (e.g. <id>.webm). Required for --no-dry-run.",
    )
    parser.add_argument("--out-dir", default=Path("data/train_extra"), type=Path)
    parser.add_argument("--frames", default=4, type=int, help="Real frames stored per clip.")
    parser.add_argument("--source-fraction", default=0.6, type=float)
    parser.add_argument("--limit-per-class", default=None, type=int)
    parser.add_argument(
        "--video-ext", default=".webm", help="Source clip extension (default .webm)."
    )
    parser.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Actually decode and write frames (default: dry-run plan only).",
    )
    parser.set_defaults(dry_run=True)
    args = parser.parse_args()

    local_dirs = sorted(
        p for p in args.local_classes_dir.resolve().iterdir() if p.is_dir()
    )
    records = load_ssv2_records(args.ssv2_labels_json.resolve())
    targets = select_class_targets(local_dirs, records)

    per_class: dict[str, int] = {}
    plan: list[tuple[str, str]] = []
    for video_id, local_name in targets.items():
        count = per_class.get(local_name, 0)
        if args.limit_per_class is not None and count >= int(args.limit_per_class):
            continue
        per_class[local_name] = count + 1
        plan.append((video_id, local_name))

    print(f"local classes        : {len(local_dirs)}")
    print(f"ssv2 records          : {len(records)}")
    print(f"matched clips (subset): {len(targets)}")
    print(f"planned writes        : {len(plan)} (limit_per_class={args.limit_per_class})")
    print(f"out dir               : {args.out_dir.resolve()}")
    print(f"frames per clip       : {args.frames} (source_fraction={args.source_fraction})")

    if args.dry_run:
        print("DRY RUN: no frames decoded or written. Re-run with --no-dry-run to extract.")
        return 0

    if args.ssv2_videos_dir is None:
        parser.error("--ssv2-videos-dir is required with --no-dry-run.")

    written = 0
    for video_id, local_name in plan:
        src = args.ssv2_videos_dir.resolve() / f"{video_id}{args.video_ext}"
        decoded = decode_video_frames(src)
        frames = extract_real_frames(decoded, int(args.frames), float(args.source_fraction))
        out_video_dir = args.out_dir.resolve() / local_name / f"video_{video_id}"
        write_clip_frames(frames, out_video_dir)
        written += 1
        if written % 200 == 0:
            print(f"  ... wrote {written}/{len(plan)} clips")
    print(f"DONE: wrote {written} clips to {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
