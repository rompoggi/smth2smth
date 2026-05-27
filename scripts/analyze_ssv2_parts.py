#!/usr/bin/env python3
"""Analyze SSv2 tar-part layout vs our 32-class subset.

The 20 Qualcomm archives are a **sequential byte split** of one tarball. Files
inside are named ``<id>.webm`` but **not** grouped by ID: part 00 already spans
IDs from 2 to 220846. Use ``--part-archive`` to list a downloaded chunk and
measure real class coverage (do not infer coverage from ID ranges).

Usage::

    PYTHONPATH=src .venv/bin/python scripts/analyze_ssv2_parts.py \\
        --annotations-dir data/ssv2/raw/annotations
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from smth2smth.shared.data.ssv2_extended import (  # noqa: E402
    TARGET_CLASS_DIR_NAMES,
    filter_records_to_target_classes,
    load_ssv2_records,
    local_class_dirs,
)

TOTAL_VIDEOS = 220_847
NUM_PARTS = 20


def part_index_for_video_id(video_id: str, num_parts: int = NUM_PARTS) -> int:
    """Map ``video_id`` to part ``0 .. num_parts-1`` by uniform ID ranges."""
    vid = int(video_id)
    if vid < 1 or vid > TOTAL_VIDEOS:
        return -1
    # Part p covers IDs [p * chunk + 1, (p+1) * chunk] approximately.
    chunk = TOTAL_VIDEOS / num_parts
    return min(num_parts - 1, int((vid - 1) / chunk))


def video_ids_in_tar(archive: Path) -> set[str]:
    """List ``*.webm`` stems from a single Qualcomm part file (gzip tar stream)."""
    import subprocess

    proc = subprocess.run(
        ["tar", "-tzf", str(archive)],
        capture_output=True,
        text=True,
        check=False,
    )
    ids: set[str] = set()
    for line in proc.stdout.splitlines():
        if line.endswith(".webm"):
            ids.add(Path(line).stem.split("/")[-1].replace(".webm", ""))
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-dir",
        default=REPO_ROOT / "data" / "ssv2" / "raw" / "annotations",
        type=Path,
    )
    parser.add_argument("--local-train", default=REPO_ROOT / "data" / "train", type=Path)
    parser.add_argument(
        "--part-archive",
        default=None,
        type=Path,
        help="Downloaded part file (e.g. archives/20bn-something-something-v2-00).",
    )
    parser.add_argument(
        "--report",
        default=REPO_ROOT / "outputs" / "ssv2_extended" / "part_analysis.json",
        type=Path,
    )
    args = parser.parse_args()

    annot = Path(args.annotations_dir).resolve()
    train_json = annot / "something-something-v2-train.json"
    val_json = annot / "something-something-v2-validation.json"
    if not train_json.is_file():
        train_json = annot / "labels" / "train.json"
        val_json = annot / "labels" / "validation.json"

    ref_dirs = local_class_dirs(Path(args.local_train).resolve())
    train_targets = filter_records_to_target_classes(load_ssv2_records(train_json), ref_dirs)
    val_targets = filter_records_to_target_classes(load_ssv2_records(val_json), ref_dirs)

    per_part_train: dict[int, Counter[str]] = defaultdict(Counter)
    per_part_val: dict[int, Counter[str]] = defaultdict(Counter)
    for vid, folder in train_targets.items():
        p = part_index_for_video_id(vid)
        if p >= 0:
            per_part_train[p][folder] += 1
    for vid, folder in val_targets.items():
        p = part_index_for_video_id(vid)
        if p >= 0:
            per_part_val[p][folder] += 1

    chunk = int(TOTAL_VIDEOS / NUM_PARTS)
    print(f"SSv2 videos (official): {TOTAL_VIDEOS} in {NUM_PARTS} byte-split parts")
    print(f"Our 32-class train JSON rows: {len(train_targets)}")
    print(f"Our 32-class val JSON rows  : {len(val_targets)}")
    print()

    if args.part_archive is not None:
        part_path = Path(args.part_archive).resolve()
        if not part_path.is_file():
            print(f"Missing part archive: {part_path}")
            return 1
        ids = video_ids_in_tar(part_path)
        in_part = {vid: cls for vid, cls in train_targets.items() if vid in ids}
        by_class = Counter(in_part.values())
        print(f"--- Tar listing: {part_path.name} ---")
        print(f"webm files in tar listing : {len(ids)}")
        print(f"32-class train clips found: {len(in_part)} ({100*len(in_part)/max(1,len(train_targets)):.1f}%)")
        print(f"classes with >=1 clip     : {len(by_class)}/32")
        for folder in sorted(TARGET_CLASS_DIR_NAMES):
            print(f"  {folder}: {by_class.get(folder, 0)}")
        print()

    print("(Heuristic ID-range split — NOT where tar files live; for reference only)")
    for p in range(NUM_PARTS):
        id_lo = p * chunk + 1
        id_hi = min(TOTAL_VIDEOS, (p + 1) * chunk)
        tr = per_part_train[p]
        va = per_part_val[p]
        n_classes = len(set(tr) | set(va))
        print(
            f"part {p:02d}  IDs ~{id_lo}-{id_hi}  "
            f"train={sum(tr.values())} val={sum(va.values())}  "
            f"classes_present={n_classes}/32"
        )

    print()
    print(
        "CONCLUSION: Parts are byte splits; webm IDs inside are shuffled across 1..220847.\n"
        "Inspect each downloaded part with --part-archive. Part 00 alone typically\n"
        "covers all 32 classes (~24% of train JSON rows); add parts for volume only."
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "total_videos": TOTAL_VIDEOS,
        "num_parts": NUM_PARTS,
        "ids_per_part_approx": chunk,
        "train_rows_32class": len(train_targets),
        "val_rows_32class": len(val_targets),
        "per_part_train": {str(p): dict(c) for p, c in per_part_train.items()},
        "per_part_val": {str(p): dict(c) for p, c in per_part_val.items()},
    }
    args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
