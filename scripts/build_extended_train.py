#!/usr/bin/env python3
"""Build ``data/extended_train/`` from full Something-Something v2 (32-class subset).

Steps (run in order)::

    # 1) Annotations from HuggingFace (videos still need manual Qualcomm download)
    PYTHONPATH=src .venv/bin/python scripts/build_extended_train.py download-annotations

    # 2) Per-class count table: local train+val vs SSv2 train+validation JSON
    PYTHONPATH=src .venv/bin/python scripts/build_extended_train.py class-stats

    # 3) Filter, dedupe against local train/val, extract 4 professor frames
    PYTHONPATH=src .venv/bin/python scripts/build_extended_train.py build \\
        --ssv2-videos-dir data/ssv2/raw/20bn-something-something-v2

    # 4) Prune extracted .webm folder to 32-class train+val IDs only
    PYTHONPATH=src .venv/bin/python scripts/build_extended_train.py \\
        --ssv2-videos-dir data/ssv2/raw/20bn-something-something-v2 \\
        prune-videos --execute

    # 5) Per-class overlap CSV + distribution figure
    PYTHONPATH=src .venv/bin/python scripts/build_extended_train.py \\
        --ssv2-videos-dir data/ssv2/raw/20bn-something-something-v2 \\
        overlap-stats --overlap-tag full
    PYTHONPATH=src .venv/bin/python scripts/plot_ssv2_extended_class_distribution.py \\
        --csv outputs/ssv2_extended/local_vs_ssv2_full_overlap.csv \\
        --output report/figures/ssv2_extended_class_distribution_full.png

    # 6) Spot-check professor transform vs a local clip
    PYTHONPATH=src .venv/bin/python scripts/build_extended_train.py verify-transform

See ``track_b_round1.md`` (Exp 3) and ``src/smth2smth/shared/data/ssv2_extended.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from smth2smth.shared.data.ssv2_extended import (  # noqa: E402
    TARGET_CLASS_DIR_NAMES,
    build_class_count_table,
    collect_local_frame_index,
    collect_target_video_ids,
    decode_video_frames,
    download_ssv2_annotations,
    extract_professor_frames,
    extract_uniform_frames,
    filter_records_to_target_classes,
    load_ssv2_records,
    local_class_dirs,
    reserved_ssv2_video_ids,
    ssv2_clip_overlaps_local,
    write_clip_frames,
)
from smth2smth.shared.data.video_dataset import collect_video_samples, pick_frame_indices  # noqa: E402

DEFAULT_SSV2_ROOT = REPO_ROOT / "data" / "ssv2" / "raw"
DEFAULT_ANNOT = DEFAULT_SSV2_ROOT / "annotations"
DEFAULT_OUT = REPO_ROOT / "data" / "extended_train"
DEFAULT_REPORT = REPO_ROOT / "outputs" / "ssv2_extended"


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "annot": Path(args.ssv2_annotations_dir).resolve(),
        "videos": Path(args.ssv2_videos_dir).resolve() if args.ssv2_videos_dir else None,
        "train": Path(args.local_train_dir).resolve(),
        "val": Path(args.local_val_dir).resolve(),
        "out": Path(args.out_dir).resolve(),
        "report": Path(args.report_dir).resolve(),
    }


def cmd_download_annotations(args: argparse.Namespace) -> int:
    paths = _paths(args)
    written = download_ssv2_annotations(paths["annot"])
    print(f"Downloaded {len(written)} annotation files to {paths['annot']}")
    for p in written:
        print(f"  {p.name}")
    print(
        "\nVideos are NOT on HuggingFace. After registering at Qualcomm, download the "
        "20 × ~1 GB parts and extract:\n"
        "  cat 20bn-something-something-v2-?? | tar -xzf - -C data/ssv2/raw/\n"
        "Expected layout: data/ssv2/raw/20bn-something-something-v2/<id>.webm"
    )
    return 0


def cmd_class_stats(args: argparse.Namespace) -> int:
    paths = _paths(args)
    train_json = paths["annot"] / "something-something-v2-train.json"
    val_json = paths["annot"] / "something-something-v2-validation.json"
    if not train_json.is_file() or not val_json.is_file():
        print("Missing annotations. Run: build_extended_train.py download-annotations")
        return 1

    rows = build_class_count_table(paths["train"], paths["val"], train_json, val_json)
    paths["report"].mkdir(parents=True, exist_ok=True)
    out_csv = paths["report"] / "class_counts_local_vs_ssv2.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "local_folder",
                "local_idx",
                "local_train",
                "local_val",
                "local_total",
                "ssv2_train_json",
                "ssv2_val_json",
                "ssv2_train_plus_val",
                "local_is_subset_of_ssv2",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.local_folder,
                    r.local_idx,
                    r.local_train,
                    r.local_val,
                    r.local_total,
                    r.ssv2_train_json,
                    r.ssv2_val_json,
                    r.ssv2_train_val_json,
                    r.is_subset,
                ]
            )

    total_local = sum(r.local_total for r in rows)
    total_ssv2 = sum(r.ssv2_train_val_json for r in rows)
    n_subset = sum(1 for r in rows if r.is_subset is True)
    n_violate = sum(1 for r in rows if r.is_subset is False)
    print(f"Wrote {out_csv}")
    print(f"classes tracked     : {len(rows)} (expect 32; index 027 absent)")
    print(f"local clips total   : {total_local}")
    print(f"ssv2 train+val rows : {total_ssv2} (33-class slice of official JSON)")
    print(f"per-class local<=ssv2: {n_subset} ok, {n_violate} VIOLATION (local > official)")
    if n_violate:
        print("Classes where local count exceeds SSv2 JSON (investigate mapping):")
        for r in rows:
            if r.is_subset is False:
                print(
                    f"  {r.local_folder}: local={r.local_total} "
                    f"ssv2={r.ssv2_train_val_json}"
                )
    return 0 if n_violate == 0 else 2


def cmd_build(args: argparse.Namespace) -> int:
    paths = _paths(args)
    if paths["videos"] is None or not paths["videos"].is_dir():
        print("ERROR: --ssv2-videos-dir must point at extracted .webm folder.")
        return 1
    paths = {**paths, "videos": _resolve_ssv2_videos_dir(paths["videos"])}

    train_json = paths["annot"] / "something-something-v2-train.json"
    val_json = paths["annot"] / "something-something-v2-validation.json"
    test_json = paths["annot"] / "something-something-v2-test.json"
    if not train_json.is_file():
        print("Missing train.json — run download-annotations first.")
        return 1

    ref_dirs = local_class_dirs(paths["train"]) or local_class_dirs(paths["val"])
    candidates = filter_records_to_target_classes(load_ssv2_records(train_json), ref_dirs)
    reserved = reserved_ssv2_video_ids(
        val_json,
        test_json,
        exclude_all_official_val=not args.allow_official_val_ids,
    )

    print("Indexing local train+val frames for deduplication …")
    local_index = collect_local_frame_index(
        paths["train"],
        paths["val"],
        max_clips=args.max_local_clips_for_index,
    )
    print(f"  {len(local_index)} local frames indexed")

    manifest: list[dict] = []
    removed_overlap = 0
    removed_reserved = 0
    written = 0
    ext = args.video_ext

    for i, (video_id, local_name) in enumerate(sorted(candidates.items()), start=1):
        if video_id in reserved:
            removed_reserved += 1
            continue

        src = paths["videos"] / f"{video_id}{ext}"
        if not src.is_file():
            continue

        decoded = decode_video_frames(src)
        overlap, reason = ssv2_clip_overlaps_local(
            decoded,
            local_index,
            max_hamming=int(args.dhash_max_hamming),
            check_uniform_16=not args.skip_uniform_16_check,
        )
        if overlap:
            removed_overlap += 1
            if len(manifest) < 20:
                manifest.append(
                    {"video_id": video_id, "status": "dropped_overlap", "reason": reason}
                )
            continue

        frames = extract_professor_frames(
            decoded,
            num_frames=int(args.frames),
            source_fraction=float(args.source_fraction),
        )
        out_dir = paths["out"] / local_name / f"video_{video_id}"
        write_clip_frames(frames, out_dir)
        written += 1
        if written <= 20:
            manifest.append({"video_id": video_id, "status": "kept", "class": local_name})

        if i % 500 == 0:
            print(f"  … scanned {i}/{len(candidates)} kept={written}")

    paths["report"].mkdir(parents=True, exist_ok=True)
    summary = {
        "candidates_train_json": len(candidates),
        "removed_reserved_ids": removed_reserved,
        "removed_overlap_local": removed_overlap,
        "written_clips": written,
        "out_dir": str(paths["out"]),
        "professor_transform": {
            "source_fraction": args.source_fraction,
            "num_frames": args.frames,
            "sampler": "pick_frame_indices(linspace)",
        },
        "dedupe": {
            "dhash_max_hamming": args.dhash_max_hamming,
            "check_uniform_16": not args.skip_uniform_16_check,
        },
    }
    summary_path = paths["report"] / "extended_train_build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (paths["report"] / "extended_train_manifest_sample.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Summary → {summary_path}")
    return 0


def _resolve_ssv2_videos_dir(videos_root: Path) -> Path:
    """Return directory containing ``<id>.webm`` (handles nested extract layout)."""
    if not videos_root.is_dir():
        return videos_root
    if any(videos_root.glob("*.webm")):
        return videos_root
    nested = videos_root / "20bn-something-something-v2"
    if nested.is_dir() and any(nested.glob("*.webm")):
        return nested
    return videos_root


def cmd_prune_videos(args: argparse.Namespace) -> int:
    """Delete ``.webm`` files not in our 32-class train+validation JSON subset."""
    paths = _paths(args)
    if paths["videos"] is None:
        print("ERROR: pass --ssv2-videos-dir (extracted SSv2 folder).")
        return 1

    train_json = paths["annot"] / "something-something-v2-train.json"
    val_json = paths["annot"] / "something-something-v2-validation.json"
    if not train_json.is_file() or not val_json.is_file():
        print("Missing train/validation JSON — run download-annotations first.")
        return 1

    videos_dir = _resolve_ssv2_videos_dir(paths["videos"])
    if not videos_dir.is_dir():
        print(f"Videos directory not found: {videos_dir}")
        return 1

    ref_dirs = local_class_dirs(paths["train"]) or local_class_dirs(paths["val"])
    keep_map = collect_target_video_ids(train_json, val_json, ref_dirs)
    keep_ids = set(keep_map)

    on_disk = sorted(videos_dir.glob(f"*{args.video_ext}"))
    to_remove = [p for p in on_disk if p.stem not in keep_ids]
    to_keep = [p for p in on_disk if p.stem in keep_ids]

    paths["report"].mkdir(parents=True, exist_ok=True)
    summary = {
        "videos_dir": str(videos_dir),
        "keep_ids_from_json": len(keep_ids),
        "on_disk_before": len(on_disk),
        "on_disk_keep": len(to_keep),
        "on_disk_remove": len(to_remove),
        "execute": bool(args.execute),
    }
    summary_path = paths["report"] / "prune_videos_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Videos dir          : {videos_dir}")
    print(f"32-class JSON IDs   : {len(keep_ids)} (train + validation)")
    print(f".webm on disk       : {len(on_disk)}")
    print(f"keep / remove       : {len(to_keep)} / {len(to_remove)}")
    print(f"Summary             : {summary_path}")

    if not args.execute:
        print("\nDry-run only. Re-run with --execute to delete removed files.")
        return 0

    removed = 0
    for p in to_remove:
        p.unlink(missing_ok=True)
        removed += 1
        if removed % 10000 == 0:
            print(f"  removed {removed}/{len(to_remove)} …")

    print(f"Done. Removed {removed} files; {len(to_keep)} remain.")
    return 0


def cmd_overlap_stats(args: argparse.Namespace) -> int:
    """Per-class overlap: local train+val vs pruned SSv2 webms on disk."""
    paths = _paths(args)
    if paths["videos"] is None:
        print("ERROR: pass --ssv2-videos-dir (extracted SSv2 folder).")
        return 1

    train_json = paths["annot"] / "something-something-v2-train.json"
    val_json = paths["annot"] / "something-something-v2-validation.json"
    if not train_json.is_file() or not val_json.is_file():
        print("Missing train/validation JSON — run download-annotations first.")
        return 1

    videos_dir = _resolve_ssv2_videos_dir(paths["videos"])
    ref_dirs = local_class_dirs(paths["train"]) or local_class_dirs(paths["val"])
    id_to_class = collect_target_video_ids(train_json, val_json, ref_dirs)

    on_disk_by_class: dict[str, set[str]] = {name: set() for name in TARGET_CLASS_DIR_NAMES}
    for p in videos_dir.glob(f"*{args.video_ext}"):
        folder = id_to_class.get(p.stem)
        if folder is not None:
            on_disk_by_class[folder].add(p.stem)

    local_by_class: dict[str, set[str]] = {name: set() for name in TARGET_CLASS_DIR_NAMES}
    vid_re = re.compile(r"video_(\d+)$")
    for root in (paths["train"], paths["val"]):
        if not root.is_dir():
            continue
        for class_dir in local_class_dirs(root):
            if class_dir.name not in local_by_class:
                continue
            for vd in class_dir.iterdir():
                m = vid_re.match(vd.name)
                if m and vd.is_dir():
                    local_by_class[class_dir.name].add(m.group(1))

    paths["report"].mkdir(parents=True, exist_ok=True)
    tag = str(args.overlap_tag).strip() or "full"
    out_csv = paths["report"] / f"local_vs_ssv2_{tag}_overlap.csv"

    rows_out: list[dict[str, int | str]] = []
    total_local = 0
    total_local_found = 0
    total_extra = 0
    for folder in sorted(TARGET_CLASS_DIR_NAMES):
        local_ids = local_by_class[folder]
        disk_ids = on_disk_by_class[folder]
        json_ids = {vid for vid, cls in id_to_class.items() if cls == folder}
        local_found = local_ids & disk_ids
        extra = disk_ids - local_ids
        rows_out.append(
            {
                "class": folder,
                "local_total": len(local_ids),
                "local_in_ssv2": len(local_found),
                "local_missing_ssv2": len(local_ids - disk_ids),
                "ssv2_on_disk_class": len(disk_ids),
                "extra_vs_local": len(extra),
                "json_class_total": len(json_ids),
                "json_on_disk": len(json_ids & disk_ids),
            }
        )
        total_local += len(local_ids)
        total_local_found += len(local_found)
        total_extra += len(extra)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "class",
                "local_total",
                "local_in_ssv2",
                "local_missing_ssv2",
                "ssv2_on_disk_class",
                "extra_vs_local",
                "json_class_total",
                "json_on_disk",
            ],
        )
        w.writeheader()
        w.writerows(rows_out)

    print(f"Wrote {out_csv}")
    print(f"local unique IDs     : {total_local}")
    print(f"local found on disk  : {total_local_found} ({100 * total_local_found / max(1, total_local):.1f}%)")
    print(f"local missing        : {total_local - total_local_found}")
    print(f"on-disk extras       : {total_extra}")
    print(f"JSON 32-class IDs    : {len(id_to_class)}")
    print(f"on disk (all classes): {sum(len(s) for s in on_disk_by_class.values())}")
    return 0


def cmd_verify_transform(args: argparse.Namespace) -> int:
    """Compare professor 4-frame indices on one local clip vs one SSv2 webm."""
    paths = _paths(args)
    samples = collect_video_samples(paths["train"])
    if not samples:
        print("No local train clips found.")
        return 1
    video_dir, _ = samples[0]
    local_paths = sorted(video_dir.glob("frame_*.jpg"))
    n_disk = len(local_paths)
    picked_16 = pick_frame_indices(max(1, n_disk), 16)
    print("Local clip (on disk)")
    print(f"  folder          : {video_dir}")
    print(f"  frames on disk  : {n_disk}")
    print(f"  dataloader T=16 : indices {picked_16}")

    if paths["videos"] is None:
        print("\nSkip SSv2 decode (--ssv2-videos-dir not set).")
        return 0

    videos_dir = _resolve_ssv2_videos_dir(paths["videos"])
    train_json = paths["annot"] / "something-something-v2-train.json"
    ref_dirs = local_class_dirs(paths["train"])
    candidates = filter_records_to_target_classes(load_ssv2_records(train_json), ref_dirs)
    if not candidates:
        print("No SSv2 candidates for verify.")
        return 1
    vid = None
    for candidate_id in sorted(candidates):
        if (videos_dir / f"{candidate_id}.webm").is_file():
            vid = candidate_id
            break
    if vid is None:
        print(f"No on-disk .webm for any train candidate under {videos_dir}")
        return 1
    src = videos_dir / f"{vid}.webm"
    if not src.is_file():
        print(f"SSv2 file missing: {src}")
        return 1
    decoded = decode_video_frames(src)
    prof = extract_professor_frames(
        decoded,
        num_frames=int(args.frames),
        source_fraction=float(args.source_fraction),
    )
    uni = extract_uniform_frames(decoded, 16)
    window = max(1, int(round(len(decoded) * float(args.source_fraction))))
    prof_idx = pick_frame_indices(window, int(args.frames))
    uni_idx = pick_frame_indices(len(decoded), 16)
    print("\nSSv2 clip (decoded)")
    print(f"  video_id        : {vid}")
    print(f"  decoded frames  : {len(decoded)}")
    print(f"  professor window: first {window} frames ({args.source_fraction:.0%})")
    print(f"  professor 4 idx : {prof_idx} (within window)")
    print(f"  uniform 16 idx  : {uni_idx} (full clip — GluonCV-style)")
    print(f"  wrote professor : {len(prof)} frames, uniform sample: {len(uni)} frames")
    print(
        "\nAt train time both local and extended_train clips use the same "
        "VideoFrameDataset: 4 real JPEGs expanded to T via pick_frame_indices."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ssv2-annotations-dir", default=DEFAULT_ANNOT, type=Path)
    parser.add_argument("--ssv2-videos-dir", default=None, type=Path)
    parser.add_argument("--local-train-dir", default=REPO_ROOT / "data" / "train", type=Path)
    parser.add_argument("--local-val-dir", default=REPO_ROOT / "data" / "val", type=Path)
    parser.add_argument("--out-dir", default=DEFAULT_OUT, type=Path)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT, type=Path)
    parser.add_argument("--frames", default=4, type=int)
    parser.add_argument("--source-fraction", default=0.4, type=float)
    parser.add_argument("--video-ext", default=".webm")
    parser.add_argument("--dhash-max-hamming", default=5, type=int)
    parser.add_argument(
        "--allow-official-val-ids",
        action="store_true",
        help="Do not blanket-drop all SSv2 validation video IDs (default: drop them).",
    )
    parser.add_argument(
        "--skip-uniform-16-check",
        action="store_true",
        help="Dedupe only via professor 4-frame hashes (faster, less conservative).",
    )
    parser.add_argument(
        "--max-local-clips-for-index",
        default=None,
        type=int,
        help="Debug: cap local frames indexed for dedupe.",
    )

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download-annotations", help="Fetch JSON from HuggingFace.")
    sub.add_parser("class-stats", help="CSV: local vs SSv2 per-class counts.")
    sub.add_parser("build", help="Dedupe + extract to extended_train/.")
    p_prune = sub.add_parser(
        "prune-videos",
        help="Drop .webm not in 32-class train+validation JSON subset.",
    )
    p_prune.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete files (default: dry-run counts only).",
    )
    sub.add_parser("verify-transform", help="Print sampling indices for one clip pair.")
    p_overlap = sub.add_parser(
        "overlap-stats",
        help="CSV: local vs on-disk SSv2 overlap per class (for distribution plots).",
    )
    p_overlap.add_argument(
        "--overlap-tag",
        default="full",
        help="Suffix for outputs/ssv2_extended/local_vs_ssv2_<tag>_overlap.csv",
    )

    args = parser.parse_args()
    if args.command == "download-annotations":
        return cmd_download_annotations(args)
    if args.command == "class-stats":
        return cmd_class_stats(args)
    if args.command == "build":
        return cmd_build(args)
    if args.command == "prune-videos":
        return cmd_prune_videos(args)
    if args.command == "verify-transform":
        return cmd_verify_transform(args)
    if args.command == "overlap-stats":
        return cmd_overlap_stats(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
