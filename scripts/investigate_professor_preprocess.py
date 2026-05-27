#!/usr/bin/env python3
"""Empirically reverse-engineer professor on-disk JPEG preprocessing from SSv2 webm.

Compares local ``frame_*.jpg`` (already 224²) against SSv2 source frames under
several resize/crop hypotheses and professor temporal sampling (first ~40% → 4).

Usage::

    PYTHONPATH=src .venv/bin/python scripts/investigate_professor_preprocess.py
    PYTHONPATH=src .venv/bin/python scripts/investigate_professor_preprocess.py \\
        --video-id 278 --local-dir data/train/000_Closing_something/video_278
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageOps
from torchvision.transforms import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from smth2smth.shared.data.ssv2_extended import (  # noqa: E402
    compute_dhash,
    decode_video_frames,
    extract_professor_frames,
    hamming_distance,
)
from smth2smth.shared.data.video_dataset import pick_frame_indices  # noqa: E402

ResizeFn = Callable[[Image.Image, int], Image.Image]


def resize_squash(img: Image.Image, size: int) -> Image.Image:
    """Stretch to ``size×size`` (torchvision default path)."""
    return F.resize(img, [size, size])


def resize_short_side_center_crop(img: Image.Image, size: int) -> Image.Image:
    """Scale so min(H,W)=size, then center-crop square."""
    w, h = img.size
    if w <= h:
        new_w, new_h = size, int(round(h * size / w))
    else:
        new_w, new_h = int(round(w * size / h)), size
    x = F.resize(img, [new_h, new_w])
    return F.center_crop(x, [size, size])


def resize_long_side_center_crop(img: Image.Image, size: int) -> Image.Image:
    """Scale so max(H,W)=size, then center-crop square."""
    w, h = img.size
    if w >= h:
        new_w, new_h = size, int(round(h * size / w))
    else:
        new_w, new_h = int(round(w * size / h)), size
    x = F.resize(img, [new_h, new_w])
    return F.center_crop(x, [size, size])


def center_square_then_resize(img: Image.Image, size: int) -> Image.Image:
    """Center-crop source square, then resize."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    x = F.crop(img, top, left, side, side)
    return F.resize(x, [size, size])


def letterbox_pad(img: Image.Image, size: int) -> Image.Image:
    """Fit inside ``size×size`` preserving aspect, pad with black."""
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    x = F.resize(img, [new_h, new_w])
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    canvas.paste(x, (left, top))
    return canvas


RESIZE_METHODS: dict[str, ResizeFn] = {
    "squash_bicubic": resize_squash,
    "short_side_center_crop": resize_short_side_center_crop,
    "long_side_center_crop": resize_long_side_center_crop,
    "center_square_resize": center_square_then_resize,
    "letterbox_black": letterbox_pad,
}


def mse_rgb(a: Image.Image, b: Image.Image) -> float:
    """Mean squared error in RGB uint8 space."""
    aa = np.asarray(a.convert("RGB"), dtype=np.float32)
    bb = np.asarray(b.convert("RGB"), dtype=np.float32)
    if aa.shape != bb.shape:
        bb = np.asarray(b.convert("RGB").resize(a.size, Image.Resampling.BILINEAR), dtype=np.float32)
    return float(np.mean((aa - bb) ** 2))


def compare_clip(
    local_dir: Path,
    webm: Path,
    *,
    image_size: int,
    source_fraction: float,
) -> dict:
    """Score resize hypotheses for one clip."""
    local_paths = sorted(local_dir.glob("frame_*.jpg"))
    local_imgs = [Image.open(p).convert("RGB") for p in local_paths]
    decoded = decode_video_frames(webm)
    prof_frames = extract_professor_frames(
        decoded, num_frames=len(local_imgs), source_fraction=source_fraction
    )
    window = max(1, int(round(len(decoded) * source_fraction)))
    prof_idx = pick_frame_indices(window, len(local_imgs))

    results: dict = {
        "local_dir": str(local_dir),
        "webm": str(webm),
        "n_decoded": len(decoded),
        "native_size": list(decoded[0].size) if decoded else None,
        "local_size": list(local_imgs[0].size) if local_imgs else None,
        "professor_indices": prof_idx,
        "window_frames": window,
        "n_local_frames": len(local_imgs),
    }

    # Best hypothesis per (local_slot, resize_method) using dHash + MSE
    by_method: dict[str, list[dict]] = defaultdict(list)
    for method_name, fn in RESIZE_METHODS.items():
        resized = [fn(f, image_size) for f in prof_frames]
        for slot, (local, cand) in enumerate(zip(local_imgs, resized)):
            by_method[method_name].append(
                {
                    "slot": slot,
                    "prof_index": prof_idx[slot],
                    "mse": mse_rgb(local, cand),
                    "dhash_dist": hamming_distance(compute_dhash(local), compute_dhash(cand)),
                }
            )

    method_scores = []
    for method_name, rows in by_method.items():
        mse_mean = float(np.mean([r["mse"] for r in rows]))
        dhash_mean = float(np.mean([r["dhash_dist"] for r in rows]))
        method_scores.append(
            {"method": method_name, "mse_mean": mse_mean, "dhash_mean": dhash_mean, "per_slot": rows}
        )
    method_scores.sort(key=lambda r: (r["dhash_mean"], r["mse_mean"]))
    results["methods"] = method_scores
    results["best_method"] = method_scores[0]["method"] if method_scores else None

    # Frame-index alignment: does local slot k match prof index k or best other index?
    best_method = results["best_method"]
    fn = RESIZE_METHODS[best_method]
    alignment = []
    for slot, local in enumerate(local_imgs):
        best_j, best_mse, best_dh = -1, float("inf"), 999
        for j, raw in enumerate(decoded):
            cand = fn(raw, image_size)
            m = mse_rgb(local, cand)
            d = hamming_distance(compute_dhash(local), compute_dhash(cand))
            if d < best_dh or (d == best_dh and m < best_mse):
                best_j, best_mse, best_dh = j, m, d
        alignment.append(
            {
                "local_slot": slot,
                "prof_pick_index": prof_idx[slot],
                "best_raw_index": best_j,
                "best_dhash": best_dh,
                "best_mse": best_mse,
                "index_matches_pick": best_j == prof_idx[slot],
            }
        )
    results["index_alignment"] = alignment
    return results


def best_source_fraction(
    local_dir: Path,
    webm: Path,
    *,
    image_size: int,
    fractions: tuple[float, ...] = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6),
) -> tuple[float, float]:
    """Return (best_fraction, mean_dhash) for squash resize vs local JPEGs."""
    local_paths = sorted(local_dir.glob("frame_*.jpg"))
    if not local_paths or not webm.is_file():
        return fractions[0], float("inf")
    local_imgs = [Image.open(p).convert("RGB") for p in local_paths]
    decoded = decode_video_frames(webm)
    fn = resize_squash
    best_frac, best_score = fractions[0], float("inf")
    for frac in fractions:
        prof_frames = extract_professor_frames(
            decoded, num_frames=len(local_imgs), source_fraction=frac
        )
        resized = [fn(f, image_size) for f in prof_frames]
        score = float(
            np.mean(
                [
                    hamming_distance(compute_dhash(local), compute_dhash(cand))
                    for local, cand in zip(local_imgs, resized)
                ]
            )
        )
        if score < best_score:
            best_frac, best_score = frac, score
    return best_frac, best_score


def find_local_clip(train: Path, val: Path, video_id: str) -> Path | None:
    """Locate ``video_<id>`` under train or val."""
    name = f"video_{video_id}"
    for root in (train, val):
        if not root.is_dir():
            continue
        for class_dir in root.iterdir():
            cand = class_dir / name
            if cand.is_dir():
                return cand
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", default="278")
    parser.add_argument("--local-dir", type=Path, default=None)
    parser.add_argument(
        "--ssv2-videos-dir",
        type=Path,
        default=REPO_ROOT / "data/ssv2/raw/20bn-something-something-v2",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--source-fraction", type=float, default=0.4)
    parser.add_argument("--sample-ids", type=int, default=30, help="Also scan N overlapping local clips.")
    parser.add_argument(
        "--sweep-fractions",
        action="store_true",
        help="Per clip, pick best source_fraction in {0.3..0.6} before scoring resize.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "outputs/ssv2_extended/professor_preprocess_investigation.json",
    )
    args = parser.parse_args()

    videos_dir = args.ssv2_videos_dir
    nested = videos_dir / "20bn-something-something-v2"
    if nested.is_dir() and any(nested.glob("*.webm")):
        videos_dir = nested

    clips: list[tuple[Path, Path]] = []
    if args.local_dir is not None:
        webm = videos_dir / f"{args.video_id}.webm"
        clips.append((args.local_dir, webm))
    else:
        local = find_local_clip(REPO_ROOT / "data/train", REPO_ROOT / "data/val", args.video_id)
        if local is None:
            print(f"No local clip for id {args.video_id}")
            return 1
        clips.append((local, videos_dir / f"{args.video_id}.webm"))

    # Add more overlapping clips
    vid_re = re.compile(r"video_(\d+)$")
    on_disk = {p.stem for p in videos_dir.glob("*.webm")}
    for root in (REPO_ROOT / "data/train", REPO_ROOT / "data/val"):
        if not root.is_dir():
            continue
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            for vd in class_dir.iterdir():
                m = vid_re.match(vd.name)
                if not m or m.group(1) not in on_disk:
                    continue
                pair = (vd, videos_dir / f"{m.group(1)}.webm")
                if pair not in clips:
                    clips.append(pair)
                if len(clips) >= args.sample_ids + 1:
                    break
            if len(clips) >= args.sample_ids + 1:
                break

    all_results = []
    method_wins: dict[str, int] = defaultdict(int)
    fraction_wins: dict[float, int] = defaultdict(int)
    clip_list = clips[: args.sample_ids + 1]
    for local_dir, webm in clip_list:
        if not webm.is_file():
            continue
        frac = args.source_fraction
        if args.sweep_fractions:
            frac, _ = best_source_fraction(local_dir, webm, image_size=args.image_size)
            fraction_wins[frac] += 1
        r = compare_clip(
            local_dir,
            webm,
            image_size=args.image_size,
            source_fraction=frac,
        )
        r["source_fraction_used"] = frac
        all_results.append(r)
        if r.get("best_method"):
            method_wins[r["best_method"]] += 1
        print(f"\n=== {local_dir.name} (id={webm.stem}) ===")
        print(f"  decoded {r['n_decoded']} @ {r['native_size']} → local {r['local_size']}")
        used = r.get("source_fraction_used", args.source_fraction)
        print(f"  professor idx ({used:.0%} window): {r['professor_indices']}")
        print(f"  best resize: {r['best_method']}")
        top = r["methods"][0]
        print(f"    mse_mean={top['mse_mean']:.1f} dhash_mean={top['dhash_mean']:.1f}")
        for row in r["index_alignment"]:
            print(
                f"    slot {row['local_slot']}: pick={row['prof_pick_index']} "
                f"best_raw={row['best_raw_index']} dH={row['best_dhash']} "
                f"pick_ok={row['index_matches_pick']}"
            )

    summary = {
        "image_size": args.image_size,
        "source_fraction": args.source_fraction,
        "sweep_fractions": args.sweep_fractions,
        "fraction_wins": dict(fraction_wins) if args.sweep_fractions else None,
        "n_clips": len(all_results),
        "method_wins": dict(method_wins),
        "clips": all_results,
        "notes": (
            "Legacy eval transform uses F.resize to square (squash_bicubic). "
            "Professor on-disk JPEGs are 224²; compare with decode+resize hypotheses."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {args.report}")
    print("Method wins:", dict(method_wins))
    if args.sweep_fractions:
        print("Best source_fraction per clip:", dict(sorted(fraction_wins.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
