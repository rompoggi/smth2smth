#!/usr/bin/env python3
"""Zero-shot evaluation of V-JEPA 2 (SSv2-finetuned) on our val/ split.

See :mod:`smth2smth.track_b.zero_shot` for the matching strategy that
aligns our 33 folder names against the model's 174 SSv2 ``id2label``
entries. This script only handles the CLI + I/O + reporting layer.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/eval_vjepa2_ssv2_zero_shot.py
    PYTHONPATH=src .venv/bin/python scripts/eval_vjepa2_ssv2_zero_shot.py \\
        --hf-repo facebook/vjepa2-vitg-fpc64-384-ssv2 \\
        --image-size 384 --num-frames 32 --batch-size 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smth2smth.shared.data.video_dataset import (  # noqa: E402
    collect_video_samples,
)
from smth2smth.shared.engine.metrics import accuracy_topk  # noqa: E402
from smth2smth.track_b.zero_shot import (  # noqa: E402
    VideoFramesDataset,
    build_label_mapping,
    normalize_class_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-repo",
        default="facebook/vjepa2-vitl-fpc16-256-ssv2",
        help="HuggingFace V-JEPA 2 checkpoint (must be SSv2-finetuned).",
    )
    parser.add_argument("--val-dir", default=str(REPO_ROOT / "data" / "val"))
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap the number of validation samples (handy for a smoke run).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float16",
        help="Inference dtype. ``float16`` halves memory + roughly doubles speed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    val_dir = Path(args.val_dir).resolve()
    if not val_dir.is_dir():
        raise SystemExit(f"Val dir not found: {val_dir}")

    print(f"[load] {args.hf_repo}")
    from transformers import VJEPA2ForVideoClassification

    model = VJEPA2ForVideoClassification.from_pretrained(args.hf_repo)
    model.eval()
    device = torch.device(args.device)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    model.to(device=device, dtype=dtype)

    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    print(f"[load] backbone hidden_size={model.config.hidden_size} num_labels={len(id2label)}")

    class_dirs = sorted(p for p in val_dir.iterdir() if p.is_dir())
    mapping, unmatched = build_label_mapping(class_dirs, id2label)
    print(f"[map] matched {len(mapping)} / {len(class_dirs)} of our classes to SSv2 indices")
    if unmatched:
        print(
            f"[map] WARNING: {len(unmatched)} class folder(s) could not be "
            f"aligned; their videos will be excluded from the evaluation:"
        )
        for name in unmatched:
            print(f"        - {name}  (norm: {normalize_class_name(name)!r})")

    our_indices_sorted = sorted(mapping.keys())
    ssv2_keep_indices = torch.tensor(
        [mapping[i] for i in our_indices_sorted],
        dtype=torch.long,
        device=device,
    )
    our_idx_to_rank = {idx: rank for rank, idx in enumerate(our_indices_sorted)}

    all_samples = collect_video_samples(val_dir)
    samples = [(p, lbl) for (p, lbl) in all_samples if lbl in mapping]
    excluded = len(all_samples) - len(samples)
    if args.max_samples is not None:
        samples = samples[: int(args.max_samples)]
    print(
        f"[data] val samples used: {len(samples)} (excluded {excluded} from "
        f"unmatched classes; {len(all_samples)} total)"
    )

    dataset = VideoFramesDataset(
        video_dirs=[p for p, _ in samples],
        meta_list=[int(lbl) for _, lbl in samples],
        num_frames=int(args.num_frames),
        image_size=int(args.image_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    running_top1 = 0.0
    running_top5 = 0.0
    seen = 0
    per_class_correct: dict[int, int] = dict.fromkeys(our_indices_sorted, 0)
    per_class_total: dict[int, int] = dict.fromkeys(our_indices_sorted, 0)
    t0 = time.time()
    last_log = t0
    log_every = 10  # batches

    with torch.inference_mode():
        for step, (videos, labels_our) in enumerate(loader, start=1):
            videos = videos.to(device=device, dtype=dtype, non_blocking=True)
            labels_our = labels_our.to(device, non_blocking=True)

            outputs = model(pixel_values_videos=videos)
            logits_174 = outputs.logits
            logits_sub = logits_174.index_select(dim=1, index=ssv2_keep_indices)

            targets_rank = torch.tensor(
                [our_idx_to_rank[int(x)] for x in labels_our.cpu().tolist()],
                dtype=torch.long,
                device=device,
            )

            top1, top5 = accuracy_topk(logits_sub.float(), targets_rank, topk=(1, 5))
            batch_size = labels_our.size(0)
            running_top1 += float(top1.item()) * batch_size
            running_top5 += float(top5.item()) * batch_size
            seen += batch_size

            preds_rank = logits_sub.argmax(dim=1)
            correct_flags = preds_rank.eq(targets_rank).cpu().tolist()
            for c_label, ok in zip(labels_our.cpu().tolist(), correct_flags, strict=True):
                per_class_total[int(c_label)] += 1
                per_class_correct[int(c_label)] += int(ok)

            now = time.time()
            if step % log_every == 0 or now - last_log > 30.0:
                last_log = now
                speed = seen / max(1e-6, now - t0)
                print(
                    f"  step {step}/{len(loader)}  seen={seen}  "
                    f"top1={running_top1 / seen:.4f}  "
                    f"top5={running_top5 / seen:.4f}  "
                    f"speed={speed:.1f} vid/s"
                )

    if seen == 0:
        raise SystemExit("No samples processed; check --val-dir.")

    final_top1 = running_top1 / seen
    final_top5 = running_top5 / seen
    elapsed = time.time() - t0
    print()
    print("=" * 56)
    print(f"V-JEPA 2 zero-shot eval on {val_dir}")
    print(f"  hf_repo      : {args.hf_repo}")
    print(f"  num_frames   : {args.num_frames}")
    print(f"  image_size   : {args.image_size}")
    print(f"  dtype        : {args.dtype}")
    print(f"  samples used : {seen}")
    print(f"  matched cls  : {len(mapping)} / {len(class_dirs)}")
    print(f"  wall time    : {elapsed:.1f} s ({seen / max(1e-6, elapsed):.1f} vid/s)")
    print(f"  TOP-1        : {final_top1:.4f}")
    print(f"  TOP-5        : {final_top5:.4f}")
    print("=" * 56)

    print()
    print("Per-class top-1 (sorted by accuracy, ascending):")
    rows = []
    for class_idx in our_indices_sorted:
        if per_class_total[class_idx] == 0:
            continue
        our_name = next(
            (d.name for d in class_dirs if d.name.startswith(f"{class_idx:03d}_")),
            f"<class {class_idx}>",
        )
        rows.append(
            (
                class_idx,
                our_name,
                per_class_correct[class_idx] / max(1, per_class_total[class_idx]),
                per_class_total[class_idx],
                id2label[mapping[class_idx]],
            )
        )
    rows.sort(key=lambda r: (r[2], r[0]))
    for class_idx, our_name, acc, total, ssv2_name in rows:
        print(
            f"  [{class_idx:03d}] {acc:.3f}  ({total:5d} videos)  "
            f"{our_name}  ->  SSv2: {ssv2_name!r}"
        )


if __name__ == "__main__":
    main()
