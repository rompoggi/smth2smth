#!/usr/bin/env python3
"""Zero-shot Track-B submission with V-JEPA 2 (SSv2-finetuned).

Writes a Kaggle-ready ``video_name,predicted_class`` CSV by running the
SSv2-finetuned V-JEPA 2 classifier on ``data/test/`` and restricting the
174-class logits to the 33-class subset used in the competition.

No training, no checkpoint of ours -- the entire pipeline is a HuggingFace
``VJEPA2ForVideoClassification`` forward followed by a logit slice and
argmax. The label mapping is auto-derived from the
``data/train/`` folder names (which carry the canonical class index +
class name); we never look at the test labels.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/submit_vjepa2_ssv2_zero_shot.py
    PYTHONPATH=src .venv/bin/python scripts/submit_vjepa2_ssv2_zero_shot.py \\
        --hf-repo facebook/vjepa2-vitg-fpc64-384-ssv2 \\
        --image-size 384 --num-frames 32 --batch-size 1 \\
        --output submissions/track_b_vjepa2_vitg384_zero_shot.csv

By default the CSV is written to
``submissions/track_b_vjepa2_zero_shot.csv``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smth2smth.shared.io.submission import (  # noqa: E402
    SubmissionFormatError,
    discover_all_test_videos,
    validate_submission_csv,
    write_submission_csv,
)
from smth2smth.track_b.zero_shot import (  # noqa: E402
    VideoFramesDataset,
    build_label_mapping,
    normalize_class_name,
)


def _make_default_output(hf_repo: str) -> Path:
    """Default submission path under ``submissions/`` derived from the HF repo id."""
    tag = hf_repo.split("/")[-1].replace("-", "_")
    return REPO_ROOT / "submissions" / f"track_b_{tag}_zero_shot.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-repo",
        default="facebook/vjepa2-vitl-fpc16-256-ssv2",
        help="HuggingFace V-JEPA 2 checkpoint (must be SSv2-finetuned).",
    )
    parser.add_argument(
        "--label-source-dir",
        default=str(REPO_ROOT / "data" / "train"),
        help="Directory whose class folders define the (idx, name) mapping. "
        "Default: data/train/. We never look at labels here -- only "
        "folder names -- so val/ would work equivalently.",
    )
    parser.add_argument("--test-dir", default=str(REPO_ROOT / "data" / "test"))
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
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
    parser.add_argument(
        "--num-classes",
        type=int,
        default=33,
        help="Number of classes in the official taxonomy (used to validate the produced CSV).",
    )
    parser.add_argument(
        "--tta-flip",
        action="store_true",
        help="Also run a horizontally-flipped pass per clip and average "
        "softmax probabilities. Adds ~2x inference cost. The auto "
        "left/right class-pair remap mirrors what ``submit.py`` does "
        "for trained checkpoints (so pulling-L/R get re-aligned).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Destination CSV path. Default: submissions/track_b_<repo>_zero_shot.csv",
    )
    return parser.parse_args()


def _build_flip_perm(
    mapping: dict[int, int],
    our_class_dirs: list[Path],
    device: torch.device,
) -> torch.Tensor:
    """Permutation aligning a horizontally-flipped softmax to the original frame.

    Returns a ``LongTensor`` of length ``len(mapping)`` indexed in *rank*
    space (the position of the class in our sorted-class slice). For
    most classes the permutation is identity; for left/right paired
    classes (e.g. "Pulling something from left to right" and
    "Pulling something from right to left") it swaps them.
    """
    our_indices_sorted = sorted(mapping.keys())
    rank_of = {idx: rank for rank, idx in enumerate(our_indices_sorted)}
    stem_by_idx: dict[int, str] = {}
    for class_dir in our_class_dirs:
        norm = normalize_class_name(class_dir.name)
        match = norm.split()
        if not match:
            continue
        # Recover our integer index from the folder name's leading digits.
        prefix = class_dir.name.split("_", 1)[0]
        if not prefix.isdigit():
            continue
        idx = int(prefix)
        if idx not in mapping:
            continue
        stem_by_idx[idx] = norm

    perm_rank = list(range(len(our_indices_sorted)))
    paired: list[tuple[int, int]] = []
    LR = "from left to right"
    RL = "from right to left"
    for idx, stem in stem_by_idx.items():
        if LR in stem:
            mirror_stem = stem.replace(LR, RL)
            for other_idx, other_stem in stem_by_idx.items():
                if other_stem == mirror_stem and other_idx != idx:
                    perm_rank[rank_of[idx]] = rank_of[other_idx]
                    perm_rank[rank_of[other_idx]] = rank_of[idx]
                    paired.append((idx, other_idx))
                    break
    if paired:
        print(f"[tta] flip-pair remap: {paired} (others identity).")
    else:
        print("[tta] no left/right class pair detected; flip remap is identity.")
    return torch.tensor(perm_rank, dtype=torch.long, device=device)


def main() -> None:
    args = parse_args()

    label_dir = Path(args.label_source_dir).resolve()
    test_root = Path(args.test_dir).resolve()
    if not label_dir.is_dir():
        raise SystemExit(f"Label-source dir not found: {label_dir}")
    if not test_root.is_dir():
        raise SystemExit(f"Test dir not found: {test_root}")

    output_path = Path(args.output) if args.output else _make_default_output(args.hf_repo)
    output_path = output_path.resolve()

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

    class_dirs = sorted(p for p in label_dir.iterdir() if p.is_dir())
    mapping, unmatched = build_label_mapping(class_dirs, id2label)
    print(f"[map] matched {len(mapping)} / {len(class_dirs)} of our classes to SSv2 indices")
    if unmatched:
        print(
            f"[map] WARNING: {len(unmatched)} class folder(s) could not be aligned; "
            f"the model can never predict these classes on the test set:"
        )
        for name in unmatched:
            print(f"        - {name}  (norm: {normalize_class_name(name)!r})")
    if not mapping:
        raise SystemExit("Empty class mapping; aborting.")

    our_indices_sorted = sorted(mapping.keys())
    ssv2_keep_indices = torch.tensor(
        [mapping[i] for i in our_indices_sorted],
        dtype=torch.long,
        device=device,
    )
    rank_to_our_idx = torch.tensor(
        our_indices_sorted,
        dtype=torch.long,
        device=device,
    )

    flip_perm: torch.Tensor | None = None
    if args.tta_flip:
        flip_perm = _build_flip_perm(
            mapping=mapping,
            our_class_dirs=class_dirs,
            device=device,
        )

    print(f"[data] indexing test videos under {test_root}")
    video_names, video_dirs = discover_all_test_videos(test_root)
    print(f"[data] {len(video_names)} test videos.")

    dataset = VideoFramesDataset(
        video_dirs=video_dirs,
        meta_list=list(video_names),
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

    predictions_by_name: dict[str, int] = {}
    t0 = time.time()
    last_log = t0
    log_every = 20

    with torch.inference_mode():
        for step, (videos, names_batch) in enumerate(loader, start=1):
            videos = videos.to(device=device, dtype=dtype, non_blocking=True)

            outputs = model(pixel_values_videos=videos)
            logits_174 = outputs.logits.float()
            logits_sub = logits_174.index_select(dim=1, index=ssv2_keep_indices)
            probs = F.softmax(logits_sub, dim=1)

            if flip_perm is not None:
                flipped_videos = torch.flip(videos, dims=[-1])
                flipped_outputs = model(pixel_values_videos=flipped_videos)
                flipped_logits = flipped_outputs.logits.float().index_select(
                    dim=1, index=ssv2_keep_indices
                )
                flipped_probs = F.softmax(flipped_logits, dim=1).index_select(
                    dim=1, index=flip_perm
                )
                probs = (probs + flipped_probs) * 0.5

            preds_rank = probs.argmax(dim=1)
            preds_our_idx = rank_to_our_idx.index_select(dim=0, index=preds_rank)
            for name, pred in zip(names_batch, preds_our_idx.cpu().tolist(), strict=True):
                predictions_by_name[str(name)] = int(pred)

            now = time.time()
            if step % log_every == 0 or now - last_log > 30.0:
                last_log = now
                seen = len(predictions_by_name)
                speed = seen / max(1e-6, now - t0)
                print(
                    f"  step {step}/{len(loader)}  seen={seen}/{len(video_names)}  "
                    f"speed={speed:.1f} vid/s"
                )

    elapsed = time.time() - t0
    print(
        f"[done] processed {len(predictions_by_name)} videos in {elapsed:.1f} s "
        f"({len(predictions_by_name) / max(1e-6, elapsed):.1f} vid/s)"
    )

    if len(predictions_by_name) != len(video_names):
        raise SystemExit(
            f"Prediction count mismatch: have {len(predictions_by_name)} vs "
            f"{len(video_names)} expected."
        )

    ordered_predictions = [predictions_by_name[name] for name in video_names]
    csv_path = write_submission_csv(output_path, list(video_names), ordered_predictions)
    print(f"[csv ] wrote {len(ordered_predictions)} rows to {csv_path}")

    try:
        report = validate_submission_csv(
            csv_path,
            num_classes=int(args.num_classes),
            expected_video_names=set(video_names),
        )
        print(
            f"[csv ] validation OK: {report.num_rows} rows, "
            f"{report.unique_videos} unique video_name, "
            f"all predicted_class in [0, {args.num_classes})"
        )
    except SubmissionFormatError as exc:
        raise SystemExit(f"Submission CSV failed validation: {exc}") from exc

    print()
    print("Prediction histogram (predicted_class -> count):")
    counts: dict[int, int] = {}
    for pred in ordered_predictions:
        counts[pred] = counts.get(pred, 0) + 1
    for cls in sorted(counts):
        our_name = next(
            (d.name for d in class_dirs if d.name.startswith(f"{cls:03d}_")),
            f"<class {cls}>",
        )
        print(f"  {cls:>3d}  {counts[cls]:>5d}  {our_name}")


if __name__ == "__main__":
    main()
