#!/usr/bin/env python3
"""Persist the local→SSv2 native-index map for the SSv2-FT head-slice path.

Reads ``data/train/`` (or ``--label-source-dir``) and the SSv2-finetuned
V-JEPA 2 ``id2label`` from ``--hf-repo``, builds the mapping via
:func:`smth2smth.track_b.zero_shot.build_label_mapping`, and saves a
``(num_classes,)`` ``torch.LongTensor`` to ``--output``. Unresolved local
indices (e.g. the missing 027 in the 32-class subset) are stored as ``-1``.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/build_local_to_ssv2_idx.py \\
        --output checkpoints/track_b/local_to_ssv2_idx.pt

The training builder in :mod:`smth2smth.track_b.vjepa2` derives the same map
on-the-fly from ``cfg.dataset.train_dir``; this script is only useful when
you want a precomputed artifact (e.g. for use outside Hydra).
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

from smth2smth.track_b.zero_shot import build_label_mapping, normalize_class_name  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-repo",
        default="facebook/vjepa2-vitl-fpc16-256-ssv2",
        help="V-JEPA 2 SSv2-finetuned HF repo (provides id2label).",
    )
    parser.add_argument(
        "--label-source-dir",
        default=str(REPO_ROOT / "data" / "train"),
        help="Directory whose subfolders are 'NNN_<class name>' (SSv2 label).",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=33,
        help="Width of the output index tensor (default 33; index 27 stays -1 for our subset).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "track_b" / "local_to_ssv2_idx.pt"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_dir = Path(args.label_source_dir).resolve()
    if not label_dir.is_dir():
        raise SystemExit(f"Label-source dir not found: {label_dir}")

    print(f"[load] {args.hf_repo}")
    from transformers import VJEPA2ForVideoClassification

    config = VJEPA2ForVideoClassification.from_pretrained(args.hf_repo).config
    id2label = {int(k): str(v) for k, v in config.id2label.items()}
    print(f"[load] num_labels={len(id2label)} hidden_size={config.hidden_size}")

    class_dirs = sorted(p for p in label_dir.iterdir() if p.is_dir())
    mapping, unmatched = build_label_mapping(class_dirs, id2label)
    print(f"[map] matched {len(mapping)}/{len(class_dirs)} local class folders to SSv2 idx")
    if unmatched:
        print(f"[map] UNMATCHED ({len(unmatched)}):")
        for name in unmatched:
            print(f"        - {name} (norm: {normalize_class_name(name)!r})")
        raise SystemExit("Aborting: at least one folder is unmapped.")

    idx_tensor = torch.full((int(args.num_classes),), -1, dtype=torch.long)
    for local_idx, ssv2_idx in mapping.items():
        if 0 <= local_idx < args.num_classes:
            idx_tensor[local_idx] = int(ssv2_idx)

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(idx_tensor, out_path)
    n_resolved = int((idx_tensor >= 0).sum().item())
    print(f"[done] wrote {out_path} ({n_resolved}/{args.num_classes} resolved)")
    for local_idx in range(int(args.num_classes)):
        ssv2_idx = int(idx_tensor[local_idx].item())
        name = id2label.get(ssv2_idx, "<unresolved>") if ssv2_idx >= 0 else "<unresolved>"
        print(f"  local {local_idx:>3d} -> ssv2 {ssv2_idx:>4d}  {name}")


if __name__ == "__main__":
    main()
