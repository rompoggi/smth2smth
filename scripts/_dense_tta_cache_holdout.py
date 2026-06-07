#!/usr/bin/env python3
"""FAST dense-TTA holdout cache (676 clips)."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from smth2smth.ensemble.inference import collect_logits_for_videos, TtaMode
from smth2smth.ensemble.holdout import build_official_val_holdout, write_holdout_manifest

ap = argparse.ArgumentParser()
ap.add_argument("--member", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--cache-dir", default=str(REPO / "outputs/ensemble/dense_tta"))
ap.add_argument("--data-dir", default=str(REPO / "data"))
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--num-workers", type=int, default=6)
args = ap.parse_args()

cache = Path(args.cache_dir).resolve()
cache.mkdir(parents=True, exist_ok=True)
val_dir = Path(args.data_dir) / "val"
train_dir = Path(args.data_dir) / "train"
ckpt = Path(args.ckpt).resolve()
m = args.member

holdout = build_official_val_holdout(val_dir, holdout_ratio=0.1, split_seed=42)
if not (cache / "labels_holdout.npy").is_file():
    np.save(cache / "labels_holdout.npy", np.array([lab for _, lab in holdout], dtype=np.int64))
    write_holdout_manifest(holdout, cache / "holdout_manifest.json")

print(f"[dense-cache] {m}: holdout N={len(holdout)} OFFICIAL_2X3 TTA ...", flush=True)
ho = collect_logits_for_videos(
    ckpt, list(holdout), data_root=val_dir, train_dir=train_dir,
    tta_mode=TtaMode.OFFICIAL_2X3, batch_size=args.batch_size, num_workers=args.num_workers,
)
np.save(cache / f"holdout_{m}.npy", ho.numpy())
print(f"[dense-cache] {m}: DONE holdout{tuple(ho.shape)}", flush=True)
