#!/usr/bin/env python3
"""FAST dense-TTA test cache for one model. Skip holdout (already have it)."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from smth2smth.ensemble.inference import collect_logits_for_videos, TtaMode
from smth2smth.shared.io.checkpoints import load_checkpoint
from smth2smth.pipelines.submit import _resolve_test_videos

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
base = Path(args.data_dir)
test_dir = base / "test"; train_dir = base / "train"
ckpt = Path(args.ckpt).resolve()
m = args.member

names, video_dirs = _resolve_test_videos(test_dir, None)
if not (cache / "test_names.json").is_file():
    (cache / "test_names.json").write_text(json.dumps(names), encoding="utf-8")
print(f"[dense-cache] {m}: test N={len(video_dirs)} OFFICIAL_2X3 TTA ...", flush=True)
te = collect_logits_for_videos(
    ckpt, [(p, 0) for p in video_dirs], data_root=test_dir, train_dir=train_dir,
    tta_mode=TtaMode.OFFICIAL_2X3, batch_size=args.batch_size, num_workers=args.num_workers,
)
np.save(cache / f"test_{m}.npy", te.numpy())
ck = load_checkpoint(ckpt, map_location="cpu")
tci = (ck.get("extra") or {}).get("trained_class_indices")
(cache / f"meta_{m}.json").write_text(
    json.dumps({"member": m, "ckpt": str(ckpt),
                "trained_class_indices": tci, "num_classes": int(te.shape[1])}),
    encoding="utf-8",
)
print(f"[dense-cache] {m}: DONE test{tuple(te.shape)}", flush=True)
