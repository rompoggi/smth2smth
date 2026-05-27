#!/usr/bin/env python3
"""Diverse-heads ensemble (champion TTA) for Round-3 holdout members.

Mirrors the champion-TTA protocol of experiments/results_tta_ensembling.md, but
keyed by *member name* (arbitrary checkpoints) instead of the s{42,43,44} mae
pattern. Two stages:

  cache    cache champion-TTA logits for ONE member on the split-seed-42 holdout
           (N=676, for fitting) and on the test set (for the blend). Run on a GPU.
  combine  load all members' cached logits, report holdout metrics for
           singles + mean / softmax / WS, and write test submission CSVs.
           CPU-only; run after gathering all members' .npy onto one host.

Leakage note: the q16 seeds (123, 7) and mae500/k6 used *different* val-holdout
splits, so the split-seed-42 holdout is clean only for the seed-42 members. WS
(fit on holdout-42) is therefore reported but flagged; mean/softmax need no fit
and are leakage-free — prefer those for the actual submission.

Examples:
  PYTHONPATH=src uv run python scripts/ensemble_diverse_heads_round3.py cache \
    --member q16_s7 --ckpt checkpoints/track_a/videomaev2+ft/arch2-perceiver-q16-stab-s7.pt \
    --cache-dir outputs/ensemble/diverse_heads_r3

  PYTHONPATH=src uv run python scripts/ensemble_diverse_heads_round3.py combine \
    --cache-dir outputs/ensemble/diverse_heads_r3 \
    --members q16_s42 q16_seed42 q16_s7 mae500 k6_stab \
    --out-prefix submissions/track_a_ensemble_r3_20260527
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from smth2smth.ensemble.combiners import logits_to_probs, sanitize_logits
from smth2smth.ensemble.holdout import build_official_val_holdout, write_holdout_manifest
from smth2smth.ensemble.inference import TtaMode, collect_logits_for_videos
from smth2smth.ensemble.optimize import combine_logits, metrics_from_logits, optimize_mix_weights
from smth2smth.pipelines.submit import _build_untrained_mask, _resolve_test_videos
from smth2smth.shared.io.checkpoints import load_checkpoint
from smth2smth.shared.io.submission import write_submission_csv

HOLDOUT_SPLIT_SEED = 42
HOLDOUT_RATIO = 0.1


def _softmax_avg_logits(arrays: list[np.ndarray]) -> np.ndarray:
    return np.log(np.clip(np.mean([logits_to_probs(a) for a in arrays], axis=0), 1e-12, 1.0))


def cmd_cache(args: argparse.Namespace) -> None:
    cache = Path(args.cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    base = Path(args.data_dir)
    val_dir, train_dir, test_dir = base / "val", base / "train", base / "test"
    ckpt = Path(args.ckpt).resolve()
    if not ckpt.is_file():
        raise SystemExit(f"missing checkpoint {ckpt}")
    m = args.member

    holdout = build_official_val_holdout(
        val_dir, holdout_ratio=HOLDOUT_RATIO, split_seed=HOLDOUT_SPLIT_SEED
    )
    if len(holdout) < 100:
        raise SystemExit(f"holdout too small N={len(holdout)} (check {val_dir})")
    if not (cache / "labels_holdout.npy").is_file():
        np.save(cache / "labels_holdout.npy", np.array([lab for _, lab in holdout], dtype=np.int64))
        write_holdout_manifest(holdout, cache / "holdout_manifest.json")
    names, video_dirs = _resolve_test_videos(test_dir, None)
    if not (cache / "test_names.json").is_file():
        (cache / "test_names.json").write_text(json.dumps(names), encoding="utf-8")

    ho_out, te_out = cache / f"holdout_{m}.npy", cache / f"test_{m}.npy"
    if ho_out.is_file() and te_out.is_file() and not args.force:
        print(f"[cache] {m}: already present, skip")
        return

    print(f"[cache] {m}: holdout N={len(holdout)} champion TTA ...", flush=True)
    ho = collect_logits_for_videos(
        ckpt, list(holdout), data_root=val_dir, train_dir=train_dir,
        tta_mode=TtaMode.CHAMPION, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    np.save(ho_out, ho.numpy())
    print(f"[cache] {m}: test N={len(video_dirs)} champion TTA ...", flush=True)
    te = collect_logits_for_videos(
        ckpt, [(p, 0) for p in video_dirs], data_root=test_dir, train_dir=train_dir,
        tta_mode=TtaMode.CHAMPION, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    np.save(te_out, te.numpy())

    ck = load_checkpoint(ckpt, map_location="cpu")
    tci = (ck.get("extra") or {}).get("trained_class_indices")
    (cache / f"meta_{m}.json").write_text(
        json.dumps({"member": m, "ckpt": str(ckpt),
                    "trained_class_indices": tci, "num_classes": int(ho.shape[1])}),
        encoding="utf-8",
    )
    print(f"[cache] {m}: done holdout{tuple(ho.shape)} test{tuple(te.shape)}")


def cmd_combine(args: argparse.Namespace) -> None:
    cache = Path(args.cache_dir).resolve()
    members = list(args.members)
    labels = np.load(cache / "labels_holdout.npy")
    names = json.loads((cache / "test_names.json").read_text())

    ho = [sanitize_logits(np.load(cache / f"holdout_{m}.npy")) for m in members]
    te = [sanitize_logits(np.load(cache / f"test_{m}.npy")) for m in members]
    n = len(members)

    print("== singles (holdout-42 champion) ==")
    singles = {}
    for m, a in zip(members, ho):
        t1, t5, ce = metrics_from_logits(a, labels)
        singles[m] = t1
        print(f"   {m:14s} top1={t1:6.2f}  top5={t5:6.2f}  ce={ce:.4f}")

    mean_ho = combine_logits(ho, np.array([1.0 / n] * n))
    mt1, _, mce = metrics_from_logits(mean_ho, labels)
    print(f"== mean        holdout top1={mt1:6.2f}  ce={mce:.4f}")

    sm_ho = _softmax_avg_logits(ho)
    st1, _, sce = metrics_from_logits(sm_ho, labels)
    print(f"== softmax     holdout top1={st1:6.2f}  ce={sce:.4f}")

    ws = optimize_mix_weights(ho, labels, name="ws")
    wpairs = ", ".join(f"{m}={w:.3f}" for m, w in zip(members, ws.weights))
    print(f"== ws (LEAKY*) holdout top1={ws.top1:6.2f}  ce={ws.cross_entropy:.4f}  w=[{wpairs}]")
    print("   * WS fit on holdout-42 over-weights members that trained on those clips "
          "(seeds 123/7). Prefer mean/softmax for submission.")

    meta = json.loads((cache / f"meta_{members[0]}.json").read_text())
    mask = _build_untrained_mask(
        meta["trained_class_indices"], int(meta["num_classes"]), torch.device("cpu")
    )

    def _write(combined: np.ndarray, tag: str) -> None:
        lt = torch.from_numpy(combined.astype(np.float32))
        if mask is not None:
            lt = lt + mask
        preds = lt.argmax(dim=1).cpu().tolist()
        out = Path(f"{args.out_prefix}_{tag}.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        write_submission_csv(out, names, preds)
        c = collections.Counter(preds)
        top_cls, top_n = c.most_common(1)[0]
        print(f"[submit] {tag:8s} -> {out}  rows={len(preds)} distinct={len(c)} "
              f"top_class={top_cls}({100 * top_n / len(preds):.1f}%)")

    _write(combine_logits(te, np.array([1.0 / n] * n)), "mean")
    _write(_softmax_avg_logits(te), "softmax")
    _write(combine_logits(te, ws.weights), "ws")
    print("[done] members:", ", ".join(members))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("cache", help="cache champion-TTA holdout+test logits for one member")
    pc.add_argument("--member", required=True)
    pc.add_argument("--ckpt", required=True)
    pc.add_argument("--cache-dir", default=str(REPO / "outputs/ensemble/diverse_heads_r3"))
    pc.add_argument("--data-dir", default=str(REPO / "data"))
    pc.add_argument("--batch-size", type=int, default=8)
    pc.add_argument("--num-workers", type=int, default=4)
    pc.add_argument("--force", action="store_true")
    pc.set_defaults(func=cmd_cache)

    po = sub.add_parser("combine", help="combine cached logits -> holdout metrics + submission CSVs")
    po.add_argument("--cache-dir", default=str(REPO / "outputs/ensemble/diverse_heads_r3"))
    po.add_argument("--members", nargs="+", required=True)
    po.add_argument("--out-prefix", required=True)
    po.set_defaults(func=cmd_combine)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
