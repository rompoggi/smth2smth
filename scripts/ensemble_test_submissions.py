#!/usr/bin/env python3
"""Step 6 — test-set predictions for MAE500 diverse heads (LB confirm).

``dump`` — cache per-member test logits (GPU; basic, no TTA; same forward as the
           val dump so val OOF and LB are apples-to-apples).
``csv``  — write submission CSVs: individuals (argmax) + ensembles (softmax-avg
           over members, the locked combiner) -> argmax.

All inference is basic (no TTA) on the full test set, to match the basic val OOF.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from smth2smth.ensemble.combiners import logits_to_probs, sanitize_logits
from smth2smth.ensemble.inference import TtaMode, collect_logits_for_videos
from smth2smth.shared.io.submission import discover_all_test_videos, write_submission_csv

DEFAULT_CACHE = REPO / "outputs/ensemble/mae500_stab_val/test_logits"
SUBMIT_DIR = REPO / "submissions"
DATE = "20260607"

# Individual members to submit (span archs + the Q8 seed triple).
INDIVIDUALS = [
    "meanpool-mae500-s42",
    "perceiverQ8-mae500-s42",
    "perceiverQ8-mae500-s43",
    "perceiverQ8-mae500-s44",
    "DivSpaceTimeK9-mae500-s42",
    "perceiverQ16-mae500-s43",
]

# Ensembles (softmax-avg over members), matching the val diversity sets.
ENSEMBLES = {
    "ens-seed-Q8x3": ["perceiverQ8-mae500-s42", "perceiverQ8-mae500-s43", "perceiverQ8-mae500-s44"],
    "ens-arch-s42": ["meanpool-mae500-s42", "perceiverQ8-mae500-s42", "DivSpaceTimeK9-mae500-s42"],
    "ens-allaxes": [
        "meanpool-mae500-s42",
        "perceiverQ8-mae500-s44",
        "DivSpaceTimeK9-mae500-s43",
        "perceiverQ16-mae500-s43",
    ],
    "ens-diverse4-s42": [
        "meanpool-mae500-s42",
        "perceiverQ8-mae500-s42",
        "DivSpaceTimeK9-mae500-s42",
        "perceiverQ16-mae500-s43",
    ],
}


def cmd_dump(args: argparse.Namespace) -> None:
    test_root = Path(args.test_dir).resolve()
    train_dir = Path(args.train_dir).resolve()
    cache = Path(args.cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    tta = TtaMode(args.tta)

    entries = json.loads(Path(args.members_manifest).read_text(encoding="utf-8"))
    names, dirs = discover_all_test_videos(test_root)
    samples = [(d, 0) for d in dirs]  # dummy labels; test is unlabelled
    (cache / "test_names.json").write_text(json.dumps(names), encoding="utf-8")
    print(f"[test-dump] {len(entries)} members; test N={len(names)}; tta={tta.value}")

    for e in entries:
        name = str(e["name"])
        out = cache / f"test_{name}.npy"
        if out.is_file() and not args.force:
            print(f"[test-dump] skip {name} (exists)")
            continue
        ckpt = Path(e["ckpt"])
        if not ckpt.is_absolute():
            ckpt = REPO / ckpt
        if not ckpt.is_file():
            print(f"[test-dump] MISSING ckpt for {name}: {ckpt}")
            continue
        print(f"[test-dump] {name} <- {ckpt.name} (tta={tta.value}) ...", flush=True)
        logits = collect_logits_for_videos(
            ckpt,
            samples,
            data_root=test_root,
            train_dir=train_dir,
            tta_mode=tta,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
        )
        np.save(out, logits.numpy())
    print(f"[test-dump] done -> {cache}")


def _load_test(cache: Path, name: str) -> np.ndarray:
    p = cache / f"test_{name}.npy"
    if not p.is_file():
        raise FileNotFoundError(f"missing test logits {p} — run: ensemble_test_submissions.py dump")
    return sanitize_logits(np.load(p))


def cmd_csv(args: argparse.Namespace) -> None:
    cache = Path(args.cache_dir).resolve()
    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    names = json.loads((cache / "test_names.json").read_text(encoding="utf-8"))

    tag_suffix = str(args.tag)
    written: list[tuple[str, Path]] = []
    for m in INDIVIDUALS:
        preds = _load_test(cache, m).argmax(axis=1).tolist()
        out = SUBMIT_DIR / f"track_a_single_{m}_{tag_suffix}_{DATE}.csv"
        write_submission_csv(out, names, preds)
        written.append((f"single:{m}", out))

    for name, members in ENSEMBLES.items():
        probs = np.mean([logits_to_probs(_load_test(cache, m)) for m in members], axis=0)
        preds = probs.argmax(axis=1).tolist()
        out = SUBMIT_DIR / f"track_a_{name}_softmax_{tag_suffix}_{DATE}.csv"
        write_submission_csv(out, names, preds)
        written.append((f"{name} ({len(members)} members)", out))

    print(f"[csv] wrote {len(written)} CSVs (N={len(names)} rows each) -> {SUBMIT_DIR}")
    for label, path in written:
        print(f"  {label:36s} {path.relative_to(REPO)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("dump", help="cache per-member test logits (GPU)")
    pd.add_argument("--members-manifest", type=Path, required=True)
    pd.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    pd.add_argument("--test-dir", type=Path, default=REPO / "data/test")
    pd.add_argument("--train-dir", type=Path, default=REPO / "data/train")
    pd.add_argument("--tta", choices=[m.value for m in TtaMode], default=TtaMode.CHAMPION.value)
    pd.add_argument("--force", action="store_true")
    pd.add_argument("--batch-size", type=int, default=16)
    pd.add_argument("--num-workers", type=int, default=8)
    pd.set_defaults(func=cmd_dump)

    pc = sub.add_parser("csv", help="write individual + ensemble submission CSVs")
    pc.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    pc.add_argument("--tag", default="champion", help="filename suffix (e.g. champion, basic)")
    pc.set_defaults(func=cmd_csv)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
