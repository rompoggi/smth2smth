#!/usr/bin/env python3
"""Track A VideoMAE 3-seed ensemble ablation (cache / optimize / submit)."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from smth2smth.ensemble.combiners import run_combiner, sanitize_logits
from smth2smth.ensemble.holdout import build_official_val_holdout, write_holdout_manifest
from smth2smth.ensemble.inference import TtaMode, collect_logits_and_probs_for_videos
from smth2smth.pipelines.submit import _resolve_test_videos

SEEDS = (42, 43, 44)
CHECKPOINT_PATTERN = re.compile(r"^s(\d+)_ft_val90_holdout\.pt$")
V1_CACHE = REPO_ROOT / "outputs/ensemble/videomaev2_3seed"
V2_CACHE_DEFAULT = REPO_ROOT / "outputs/ensemble/videomaev2_3seed_v2"
V3_CACHE_DEFAULT = REPO_ROOT / "outputs/ensemble/videomaev2_3seed_v3_champion"
CLUSTER_DATA = Path("/Data/thomas.turkieh/smth2smth/data")
RESULTS_JSON = "ensemble_results.json"

COMBINERS = ("mean", "softmax", "vote", "ws", "cws", "lsg")
PROTOCOLS = {
    "T": ("tta", "tta"),
    "BT": ("basic", "tta"),
    "B": ("basic", "basic"),
}


def _default_data_dir() -> Path:
    if (CLUSTER_DATA / "val").is_dir() and len(list((CLUSTER_DATA / "val").iterdir())) > 100:
        return CLUSTER_DATA
    return REPO_ROOT / "data"


def _resolve_data_roots(val_dir: Path | None, train_dir: Path | None, test_dir: Path | None):
    """Prefer cluster paths when repo ``data/`` is empty or missing."""
    val = Path(val_dir) if val_dir else _default_data_dir() / "val"
    train = Path(train_dir) if train_dir else _default_data_dir() / "train"
    test = Path(test_dir) if test_dir else _default_data_dir() / "test"
    if not val.is_dir() or len(list(val.iterdir())) < 100:
        val = CLUSTER_DATA / "val"
        train = CLUSTER_DATA / "train"
    if test_dir is None and (not test.is_dir() or len(list(test.iterdir())) < 100):
        test = CLUSTER_DATA / "test"
    return val.resolve(), train.resolve(), test.resolve()


def _discover_checkpoints(checkpoints_dir: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for path in sorted(checkpoints_dir.glob("s*_ft_val90_holdout.pt")):
        m = CHECKPOINT_PATTERN.match(path.name)
        if m:
            out[int(m.group(1))] = path.resolve()
    return out


def _tta_tag(cache_dir: Path) -> str:
    meta = cache_dir / "cache_meta.json"
    if meta.is_file():
        return json.loads(meta.read_text(encoding="utf-8")).get("tta_tag", "champion")
    return "champion"


def _logits_path(cache_dir: Path, seed: int, branch: str, *, tta_tag: str) -> Path:
    if branch == "basic":
        return cache_dir / f"logits_s{seed}_none.npy"
    return cache_dir / f"logits_s{seed}_{tta_tag}.npy"


def _probs_path(cache_dir: Path, seed: int, *, tta_tag: str) -> Path:
    return cache_dir / f"probs_s{seed}_{tta_tag}_probs.npy"


def _load_branch(
    cache_dir: Path,
    members: list[int],
    branch: str,
    *,
    tta_tag: str,
) -> tuple[list[np.ndarray], list[np.ndarray] | None]:
    from smth2smth.ensemble.combiners import logits_to_probs

    logits = [
        sanitize_logits(np.load(_logits_path(cache_dir, s, branch, tta_tag=tta_tag)))
        for s in members
    ]
    probs = None
    if branch == "tta":
        prob_paths = [_probs_path(cache_dir, s, tta_tag=tta_tag) for s in members]
        if all(p.is_file() for p in prob_paths):
            probs = [np.load(p) for p in prob_paths]
        else:
            probs = [logits_to_probs(arr) for arr in logits]
    return logits, probs


def _seed_holdout_samples(val_dir: Path, split_seed: int, holdout_ratio: float):
    return build_official_val_holdout(val_dir, holdout_ratio=holdout_ratio, split_seed=split_seed)


def cmd_cache(args: argparse.Namespace) -> None:
    ckpts = _discover_checkpoints(Path(args.checkpoints_dir))
    missing = [s for s in SEEDS if s not in ckpts]
    if missing:
        raise SystemExit(f"Missing checkpoints for seeds {missing}")

    val_dir = Path(args.val_dir).resolve()
    train_dir = Path(args.train_dir).resolve()
    test_root = Path(args.test_dir).resolve() if args.test_dir else None
    out_dir = Path(args.cache_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    holdout = _seed_holdout_samples(val_dir, int(args.holdout_split_seed), float(args.holdout_ratio))
    if len(holdout) < 100:
        raise SystemExit(
            f"Holdout too small (N={len(holdout)}); check --val-dir={val_dir}"
        )
    write_holdout_manifest(holdout, out_dir / "holdout_manifest.json")
    np.save(out_dir / "labels_holdout.npy", np.array([lab for _, lab in holdout], dtype=np.int64))
    print(f"[cache] holdout N={len(holdout)} split_seed={args.holdout_split_seed}")

    tta_tag = str(args.tta_tag)
    tta_mode = TtaMode(str(args.tta_mode))

    for src_dir in (V1_CACHE, V2_CACHE_DEFAULT):
        if not src_dir.is_dir():
            continue
        for seed in SEEDS:
            for name in (f"logits_s{seed}_none.npy", f"logits_s{seed}_{tta_tag}.npy"):
                src, dst = src_dir / name, out_dir / name
                if src.is_file() and not dst.is_file() and not args.force:
                    shutil.copy2(src, dst)
                    print(f"[cache] copied {dst.name} from {src_dir.name}")
            for split in ("", "test_"):
                name = f"logits_{split}s{seed}_{tta_tag}.npy"
                src, dst = src_dir / name, out_dir / name
                if src.is_file() and not dst.is_file() and not args.force:
                    shutil.copy2(src, dst)
                    print(f"[cache] copied {dst.name} from {src_dir.name}")

    samples_holdout = list(holdout)
    test_samples = None
    if args.also_test and test_root.is_dir():
        _names, video_dirs = _resolve_test_videos(test_root, None)
        test_samples = [(p, 0) for p in video_dirs]
        print(f"[cache] test N={len(test_samples)}")

    for seed in SEEDS:
        ckpt = ckpts[seed]
        for split_name, samples, root in [
            ("holdout", samples_holdout, val_dir),
            *(
                [("test", test_samples, test_root)]
                if test_samples is not None and test_root is not None
                else []
            ),
        ]:
            prefix = "logits" if split_name == "holdout" else "logits_test"
            log_out = out_dir / f"{prefix}_s{seed}_{tta_tag}.npy"
            if log_out.is_file() and not args.force:
                print(f"[cache] skip {split_name} seed={seed} (exists)")
                continue
            print(f"[cache] {split_name} seed={seed} {tta_tag} ({tta_mode.value}) ...", flush=True)
            if tta_mode in (TtaMode.OFFICIAL_2X3, TtaMode.DENSE_2X3):
                logits, probs = collect_logits_and_probs_for_videos(
                    ckpt,
                    samples,
                    data_root=root,
                    train_dir=train_dir,
                    tta_mode=tta_mode,
                    batch_size=int(args.batch_size),
                    num_workers=int(args.num_workers),
                )
                prob_out = out_dir / (
                    f"probs_s{seed}_{tta_tag}_probs.npy"
                    if split_name == "holdout"
                    else f"probs_test_s{seed}_{tta_tag}_probs.npy"
                )
                np.save(prob_out, probs.numpy())
            else:
                from smth2smth.ensemble.inference import collect_logits_for_videos

                logits = collect_logits_for_videos(
                    ckpt,
                    samples,
                    data_root=root,
                    train_dir=train_dir,
                    tta_mode=tta_mode,
                    batch_size=int(args.batch_size),
                    num_workers=int(args.num_workers),
                )
            np.save(log_out, logits.numpy())
            print(f"[cache] wrote {log_out.name} {tuple(logits.shape)}")

    meta = {
        "checkpoints": {str(k): str(v) for k, v in sorted(ckpts.items())},
        "tta_mode": tta_mode.value,
        "tta_tag": tta_tag,
    }
    (out_dir / "cache_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _v2_grid_specs() -> list[tuple[str, str | None, str | None, str]]:
    """(exp_id, fit_branch, eval_branch, combiner). Singles use combiner ``single``."""
    rows: list[tuple[str, str | None, str | None, str]] = []
    for seed in SEEDS:
        rows.append((f"S0-s{seed}-B", "basic", "basic", "single"))
        rows.append((f"S1-s{seed}-T", "tta", "tta", "single"))
    for proto in PROTOCOLS:
        fit_b, eval_b = PROTOCOLS[proto]
        for comb in COMBINERS:
            rows.append((f"{proto}-{comb}", fit_b, eval_b, comb))
    return rows


def cmd_optimize(args: argparse.Namespace) -> None:
    cache_dir = Path(args.cache_dir).resolve()
    tta_tag = _tta_tag(cache_dir)
    labels = np.load(cache_dir / "labels_holdout.npy")

    rows: list[dict] = []
    best_top1 = -1.0
    best_id = ""

    for exp_id, fit_branch, eval_branch, combiner in _v2_grid_specs():
        assert fit_branch is not None and eval_branch is not None
        if combiner == "single":
            seed = int(exp_id.split("-")[1].replace("s", ""))
            members = [seed]
            eval_arrays, eval_probs = _load_branch(
                cache_dir, members, eval_branch, tta_tag=tta_tag
            )
            fit_arrays = eval_arrays
            fit_probs = eval_probs
        else:
            members = list(SEEDS)
            fit_arrays, fit_probs = _load_branch(
                cache_dir, members, fit_branch, tta_tag=tta_tag
            )
            eval_arrays, eval_probs = _load_branch(
                cache_dir, members, eval_branch, tta_tag=tta_tag
            )

        comb = "mean" if combiner == "single" else combiner
        r = run_combiner(
            comb,
            fit_arrays,
            eval_arrays,
            eval_probs,
            labels,
            name=exp_id,
        )
        w_serial: dict | list
        if isinstance(r.weights, np.ndarray) and r.weights.ndim == 2:
            w_serial = r.weights.tolist()
        elif combiner != "single":
            w_serial = {str(s): float(r.weights[i]) for i, s in enumerate(members)}
        else:
            w_serial = {str(members[0]): 1.0}

        proto_tag = exp_id.split("-")[0] if combiner != "single" else exp_id[:2]
        row = {
            "exp": exp_id,
            "protocol": proto_tag,
            "combiner": combiner,
            "fit": fit_branch,
            "eval": eval_branch,
            "top1": round(r.top1, 4),
            "top5": round(r.top5, 4),
            "ce": round(r.cross_entropy, 6),
            "weights": w_serial,
        }
        rows.append(row)
        print(
            f"{exp_id:14s}  top1={r.top1:6.2f}%  top5={r.top5:6.2f}%  "
            f"ce={r.cross_entropy:.4f}"
        )
        if (
            r.top1 > best_top1
            and combiner not in ("single", "lsg")
            and exp_id[0] in "TB"
        ):
            best_top1 = r.top1
            best_id = exp_id

    out_path = cache_dir / RESULTS_JSON
    payload = {"holdout_metrics": rows, "best_exp": best_id, "best_top1": best_top1}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[optimize] wrote {out_path}")
    print(f"[optimize] best ensemble candidate: {best_id} ({best_top1:.2f}%)")


def cmd_submit(args: argparse.Namespace) -> None:
    from smth2smth.ensemble.submit import run_ensemble_submit_v2

    cache_dir = Path(args.cache_dir).resolve()
    exp_id = args.exp
    if exp_id == "best":
        data = json.loads((cache_dir / RESULTS_JSON).read_text(encoding="utf-8"))
        exp_id = data["best_exp"]
        print(f"[submit] using best_exp={exp_id}")

    _, train_dir, test_root = _resolve_data_roots(None, args.train_dir, args.test_dir)

    run_ensemble_submit_v2(
        exp_id=exp_id,
        cache_dir=cache_dir,
        checkpoints_dir=Path(args.checkpoints_dir).resolve(),
        output_csv=Path(args.output).resolve(),
        test_root=test_root,
        train_dir=train_dir,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        force=bool(args.force_logits),
        tta_tag=_tta_tag(cache_dir),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_cache = sub.add_parser("cache", help="cache holdout/test logits per seed")
    p_cache.add_argument("--checkpoints-dir", type=Path, default=REPO_ROOT / "checkpoints/track_a/videomaev2+ft")
    p_cache.add_argument("--cache-dir", type=Path, default=V3_CACHE_DEFAULT)
    p_cache.add_argument(
        "--tta-mode",
        choices=("champion", "official_2x3"),
        default="champion",
        help="TTA branch: champion=scales3+flip (sweep winner)",
    )
    p_cache.add_argument(
        "--tta-tag",
        default="champion",
        help="Filename suffix for TTA caches (default: champion)",
    )
    p_cache.add_argument("--val-dir", type=Path, default=None)
    p_cache.add_argument("--train-dir", type=Path, default=None)
    p_cache.add_argument("--test-dir", type=Path, default=None)
    p_cache.add_argument("--holdout-ratio", type=float, default=0.1)
    p_cache.add_argument("--holdout-split-seed", type=int, default=42)
    p_cache.add_argument("--batch-size", type=int, default=8)
    p_cache.add_argument("--num-workers", type=int, default=4)
    p_cache.add_argument("--force", action="store_true")
    p_cache.add_argument("--also-test", action="store_true", default=True)

    p_opt = sub.add_parser("optimize", help="combiner grid on holdout")
    p_opt.add_argument("--cache-dir", type=Path, default=V3_CACHE_DEFAULT)

    p_sub = sub.add_parser("submit", help="ensemble CSV from grid row")
    p_sub.add_argument("--cache-dir", type=Path, default=V3_CACHE_DEFAULT)
    p_sub.add_argument("--checkpoints-dir", type=Path, default=REPO_ROOT / "checkpoints/track_a/videomaev2+ft")
    p_sub.add_argument("--exp", type=str, default="best")
    p_sub.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "submissions/track_a_ensemble_v3_champion_best.csv",
    )
    p_sub.add_argument("--test-dir", type=Path, default=None)
    p_sub.add_argument("--train-dir", type=Path, default=None)
    p_sub.add_argument("--batch-size", type=int, default=8)
    p_sub.add_argument("--num-workers", type=int, default=4)
    p_sub.add_argument("--force-logits", action="store_true")

    args = parser.parse_args()
    if args.command == "cache":
        args.val_dir, args.train_dir, args.test_dir = _resolve_data_roots(
            args.val_dir, args.train_dir, args.test_dir
        )
        print(f"[cache] val={args.val_dir} train={args.train_dir} test={args.test_dir}")
        cmd_cache(args)
    elif args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "submit":
        args.val_dir, args.train_dir, args.test_dir = _resolve_data_roots(
            None, args.train_dir, args.test_dir
        )
        cmd_submit(args)


if __name__ == "__main__":
    main()
