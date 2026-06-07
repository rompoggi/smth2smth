#!/usr/bin/env python3
"""Offline val-only ensembling for MAE500 stab diverse heads (Steps 0–4).

Step 0 ``dump``     — cache per-run val logits (GPU; last-epoch ckpts).
Step 1 ``combiners`` — nested val-fit vs val-eval combiner comparison + bootstrap CIs.
Step 2 ``diversity`` — architecture vs seed diversity 2×2 with fixed combiner.
Step 3 ``disagreement`` — pairwise error-correlation heatmap + gain vs disagreement.
Step 4 ``gain-epoch`` — ensemble gain vs SSL epoch (Q8 + meanpool ladders).

All weight fitting uses val-fit only; scoring on val-eval. No LB leakage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
for p in (REPO / "src", SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from smth2smth.ensemble.combiners import sanitize_logits
from smth2smth.ensemble.holdout import sample_key
from smth2smth.ensemble.inference import TtaMode, collect_logits_for_videos
from smth2smth.ensemble.optimize import metrics_from_logits
from smth2smth.shared.data import collect_video_samples
from smth2smth.shared.utils.splits import split_train_val_stratified

DEFAULT_CACHE = REPO / "outputs/ensemble/mae500_stab_val"
DEFAULT_OUT = REPO / "outputs/ensemble/mae500_stab_val/plots"
CLUSTER_DATA = Path("/Data/thomas.turkieh/smth2smth/data")
COMBINERS = ("mean", "softmax", "vote", "ws", "cws", "lsg")
SPLIT_SEED = 42
EVAL_RATIO = 0.2
BOOTSTRAP = 500
N_FOLDS = 5


def _default_val_dir() -> Path:
    for base in (REPO / "data", CLUSTER_DATA):
        val = base / "val"
        if val.is_dir() and len(list(val.iterdir())) > 100:
            return val
    return REPO / "data" / "val"


def _nested_val_split(val_dir: Path) -> tuple[list, list, np.ndarray, np.ndarray]:
    all_val = collect_video_samples(val_dir)
    fit, eval_ = split_train_val_stratified(all_val, val_ratio=EVAL_RATIO, seed=SPLIT_SEED)
    labels = np.array([lab for _, lab in all_val], dtype=np.int64)
    keys = [sample_key(vd) for vd, _ in all_val]
    fit_idx = np.array([keys.index(sample_key(vd)) for vd, _ in fit], dtype=np.int64)
    eval_idx = np.array([keys.index(sample_key(vd)) for vd, _ in eval_], dtype=np.int64)
    return fit, eval_, fit_idx, eval_idx


def _load_unified_mae500_stab() -> pd.DataFrame:
    from plot_unified_fleet_scaling import (  # lazy: only the non-manifest dump path needs this
        COLLECTED_LOGS,
        UNIFIED,
        WANDB_SUMMARY,
        annotate_is_stab,
        build_recipe_index,
        load_rows,
    )

    df = load_rows(UNIFIED, WANDB_SUMMARY)
    df = annotate_is_stab(df, build_recipe_index(COLLECTED_LOGS))
    df = df[(df["is_stab"]) & (df["ssl_ep"] == 500)].copy()
    return df


def _ckpt_for_row(row: pd.Series, *, last_epoch: bool = True) -> Path | None:
    col = "collected_ckpt_last" if last_epoch else "collected_ckpt_best"
    if col not in row.index or pd.isna(row.get(col)):
        col2 = "collected_ckpt_best" if last_epoch else "collected_ckpt_last"
        if col2 not in row.index or pd.isna(row.get(col2)):
            return None
        col = col2
    p = REPO / str(row[col])
    return p if p.is_file() else None


def cmd_dump(args: argparse.Namespace) -> None:
    val_dir = Path(args.val_dir).resolve()
    train_dir = Path(args.train_dir or _default_val_dir().parent / "train").resolve()
    cache = Path(args.cache_dir).resolve()
    logits_dir = cache / "logits"
    logits_dir.mkdir(parents=True, exist_ok=True)

    all_val = collect_video_samples(val_dir)
    fit, eval_, fit_idx, eval_idx = _nested_val_split(val_dir)
    manifest = {
        "split_seed": SPLIT_SEED,
        "eval_ratio": EVAL_RATIO,
        "n_val": len(all_val),
        "n_fit": len(fit),
        "n_eval": len(eval_),
        "fit_keys": [sample_key(vd) for vd, _ in fit],
        "eval_keys": [sample_key(vd) for vd, _ in eval_],
    }
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    np.save(cache / "labels_val.npy", np.array([lab for _, lab in all_val], dtype=np.int64))
    np.save(cache / "fit_idx.npy", fit_idx)
    np.save(cache / "eval_idx.npy", eval_idx)

    def _dump_one(name: str, ckpt: Path | None) -> None:
        out = logits_dir / f"{name}.npy"
        if out.is_file() and not args.force:
            print(f"[dump] skip {name} (exists)")
            return
        if ckpt is None or not Path(ckpt).is_file():
            print(f"[dump] MISSING ckpt for {name}: {ckpt}")
            return
        print(f"[dump] {name} <- {Path(ckpt).name} (basic, no TTA) ...", flush=True)
        logits = collect_logits_for_videos(
            Path(ckpt),
            all_val,
            data_root=val_dir,
            train_dir=train_dir,
            tta_mode=TtaMode.NONE,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
        )
        np.save(out, logits.numpy())

    if args.members_manifest:
        entries = json.loads(Path(args.members_manifest).read_text(encoding="utf-8"))
        print(f"[dump] {len(entries)} members from manifest; val N={len(all_val)} fit={len(fit)} eval={len(eval_)}")
        for e in entries:
            ckpt = Path(e["ckpt"])
            if not ckpt.is_absolute():
                ckpt = REPO / ckpt
            _dump_one(str(e["name"]), ckpt)
    else:
        df = _load_unified_mae500_stab()
        if args.runs:
            df = df[df["run_name"].isin(args.runs)]
        print(f"[dump] {len(df)} MAE500 stab runs; val N={len(all_val)} fit={len(fit)} eval={len(eval_)}")
        for _, row in df.iterrows():
            _dump_one(str(row["run_name"]), _ckpt_for_row(row, last_epoch=not args.best_ckpt))
    print(f"[dump] done -> {cache}")


def _load_logits(cache: Path, runs: list[str]) -> dict[str, np.ndarray]:
    logits_dir = cache / "logits"
    out: dict[str, np.ndarray] = {}
    for run in runs:
        p = logits_dir / f"{run}.npy"
        if not p.is_file():
            raise FileNotFoundError(f"missing logits {p} — run: python {Path(__file__).name} dump")
        out[run] = sanitize_logits(np.load(p))
    return out


def _combined_eval_scores(
    comb: str, fit_logits: list[np.ndarray], y_fit: np.ndarray, eval_logits: list[np.ndarray]
) -> np.ndarray:
    """Fit combiner ``comb`` once on (fit_logits, y_fit); return combined (N, C) eval scores."""
    from smth2smth.ensemble.combiners import (
        _fit_lsg,
        _predict_lsg,
        combine_logits,
        logits_to_probs,
        optimize_cws_weights,
        optimize_mix_weights,
        sanitize_logits,
    )

    F = [sanitize_logits(a) for a in fit_logits]
    E = [sanitize_logits(a) for a in eval_logits]
    if comb == "mean":
        return combine_logits(E, np.full(len(E), 1.0 / len(E)))
    if comb == "softmax":
        return np.mean([logits_to_probs(a) for a in E], axis=0)
    if comb == "vote":
        votes = np.stack([a.argmax(axis=1) for a in E], axis=0)  # (M, N)
        n_clips, n_classes = votes.shape[1], E[0].shape[1]
        out = np.zeros((n_clips, n_classes), dtype=np.float64)
        for j in range(n_clips):
            vals, cnts = np.unique(votes[:, j], return_counts=True)
            out[j, vals[cnts.argmax()]] = 1.0
        return out
    if comb == "ws":
        w = optimize_mix_weights(F, y_fit, name="ws").weights
        return combine_logits(E, w)
    if comb == "cws":
        w = optimize_cws_weights(F, y_fit, name="cws").weights
        if w.ndim == 2:
            return np.einsum("mnc,mc->nc", np.stack(E, axis=0), w)
        return combine_logits(E, w)
    if comb == "lsg":
        n_classes = int(F[0].shape[1])
        x_fit = np.concatenate(F, axis=1).astype(np.float64)
        x_eval = np.concatenate(E, axis=1).astype(np.float64)
        coef = _fit_lsg(x_fit, np.asarray(y_fit, dtype=np.int64), n_classes, c_reg=1.0)
        return _predict_lsg(x_eval, coef)
    raise ValueError(f"Unknown combiner: {comb}")


def _top1_ci(scores: np.ndarray, y_eval: np.ndarray, n_boot: int = BOOTSTRAP) -> tuple[float, float]:
    """95% bootstrap CI of top-1 (%) for fixed predictions, resampling val-eval clips."""
    correct = (scores.argmax(axis=1) == y_eval).astype(np.float64)
    rng = np.random.default_rng(SPLIT_SEED)
    boots = [
        100.0 * float(correct[rng.choice(len(correct), size=len(correct), replace=True)].mean())
        for _ in range(n_boot)
    ]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _oof_scores(
    comb: str, member_logits: list[np.ndarray], y: np.ndarray, *, n_folds: int = N_FOLDS
) -> np.ndarray:
    """Stratified K-fold out-of-fold combiner scores over all N clips.

    Fit the combiner on n_folds-1 folds, predict the held fold; pool held-fold
    predictions into one (N, C) array. Learned combiners never score clips they were fit on.
    """
    from sklearn.model_selection import StratifiedKFold

    n, c = member_logits[0].shape
    oof = np.zeros((n, c), dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SPLIT_SEED)
    for fit_idx, eval_idx in skf.split(np.zeros(n), y):
        fit_logits = [a[fit_idx] for a in member_logits]
        eval_logits = [a[eval_idx] for a in member_logits]
        oof[eval_idx] = _combined_eval_scores(comb, fit_logits, y[fit_idx], eval_logits)
    return oof


def cmd_combiners(args: argparse.Namespace) -> None:
    cache = Path(args.cache_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    y = np.load(cache / "labels_val.npy")
    members = args.members or [
        "meanpool-mae500-s42",
        "perceiverQ8-mae500-s42",
        "DivSpaceTimeK9-mae500-s42",
        "perceiverQ16-mae500-s43",
    ]
    store = _load_logits(cache, members)
    member_logits = [store[m] for m in members]

    rows = []
    for comb in COMBINERS:
        oof = _oof_scores(comb, member_logits, y)
        top1, top5, _ = metrics_from_logits(oof, y)
        lo, hi = _top1_ci(oof, y)
        rows.append(
            {
                "combiner": comb,
                "top1_eval": top1,
                "top5_eval": top5,
                "ci_lo": lo,
                "ci_hi": hi,
                "members": members,
            }
        )
        print(f"[combiners] {comb}: top1={top1:.2f}% CI=[{lo:.2f},{hi:.2f}]  (5-fold OOF, N={len(y)})")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "combiner_comparison.csv", index=False)
    with (out_dir / "combiner_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"[combiners] wrote {out_dir / 'combiner_comparison.csv'}")


def cmd_diversity(args: argparse.Namespace) -> None:
    cache = Path(args.cache_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    comb = args.combiner

    y = np.load(cache / "labels_val.npy")

    sets = {
        "seed-only (Q8×3)": [
            "perceiverQ8-mae500-s42",
            "perceiverQ8-mae500-s43",
            "perceiverQ8-mae500-s44",
        ],
        "architecture-only": [
            "meanpool-mae500-s42",
            "perceiverQ8-mae500-s42",
            "DivSpaceTimeK9-mae500-s42",
        ],
        "all-axes": [
            "meanpool-mae500-s42",
            "perceiverQ8-mae500-s44",
            "DivSpaceTimeK9-mae500-s43",
            "perceiverQ16-mae500-s43",
        ],
    }
    all_runs = sorted({r for rs in sets.values() for r in rs})
    store = _load_logits(cache, all_runs)

    rows = []
    for name, members in sets.items():
        available = [m for m in members if (cache / "logits" / f"{m}.npy").is_file()]
        if len(available) < 2:
            print(f"[diversity] skip {name}: only {len(available)} members cached")
            continue
        member_logits = [store[m] for m in available]
        oof = _oof_scores(comb, member_logits, y)
        ens_top1 = metrics_from_logits(oof, y)[0]
        singles = [metrics_from_logits(store[m], y)[0] for m in available]
        lo, hi = _top1_ci(oof, y)
        rows.append(
            {
                "set": name,
                "combiner": comb,
                "n_members": len(available),
                "ensemble_top1": ens_top1,
                "best_single_top1": max(singles),
                "gain_pp": ens_top1 - max(singles),
                "ci_lo": lo,
                "ci_hi": hi,
                "members": available,
            }
        )
        print(f"[diversity] {name}: ens={ens_top1:.2f}% best_single={max(singles):.2f}% gain={ens_top1-max(singles):+.2f}pp")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "diversity_grid.csv", index=False)

    if not df.empty:
        sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(df))
        ax.bar(x, df["ensemble_top1"], yerr=[df["ensemble_top1"] - df["ci_lo"], df["ci_hi"] - df["ensemble_top1"]],
               capsize=4, color="#0d9488", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(df["set"], rotation=15, ha="right")
        ax.set_ylabel("Val top-1 (%, 5-fold OOF)")
        ax.set_title(f"Diversity ensemble grid ({comb}, 5-fold OOF)")
        fig.tight_layout()
        fig.savefig(out_dir / "diversity_grid.png", dpi=160, facecolor="white")
        plt.close(fig)
    print(f"[diversity] wrote {out_dir}")


def cmd_disagreement(args: argparse.Namespace) -> None:
    cache = Path(args.cache_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    y = np.load(cache / "labels_val.npy")

    members = args.members or [
        "meanpool-mae500-s42",
        "perceiverQ8-mae500-s42",
        "DivSpaceTimeK9-mae500-s42",
        "perceiverQ16-mae500-s43",
    ]
    store = _load_logits(cache, [m for m in members if (cache / "logits" / f"{m}.npy").is_file()])
    members = [m for m in members if m in store]
    preds = {m: store[m].argmax(axis=1) for m in members}
    n = len(members)
    corr = np.zeros((n, n))
    disagree = np.zeros((n, n))
    for i, mi in enumerate(members):
        for j, mj in enumerate(members):
            pi, pj = preds[mi], preds[mj]
            corr[i, j] = np.corrcoef((pi == y).astype(float), (pj == y).astype(float))[0, 1]
            disagree[i, j] = float(np.mean(pi != pj))

    sns.set_theme(context="talk", font_scale=0.85)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.heatmap(corr, xticklabels=members, yticklabels=members, annot=True, fmt=".2f",
                cmap="RdYlGn", vmin=0, vmax=1, ax=axes[0])
    axes[0].set_title("Correctness correlation")
    sns.heatmap(disagree, xticklabels=members, yticklabels=members, annot=True, fmt=".2f",
                cmap="Blues", vmin=0, vmax=1, ax=axes[1])
    axes[1].set_title("Prediction disagreement")
    fig.tight_layout()
    fig.savefig(out_dir / "disagreement_heatmap.png", dpi=160, facecolor="white")
    plt.close(fig)

    # Gain vs mean pairwise disagreement for architecture triple.
    arch = [m for m in ("meanpool-mae500-s42", "perceiverQ8-mae500-s42", "DivSpaceTimeK9-mae500-s42") if m in store]
    if len(arch) >= 2:
        comb = args.combiner
        oof = _oof_scores(comb, [store[m] for m in arch], y)
        ens_top1 = metrics_from_logits(oof, y)[0]
        singles = [metrics_from_logits(store[m], y)[0] for m in arch]
        pair_dis = []
        for i in range(len(arch)):
            for j in range(i + 1, len(arch)):
                pair_dis.append(disagree[members.index(arch[i]), members.index(arch[j])])
        summary = {
            "members": arch,
            "mean_pairwise_disagreement": float(np.mean(pair_dis)),
            "ensemble_top1": ens_top1,
            "best_single_top1": max(singles),
            "gain_pp": ens_top1 - max(singles),
        }
        (out_dir / "disagreement_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[disagreement] arch triple gain={summary['gain_pp']:+.2f}pp @ disagree={summary['mean_pairwise_disagreement']:.3f}")
    print(f"[disagreement] wrote {out_dir}")


def cmd_gain_epoch(args: argparse.Namespace) -> None:
    """Ensemble gain vs SSL epoch using Q8 + meanpool ladders (requires dumped logits)."""
    cache = Path(args.cache_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    comb = args.combiner
    epochs = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    y = np.load(cache / "labels_val.npy")

    rows = []
    for ep in epochs:
        mp = f"meanpool-mae{ep:03d}-s42"  # meanpool ckpts zero-pad: mae050..mae500
        q8 = f"perceiverQ8-mae{ep}-s42"  # Q8 run names: mae50..mae500
        members = [r for r in (mp, q8) if (cache / "logits" / f"{r}.npy").is_file()]
        if len(members) < 2:
            continue
        store = _load_logits(cache, members)
        member_logits = [store[m] for m in members]
        oof = _oof_scores(comb, member_logits, y)
        ens_top1 = metrics_from_logits(oof, y)[0]
        singles = [metrics_from_logits(store[m], y)[0] for m in members]
        rows.append(
            {
                "ssl_ep": ep,
                "ensemble_top1": ens_top1,
                "best_single_top1": max(singles),
                "gain_pp": ens_top1 - max(singles),
            }
        )
        print(f"[gain-epoch] ep={ep}: ens={ens_top1:.2f}% gain={ens_top1 - max(singles):+.2f}pp")

    if not rows:
        print("[gain-epoch] no epoch ladders cached — dump SSL-epoch runs first")
        return
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ensemble_gain_vs_ssl_epoch.csv", index=False)

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["ssl_ep"], df["gain_pp"], "o-", color="#dc2626", linewidth=2.2, markersize=8)
    ax.axhline(0, color="#9ca3af", linestyle="--", linewidth=1)
    ax.set_xlabel("SSL pretrain epoch")
    ax.set_ylabel("Ensemble gain over best single (pp)")
    ax.set_title(f"Ensemble gain vs backbone strength ({comb}: meanpool + Q8)")
    fig.tight_layout()
    fig.savefig(out_dir / "ensemble_gain_vs_ssl_epoch.png", dpi=160, facecolor="white")
    plt.close(fig)
    print(f"[gain-epoch] wrote {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_dump = sub.add_parser("dump", help="Step 0: cache val logits per run (GPU)")
    p_dump.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p_dump.add_argument("--val-dir", type=Path, default=None)
    p_dump.add_argument("--train-dir", type=Path, default=None)
    p_dump.add_argument("--runs", nargs="*", default=None)
    p_dump.add_argument(
        "--members-manifest",
        type=Path,
        default=None,
        help="JSON list of {name, ckpt} to dump (self-contained; skips unified-summary lookup)",
    )
    p_dump.add_argument("--best-ckpt", action="store_true", help="Use best ckpt instead of last")
    p_dump.add_argument("--force", action="store_true")
    p_dump.add_argument("--batch-size", type=int, default=8)
    p_dump.add_argument("--num-workers", type=int, default=4)

    for name, fn in (
        ("combiners", cmd_combiners),
        ("diversity", cmd_diversity),
        ("disagreement", cmd_disagreement),
        ("gain-epoch", cmd_gain_epoch),
    ):
        p = sub.add_parser(name)
        p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
        p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
        p.add_argument("--combiner", default="softmax", choices=COMBINERS)
        if name in ("combiners", "disagreement"):
            p.add_argument("--members", nargs="*", default=None)
        p.set_defaults(func=fn)

    p_dump.set_defaults(func=cmd_dump)
    args = ap.parse_args()
    if args.cmd == "dump" and args.val_dir is None:
        args.val_dir = _default_val_dir()
    args.func(args)


if __name__ == "__main__":
    main()
