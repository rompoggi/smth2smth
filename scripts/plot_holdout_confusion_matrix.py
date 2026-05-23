#!/usr/bin/env python3
"""Plot holdout confusion matrix for an ensemble grid row (cached logits)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from smth2smth.ensemble.combiners import logits_to_probs, run_combiner, sanitize_logits
from smth2smth.ensemble.optimize import combine_logits
from smth2smth.shared.data import parse_class_index

SEEDS = (42, 43, 44)
CLUSTER_VAL = Path("/Data/thomas.turkieh/smth2smth/data/val")
N_CLASSES = 33


def _tta_tag(cache_dir: Path) -> str:
    meta = cache_dir / "cache_meta.json"
    if meta.is_file():
        return json.loads(meta.read_text(encoding="utf-8")).get("tta_tag", "champion")
    return "champion"


def _load_row(cache_dir: Path, exp_id: str) -> dict:
    for name in ("ensemble_results.json", "ensemble_results_v2.json"):
        path = cache_dir / name
        if path.is_file():
            for row in json.loads(path.read_text(encoding="utf-8"))["holdout_metrics"]:
                if row["exp"] == exp_id:
                    return row
    raise KeyError(f"exp {exp_id!r} not found under {cache_dir}")


def _load_branch(
    cache_dir: Path, members: list[int], branch: str, *, tta_tag: str
) -> tuple[list[np.ndarray], list[np.ndarray] | None]:
    logits = [
        sanitize_logits(
            np.load(cache_dir / f"logits_s{s}_{'none' if branch == 'basic' else tta_tag}.npy")
        )
        for s in members
    ]
    probs = None
    if branch == "tta":
        prob_files = [cache_dir / f"probs_s{s}_{tta_tag}_probs.npy" for s in members]
        if all(p.is_file() for p in prob_files):
            probs = [np.load(p) for p in prob_files]
        else:
            probs = [logits_to_probs(a) for a in logits]
    return logits, probs


def _combined_logits(row: dict, eval_logits: list[np.ndarray], eval_probs: list[np.ndarray] | None) -> np.ndarray:
    comb = row["combiner"]
    weights = row.get("weights")
    if comb in ("mean", "ws"):
        w = np.full(len(eval_logits), 1.0 / len(eval_logits))
        if comb == "ws" and isinstance(weights, dict):
            w = np.array([float(weights[str(s)]) for s in SEEDS[: len(eval_logits)]])
        return combine_logits(eval_logits, w)
    if comb == "softmax":
        probs = eval_probs or [logits_to_probs(a) for a in eval_logits]
        return np.log(np.clip(np.mean(probs, axis=0), 1e-12, 1.0))
    if comb == "vote":
        stacked = np.stack([a.argmax(axis=1) for a in eval_logits], axis=1)
        n_cls = eval_logits[0].shape[1]
        out = np.zeros((stacked.shape[0], n_cls))
        for i in range(stacked.shape[0]):
            out[i, int(np.argmax(np.bincount(stacked[i], minlength=n_cls)))] = 1.0
        return out
    if comb == "cws":
        w = np.array(weights, dtype=np.float64)
        return np.einsum("mnc,mc->nc", np.stack(eval_logits, axis=0), w)
    if comb == "lsg":
        coef = np.array(weights, dtype=np.float64)
        return np.concatenate(eval_logits, axis=1) @ coef.T
    if comb == "single":
        return eval_logits[0]
    raise ValueError(f"Unsupported combiner: {comb}")


def _predictions_for_exp(cache_dir: Path, exp_id: str, labels: np.ndarray) -> tuple[np.ndarray, dict]:
    row = _load_row(cache_dir, exp_id)
    tta_tag = _tta_tag(cache_dir)
    comb = row["combiner"]
    fit_b, eval_b = row["fit"], row["eval"]

    if comb == "single":
        seed = int(re.search(r"s(\d+)", exp_id).group(1))
        eval_logits, _ = _load_branch(cache_dir, [seed], eval_b, tta_tag=tta_tag)
        return eval_logits[0].argmax(axis=1), row

    members = list(SEEDS)
    fit_logits, fit_probs = _load_branch(cache_dir, members, fit_b, tta_tag=tta_tag)
    eval_logits, eval_probs = _load_branch(cache_dir, members, eval_b, tta_tag=tta_tag)

    if comb in ("cws", "lsg"):
        r = run_combiner(comb, fit_logits, eval_logits, eval_probs, labels, name=exp_id)
        row = {**row, "weights": r.weights.tolist() if hasattr(r.weights, "tolist") else r.weights}

    combined = _combined_logits(row, eval_logits, eval_probs)
    return combined.argmax(axis=1), row


def _class_names_from_val(val_dir: Path, n_classes: int) -> list[str]:
    names = [f"class_{i}" for i in range(n_classes)]
    if not val_dir.is_dir():
        return names
    for d in sorted(val_dir.iterdir()):
        if not d.is_dir():
            continue
        idx = parse_class_index(d.name)
        if idx is not None and 0 <= idx < n_classes:
            short = re.sub(r"^\d+_", "", d.name)
            if len(short) > 28:
                short = short[:25] + "..."
            names[idx] = short
    return names


def _plot_matrix(
    cm: np.ndarray,
    class_names: list[str],
    out_path: Path,
    *,
    title: str,
    normalize: bool,
) -> None:
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm, row_sums, where=row_sums > 0, out=np.zeros_like(cm, dtype=float))
        fmt = ".2f"
        vmax = 1.0
    else:
        cm_plot = cm.astype(float)
        fmt = "d"
        vmax = None

    n = len(class_names)
    fig_w = max(10, min(0.35 * n, 24))
    fig_h = max(8, min(0.3 * n, 20))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues", vmin=0, vmax=vmax)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True class",
        xlabel="Predicted class",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right", rotation_mode="anchor", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    thresh = (cm_plot.max() / 2.0) if cm_plot.size else 0.0
    for i in range(n):
        for j in range(n):
            if cm[i, j] == 0:
                continue
            val = cm_plot[i, j]
            text = f"{int(val)}" if fmt == "d" else format(val, fmt)
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=5,
            )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "outputs/ensemble/videomaev2_3seed_v3_champion",
    )
    parser.add_argument("--exp", type=str, default="T-ws")
    parser.add_argument("--val-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/ensemble/analysis",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    val_dir = args.val_dir or CLUSTER_VAL
    if not val_dir.is_dir():
        val_dir = REPO_ROOT / "data" / "val"

    labels = np.load(cache_dir / "labels_holdout.npy")
    preds, row = _predictions_for_exp(cache_dir, args.exp, labels)
    class_names = _class_names_from_val(val_dir, N_CLASSES)

    cm = confusion_matrix(labels, preds, labels=np.arange(N_CLASSES))
    out_dir = args.output_dir.resolve() / args.exp.replace("-", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "confusion_matrix.npy", cm)
    row_totals = cm.sum(axis=1)
    per_class_acc = np.divide(np.diag(cm), row_totals, where=row_totals > 0, out=np.zeros(N_CLASSES))
    summary = {
        "exp": args.exp,
        "cache_dir": str(cache_dir),
        "n_holdout": int(len(labels)),
        "top1": row["top1"],
        "top5": row["top5"],
        "per_class_accuracy": {
            class_names[i]: float(per_class_acc[i])
            for i in range(N_CLASSES)
            if row_totals[i] > 0
        },
        "worst_classes": sorted(
            [
                {"class": class_names[i], "acc": float(per_class_acc[i]), "support": int(row_totals[i])}
                for i in range(N_CLASSES)
                if row_totals[i] > 0
            ],
            key=lambda x: x["acc"],
        )[:8],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = classification_report(
        labels,
        preds,
        labels=np.arange(N_CLASSES),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    (out_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    _plot_matrix(
        cm,
        class_names,
        out_dir / "confusion_matrix_counts.png",
        title=f"Holdout confusion matrix — {args.exp} (counts)",
        normalize=False,
    )
    _plot_matrix(
        cm,
        class_names,
        out_dir / "confusion_matrix_normalized.png",
        title=f"Holdout confusion matrix — {args.exp} (row-normalized recall)",
        normalize=True,
    )

    print(f"[cm] exp={args.exp} holdout N={len(labels)} top1={row['top1']:.2f}%")
    print(f"[cm] outputs: {out_dir}/")
    print("  confusion_matrix_counts.png")
    print("  confusion_matrix_normalized.png")
    print("  summary.json  classification_report.json")


if __name__ == "__main__":
    main()
