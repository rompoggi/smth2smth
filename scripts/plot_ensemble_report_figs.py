#!/usr/bin/env python3
"""Extra report figures for the ensembling section.

1. combiner_bars.png   — Step-1 combiner top-1 with 95% bootstrap CIs.
2. learned_coeffs.png  — what the learned combiners fit on the diverse-4 quad:
   WS per-member weights (bar) + CWS member x class weights (heatmap).
   Diagnostic: shows the learned weights are near-uniform / noisy, i.e. why
   plain softmax (the locked combiner) is statistically indistinguishable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import itertools

from matplotlib.patches import Rectangle

from smth2smth.ensemble.combiners import optimize_cws_weights, optimize_mix_weights, sanitize_logits

CACHE = REPO / "outputs/ensemble/mae500_stab_val"
PLOTS = CACHE / "plots"
DIVERSE4 = [
    "meanpool-mae500-s42",
    "perceiverQ8-mae500-s42",
    "DivSpaceTimeK9-mae500-s42",
    "perceiverQ16-mae500-s43",
]
SHORT = {
    "meanpool-mae500-s42": "meanpool",
    "perceiverQ8-mae500-s42": "Q8",
    "DivSpaceTimeK9-mae500-s42": "DivST-K9",
    "perceiverQ16-mae500-s43": "Q16",
}


def combiner_bars() -> None:
    df = pd.read_csv(PLOTS / "combiner_comparison.csv").sort_values("top1_eval")
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(df))
    err = [df["top1_eval"] - df["ci_lo"], df["ci_hi"] - df["top1_eval"]]
    colors = ["#0d9488" if c == "softmax" else "#94a3b8" for c in df["combiner"]]
    ax.bar(x, df["top1_eval"], yerr=err, capsize=5, color=colors, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in df["combiner"]])
    ax.set_ylim(df["ci_lo"].min() - 0.5, df["ci_hi"].max() + 0.5)
    ax.set_ylabel("Val top-1 (%, 5-fold OOF)")
    ax.set_title("Combiner comparison (diverse-4, 5-fold OOF, 95% bootstrap CI)")
    ax.axhspan(df["ci_lo"].max(), df["ci_hi"].min(), color="#fca5a5", alpha=0.15,
               label="all-CI overlap band")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS / "combiner_bars.png", dpi=160, facecolor="white")
    plt.close(fig)
    print(f"wrote {PLOTS / 'combiner_bars.png'}")


def learned_coeffs() -> None:
    y = np.load(CACHE / "labels_val.npy")
    arrays = [sanitize_logits(np.load(CACHE / "logits" / f"{m}.npy")) for m in DIVERSE4]
    ws = optimize_mix_weights(arrays, y, name="ws").weights  # (M,)
    cws = optimize_cws_weights(arrays, y, name="cws").weights  # (M, C) expected
    labels = [SHORT[m] for m in DIVERSE4]

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.8)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 2.4]})

    axes[0].bar(np.arange(len(ws)), ws, color="#0d9488", alpha=0.9)
    axes[0].axhline(1.0 / len(ws), ls="--", c="#dc2626", lw=1.2, label="uniform (softmax)")
    axes[0].set_xticks(np.arange(len(ws)))
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel("WS weight")
    axes[0].set_title("Weighted-sum: per-member weight")
    axes[0].legend(frameon=False, fontsize=9)

    if cws.ndim == 1:
        cws = np.repeat(cws[:, None], int(arrays[0].shape[1]), axis=1)
    sns.heatmap(cws, ax=axes[1], cmap="RdBu_r", center=1.0 / len(DIVERSE4),
                yticklabels=labels, xticklabels=4, cbar_kws={"label": "CWS weight"})
    axes[1].set_xlabel("class index (0-31)")
    axes[1].set_title("Class-weighted: member x class weight")
    fig.suptitle("Learned combiner coefficients (diverse-4 quad, fit on full val)", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS / "learned_coeffs.png", dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {PLOTS / 'learned_coeffs.png'}")
    print("WS weights:", {labels[i]: round(float(w), 3) for i, w in enumerate(ws)})


def disagreement_fig() -> None:
    """Improved disagreement / error-correlation heatmap.

    Members are ordered so the three Q8 seeds form a block (boxed): same
    architecture / different seed pairs decorrelate less than cross-architecture
    pairs. Diagonal masked, short labels, values annotated.
    """
    y = np.load(CACHE / "labels_val.npy")
    members = [
        "meanpool-mae500-s42", "DivSpaceTimeK9-mae500-s42", "perceiverQ16-mae500-s43",
        "perceiverQ8-mae500-s42", "perceiverQ8-mae500-s43", "perceiverQ8-mae500-s44",
    ]
    labels = ["mp", "DivST-K9", "Q16", "Q8·s42", "Q8·s43", "Q8·s44"]
    preds = {m: sanitize_logits(np.load(CACHE / "logits" / f"{m}.npy")).argmax(1) for m in members}
    correct = {m: (preds[m] == y).astype(float) for m in members}
    n = len(members)
    dis = np.zeros((n, n))
    corr = np.zeros((n, n))
    for i, mi in enumerate(members):
        for j, mj in enumerate(members):
            dis[i, j] = float(np.mean(preds[mi] != preds[mj]))
            corr[i, j] = float(np.corrcoef(correct[mi], correct[mj])[0, 1])

    seed_idx = [3, 4, 5]  # Q8 seeds
    arch_idx = [0, 3, 1]  # mp / Q8-s42 / DivST (the #3 arch-triple)
    seed_dis = np.mean([dis[a, b] for a, b in itertools.combinations(seed_idx, 2)])
    arch_dis = np.mean([dis[a, b] for a, b in itertools.combinations(arch_idx, 2)])

    mask = np.eye(n, dtype=bool)
    sns.set_theme(style="white", context="talk", font_scale=0.78)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
    sns.heatmap(dis, mask=mask, xticklabels=labels, yticklabels=labels, annot=True, fmt=".3f",
                cmap="Blues", vmin=0.18, vmax=0.34, ax=axes[0], square=True,
                cbar_kws={"label": "prediction disagreement", "shrink": 0.8}, linewidths=0.5, linecolor="white")
    axes[0].set_title("Prediction disagreement (higher = more diverse)")
    sns.heatmap(corr, mask=mask, xticklabels=labels, yticklabels=labels, annot=True, fmt=".2f",
                cmap="RdYlGn_r", vmin=0.65, vmax=0.80, ax=axes[1], square=True,
                cbar_kws={"label": "error correlation", "shrink": 0.8}, linewidths=0.5, linecolor="white")
    axes[1].set_title("Error correlation (lower = more diverse)")
    for ax in axes:
        ax.add_patch(Rectangle((3, 3), 3, 3, fill=False, edgecolor="#dc2626", lw=2.5))
        ax.tick_params(rotation=0)
    fig.suptitle(
        f"Diverse-head error structure  —  Q8 seed-triple (red box) disagrees less "
        f"({seed_dis:.3f}) than the cross-architecture triple ({arch_dis:.3f})",
        y=1.0, fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(PLOTS / "disagreement_heatmap.png", dpi=160, facecolor="white")
    plt.close(fig)
    print(f"wrote {PLOTS / 'disagreement_heatmap.png'}  (seed-dis={seed_dis:.3f} arch-dis={arch_dis:.3f})")


if __name__ == "__main__":
    combiner_bars()
    learned_coeffs()
    disagreement_fig()
