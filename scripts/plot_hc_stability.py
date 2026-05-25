"""Plot HC / mHC stability metrics from per-step CSV logs.

Arms: baseline (Pre-Norm), shc_n4 (Static HC), mhc_n4_sk3 (Manifold HC).
- Baseline has seeds 43, 44 only (s42 has no stability file).
- SHC and mHC have seeds 42, 43, 44.
- mHC s44 is incomplete (~53/60 epochs) — handled by per-epoch aggregation.

Outputs are written to report/figures/hc_stability/.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
STAB = ROOT / "logs" / "hc" / "stability"
OUT = ROOT / "report" / "figures" / "hc_stability"
OUT.mkdir(parents=True, exist_ok=True)

ARMS = {
    "baseline": {"seeds": [43, 44], "label": "Pre-Norm", "pattern": "baseline_s{seed}_stability.csv"},
    "shc_n4":   {"seeds": [42, 43, 44], "label": "SHC (n=4)", "pattern": "shc_n4_s{seed}_stability.csv"},
    "mhc_n4_sk3": {"seeds": [42, 43, 44], "label": "mHC (n=4, K=3)", "pattern": "mhc_n4_sk3_s{seed}_stability.csv"},
}
PALETTE = {
    "baseline": "#6c8ebf",
    "shc_n4":   "#82b366",
    "mhc_n4_sk3": "#b85450",
}

sns.set_theme(
    style="whitegrid",
    context="paper",
    rc={
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444",
        "axes.labelcolor": "#222",
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.6,
        "axes.titleweight": "regular",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.frameon": False,
        "figure.dpi": 130,
    },
)


def load_arm(arm: str) -> dict[int, pd.DataFrame]:
    out = {}
    spec = ARMS[arm]
    for s in spec["seeds"]:
        p = STAB / spec["pattern"].format(seed=s)
        df = pd.read_csv(p)
        df = df.sort_values("step").reset_index(drop=True)
        out[s] = df
    return out


def mean_band(ax, x_col: str, y_col: str, frames: dict[int, pd.DataFrame], color: str, label: str, smooth: int = 0):
    """Aggregate frames by integer x_col bin (e.g. epoch), then mean±std across seeds."""
    pivots = []
    for s, df in frames.items():
        g = df.groupby(x_col)[y_col].mean()
        pivots.append(g.rename(s))
    wide = pd.concat(pivots, axis=1).sort_index()
    mean = wide.mean(axis=1)
    std = wide.std(axis=1, ddof=0)
    if smooth and smooth > 1:
        mean = mean.rolling(smooth, min_periods=1, center=True).mean()
        std = std.rolling(smooth, min_periods=1, center=True).mean()
    ax.plot(mean.index, mean.values, color=color, label=label, linewidth=1.6)
    ax.fill_between(mean.index, (mean - std).values, (mean + std).values, color=color, alpha=0.18, linewidth=0)


def per_block_long(arm: str, frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    block_cols = [c for c in frames[next(iter(frames))].columns if c.startswith("grad_norm_encoder.blocks.")]
    rows = []
    for s, df in frames.items():
        sub = df[["epoch"] + block_cols].copy()
        g = sub.groupby("epoch").mean()
        for c in block_cols:
            blk = int(re.search(r"blocks\.(\d+)", c).group(1))
            for epoch, val in g[c].items():
                rows.append({"arm": arm, "seed": s, "epoch": int(epoch), "block": blk, "grad_norm": val})
    return pd.DataFrame(rows)


def router_metric_long(frames: dict[int, pd.DataFrame], substr: str, prefix: str) -> pd.DataFrame:
    """Average the `prefix` columns whose name contains `substr`, grouped by epoch and seed."""
    rows = []
    for s, df in frames.items():
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            continue
        avg = df[cols].mean(axis=1)
        tmp = pd.DataFrame({"epoch": df["epoch"], "value": avg})
        g = tmp.groupby("epoch")["value"].mean()
        for epoch, v in g.items():
            rows.append({"seed": s, "epoch": int(epoch), "value": v})
    return pd.DataFrame(rows)


def main():
    data = {arm: load_arm(arm) for arm in ARMS}

    # ---------- 1. Train loss vs epoch ----------
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for arm, frames in data.items():
        mean_band(ax, "epoch", "loss", frames, PALETTE[arm], ARMS[arm]["label"])
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss")
    ax.set_title("Train loss across seeds")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "01_train_loss_vs_epoch.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- 2. Global grad norm vs epoch (smoothed) ----------
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for arm, frames in data.items():
        mean_band(ax, "epoch", "grad_norm_global", frames, PALETTE[arm], ARMS[arm]["label"], smooth=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("global grad-norm (pre-clip L2)")
    ax.set_title("Global gradient norm across seeds (mean ± std)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "02_grad_norm_global_vs_epoch.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- 3. Loss oscillation (rolling std over 500-step window) vs epoch ----------
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for arm, frames in data.items():
        smoothed_frames = {}
        for s, df in frames.items():
            d = df.copy()
            d["loss_rstd"] = d["loss"].rolling(500, min_periods=50).std()
            smoothed_frames[s] = d
        mean_band(ax, "epoch", "loss_rstd", smoothed_frames, PALETTE[arm], ARMS[arm]["label"])
    ax.set_xlabel("epoch")
    ax.set_ylabel("rolling std of per-step loss (window=500)")
    ax.set_title("Loss-curve smoothness")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "03_loss_rolling_std_vs_epoch.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- 4. Grad-norm rolling std (oscillation) ----------
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for arm, frames in data.items():
        smoothed_frames = {}
        for s, df in frames.items():
            d = df.copy()
            d["gn_rstd"] = d["grad_norm_global"].rolling(500, min_periods=50).std()
            smoothed_frames[s] = d
        mean_band(ax, "epoch", "gn_rstd", smoothed_frames, PALETTE[arm], ARMS[arm]["label"])
    ax.set_xlabel("epoch")
    ax.set_ylabel("rolling std of global grad-norm (window=500)")
    ax.set_title("Gradient-norm oscillation")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "04_grad_norm_rolling_std_vs_epoch.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- 5. Per-block grad norm heatmap (one panel per arm) ----------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), sharey=True)
    arm_list = list(ARMS.keys())
    vmin = vmax = None
    heatmaps = {}
    for arm in arm_list:
        long_df = per_block_long(arm, data[arm])
        pivot = long_df.groupby(["block", "epoch"])["grad_norm"].mean().unstack("epoch").sort_index()
        heatmaps[arm] = pivot
    all_vals = np.concatenate([h.values.flatten() for h in heatmaps.values()])
    all_vals = all_vals[~np.isnan(all_vals)]
    vmin, vmax = np.percentile(all_vals, [2, 98])
    for ax, arm in zip(axes, arm_list):
        pivot = heatmaps[arm]
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="rocket_r",
            vmin=vmin,
            vmax=vmax,
            cbar=(arm == arm_list[-1]),
            cbar_kws={"label": "grad-norm"} if arm == arm_list[-1] else None,
            xticklabels=10,
        )
        ax.set_title(ARMS[arm]["label"])
        ax.set_xlabel("epoch")
        ax.set_ylabel("transformer block")
        ax.invert_yaxis()
    fig.suptitle("Per-block gradient norm (mean across seeds)", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "05_grad_norm_per_block_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- 6. Mixing matrix drift: M off-diagonal mass ----------
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for arm in ["shc_n4", "mhc_n4_sk3"]:
        long_df = router_metric_long(data[arm], "M_off_diag_mass", "M_off_diag_mass_")
        if long_df.empty:
            continue
        pivot = long_df.pivot_table(index="epoch", columns="seed", values="value")
        mean = pivot.mean(axis=1)
        std = pivot.std(axis=1, ddof=0)
        ax.plot(mean.index, mean.values, color=PALETTE[arm], label=ARMS[arm]["label"], linewidth=1.6)
        ax.fill_between(mean.index, (mean - std).values, (mean + std).values, color=PALETTE[arm], alpha=0.18, linewidth=0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("avg off-diagonal mass of M (mean over routers)")
    ax.set_title("HC mixing-matrix drift from identity")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "06_M_off_diag_mass_vs_epoch.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- 7. M max abs vs epoch (drift of largest entry) ----------
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for arm in ["shc_n4", "mhc_n4_sk3"]:
        long_df = router_metric_long(data[arm], "M_max_abs", "M_max_abs_")
        if long_df.empty:
            continue
        pivot = long_df.pivot_table(index="epoch", columns="seed", values="value")
        mean = pivot.mean(axis=1)
        std = pivot.std(axis=1, ddof=0)
        ax.plot(mean.index, mean.values, color=PALETTE[arm], label=ARMS[arm]["label"], linewidth=1.6)
        ax.fill_between(mean.index, (mean - std).values, (mean + std).values, color=PALETTE[arm], alpha=0.18, linewidth=0)
    ax.axhline(1.0, color="#888", linewidth=0.8, linestyle="--", label="identity (=1)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("max |M_ij| (mean over routers)")
    ax.set_title("HC mixing-matrix max entry")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "07_M_max_abs_vs_epoch.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- 8. mHC Sinkhorn doubly-stochastic deviation (sanity check) ----------
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    long_df = router_metric_long(data["mhc_n4_sk3"], "sk_dev", "sk_dev_")
    if not long_df.empty:
        pivot = long_df.pivot_table(index="epoch", columns="seed", values="value")
        mean = pivot.mean(axis=1)
        std = pivot.std(axis=1, ddof=0)
        ax.plot(mean.index, mean.values, color=PALETTE["mhc_n4_sk3"], linewidth=1.6, label="mHC")
        ax.fill_between(mean.index, (mean - std).values, (mean + std).values, color=PALETTE["mhc_n4_sk3"], alpha=0.18, linewidth=0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("SK(M̃) deviation from doubly stochastic")
    ax.set_title("mHC: Sinkhorn projection sanity (K=3)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "08_mhc_sk_deviation.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- 9. Combined summary panel ----------
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.4))

    ax = axes[0, 0]
    for arm, frames in data.items():
        mean_band(ax, "epoch", "loss", frames, PALETTE[arm], ARMS[arm]["label"])
    ax.set_xlabel("epoch"); ax.set_ylabel("train loss"); ax.set_title("Train loss")
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    for arm, frames in data.items():
        mean_band(ax, "epoch", "grad_norm_global", frames, PALETTE[arm], ARMS[arm]["label"], smooth=3)
    ax.set_xlabel("epoch"); ax.set_ylabel("global grad-norm"); ax.set_title("Global gradient norm")
    ax.legend(loc="upper right")

    ax = axes[1, 0]
    for arm, frames in data.items():
        smoothed_frames = {}
        for s, df in frames.items():
            d = df.copy()
            d["gn_rstd"] = d["grad_norm_global"].rolling(500, min_periods=50).std()
            smoothed_frames[s] = d
        mean_band(ax, "epoch", "gn_rstd", smoothed_frames, PALETTE[arm], ARMS[arm]["label"])
    ax.set_xlabel("epoch"); ax.set_ylabel("rolling std (grad-norm)"); ax.set_title("Grad-norm oscillation")
    ax.legend(loc="upper right")

    ax = axes[1, 1]
    for arm in ["shc_n4", "mhc_n4_sk3"]:
        long_df = router_metric_long(data[arm], "M_off_diag_mass", "M_off_diag_mass_")
        if long_df.empty:
            continue
        pivot = long_df.pivot_table(index="epoch", columns="seed", values="value")
        mean = pivot.mean(axis=1); std = pivot.std(axis=1, ddof=0)
        ax.plot(mean.index, mean.values, color=PALETTE[arm], label=ARMS[arm]["label"], linewidth=1.6)
        ax.fill_between(mean.index, (mean - std).values, (mean + std).values, color=PALETTE[arm], alpha=0.18, linewidth=0)
    ax.set_xlabel("epoch"); ax.set_ylabel("M off-diag mass"); ax.set_title("Mixing-matrix drift")
    ax.legend(loc="best")

    fig.suptitle("HC / mHC stability ablation — VideoMAE ViT-B (3 seeds, mean ± std)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "00_summary_panel.png", bbox_inches="tight")
    plt.close(fig)

    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
