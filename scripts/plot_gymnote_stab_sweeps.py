#!/usr/bin/env python3
"""Gymnote stab sweeps: Q@MAE500, K@MAE500 (DivSpaceTime), SSL epoch (Q8 stab vs meanpool).

Uses W&B scaling summary plus gymnote fleet log metrics when W&B is stale (K runs).
Only includes runs with finished FT (epoch >= 50). Q/K curves use stab runs only.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO = Path(__file__).resolve().parents[1]
WANDB_SUMMARY = REPO / "outputs/ft_mae_scaling_thesis/wandb_ft_mae_scaling_run_summary.csv"
FLEET_SUMMARY = REPO / "outputs/gymnote_fleet/fleet_run_summary.csv"
DEFAULT_OUT = REPO / "outputs/gymnote_fleet/plots"

VAL_COL = "last_val_top1"
VAL_YLABEL = "Last-epoch validation top-1"
NOSTAB_SUFFIX = "-NoStab"
Q_RE = re.compile(r"^perceiverQ(\d+)-mae(\d+)-s(\d+)$")
MEANPOOL_RE = re.compile(r"^meanpool-mae(\d+)-s(\d+)$")
DIV_K_RE = re.compile(r"^DivSpaceTimeK(\d+)-mae(\d+)-s(\d+)$")
Q8_STAB_SSL = [100, 200, 300, 400, 500]
K_VALUES = [1, 3, 6, 9]
Q_VALUES_STAB = [2, 4, 8, 16, 32, 64]


def strip_nostab(name: str) -> str:
    return name[: -len(NOSTAB_SUFFIX)] if name.endswith(NOSTAB_SUFFIX) else name


def run_is_stab(run_name: str) -> bool:
    """Heuristic from scaling fleet naming (arch2 perceiver stab @ MAE500)."""
    run_name = strip_nostab(run_name)
    m = Q_RE.match(run_name)
    if m:
        q, ep, seed = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if ep != 500:
            return False
        if q == 16:
            return seed == 42
        return q in (2, 4, 8, 32, 64)
    return False


def is_stab_run(run_name: str, *, recipe: str | None = None) -> bool:
    if run_name.endswith(NOSTAB_SUFFIX):
        return False
    if recipe == "stab":
        return True
    if DIV_K_RE.match(strip_nostab(run_name)):
        return True
    return run_is_stab(run_name)


def aggregate_sem(df: pd.DataFrame, value_col: str, group_cols: list[str]) -> pd.DataFrame:
    out = (
        df.groupby(group_cols)[value_col]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    out["sem"] = out["std"] / np.sqrt(out["n"].clip(lower=1))
    return out.fillna({"sem": 0.0})


def _load_wandb_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"ssl_pretrain_epoch": "ssl_ep"} if "ssl_pretrain_epoch" in df.columns else {})
    if "ssl_ep" not in df.columns and "ssl_pretrain_epoch" in df.columns:
        df["ssl_ep"] = df["ssl_pretrain_epoch"]
    df["run_name"] = df["run_name"].map(strip_nostab)
    df["finished_50"] = df["finished_50"].astype(bool)
    df["is_stab"] = df["run_name"].map(lambda n: is_stab_run(n))
    return df


def _fleet_rows(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    fleet = pd.read_csv(path)
    rows: list[dict] = []
    for _, r in fleet.iterrows():
        name = strip_nostab(str(r["run_name"]))
        if r.get("status") != "DONE" and not bool(r.get("log_done")):
            continue
        last_ep = int(r.get("last_epoch") or r.get("ckpt_epoch") or 0)
        if last_ep < 50:
            continue
        last_val = r.get("last_val_top1")
        if pd.isna(last_val):
            last_val = r.get("ckpt_last_val_top1")
        if pd.isna(last_val):
            continue
        m = DIV_K_RE.match(name)
        if m:
            k, ssl_ep, seed = int(m.group(1)), int(m.group(2)), int(m.group(3))
            rows.append(
                {
                    "run_name": name,
                    "head": f"DivSpaceTimeK{k}",
                    "ssl_ep": ssl_ep,
                    "seed": seed,
                    "q": None,
                    "k": k,
                    "last_val_top1": float(last_val),
                    "finished_50": True,
                    "is_stab": True,
                    "source": "fleet",
                }
            )
            continue
        m = Q_RE.match(name)
        if m and int(m.group(1)) == 8:
            q, ssl_ep, seed = int(m.group(1)), int(m.group(2)), int(m.group(3))
            rows.append(
                {
                    "run_name": name,
                    "head": f"perceiverQ{q}",
                    "ssl_ep": ssl_ep,
                    "seed": seed,
                    "q": q,
                    "k": None,
                    "last_val_top1": float(last_val),
                    "finished_50": True,
                    "is_stab": True,
                    "source": "fleet",
                }
            )
    return pd.DataFrame(rows)


def load_merged_summary(
    wandb_csv: Path,
    fleet_csv: Path,
    *,
    refresh_wandb: bool = False,
) -> pd.DataFrame:
    if refresh_wandb:
        from export_and_plot_ft_mae_scaling_wandb import epoch_curve_df, fetch_all_runs

        raw = fetch_all_runs()
        summary = epoch_curve_df(raw)
        wandb_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(wandb_csv, index=False)

    base = _load_wandb_summary(wandb_csv)
    base["source"] = "wandb"
    if "k" not in base.columns:
        base["k"] = np.nan
    fleet = _fleet_rows(fleet_csv)
    if fleet.empty:
        return base

    # Fleet overrides same run_name (log truth for gymnote K / Q8 SSL).
    names_fleet = set(fleet["run_name"])
    base = base[~base["run_name"].isin(names_fleet)]
    cols = [c for c in base.columns if c in fleet.columns or c == "source"]
    merged = pd.concat([base[cols], fleet], ignore_index=True)
    return merged


def _finished(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["finished_50"]].copy()


def plot_q_sweep_mae500(summary: pd.DataFrame, out_dir: Path) -> None:
    sub = _finished(
        summary[
            (summary["ssl_ep"] == 500)
            & summary["q"].notna()
            & summary["is_stab"]
        ]
    )
    sub = sub[sub["q"].astype(int).isin(Q_VALUES_STAB)]
    mp = _finished(summary[(summary["head"] == "meanpool") & (summary["ssl_ep"] == 500)])
    if sub.empty:
        print("SKIP plot_q_sweep: no stab perceiver @ MAE500")
        return

    sub["q_int"] = sub["q"].astype(int)
    agg_q = aggregate_sem(sub, VAL_COL, ["q_int"])
    agg_mp = aggregate_sem(mp, VAL_COL, ["head"]) if not mp.empty else pd.DataFrame()

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    color = "#0d9488"
    ax.errorbar(
        agg_q["q_int"],
        agg_q["mean"],
        yerr=agg_q["sem"],
        fmt="-",
        color=color,
        capsize=4,
        linewidth=2.2,
        label="Perceiver stab (by Q)",
    )
    for _, row in sub.iterrows():
        ax.scatter(row["q_int"], row[VAL_COL], marker="D", s=55, c=color, edgecolors="white", linewidths=0.5, zorder=5)
    for _, row in agg_q.iterrows():
        if row["n"] < 3:
            ax.annotate(
                f"n={int(row['n'])}",
                (row["q_int"], row["mean"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                color=color,
            )

    if not agg_mp.empty:
        m = agg_mp.iloc[0]
        ax.axhline(m["mean"], color="#2563eb", linestyle="--", linewidth=2, label="Mean pool MAE500")
        q_min, q_max = int(agg_q["q_int"].min()), int(agg_q["q_int"].max())
        ax.fill_between(
            [q_min - 0.5, q_max + 0.5],
            m["mean"] - m["sem"],
            m["mean"] + m["sem"],
            color="#2563eb",
            alpha=0.15,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(sub["q_int"].unique()))
    ax.set_xticklabels([str(q) for q in sorted(sub["q_int"].unique())])
    ax.set_xlabel("Perceiver queries Q (log₂ scale)")
    ax.set_ylabel(VAL_YLABEL)
    ax.set_title(
        "Q sweep @ MAE500 SSL (stab only)\n"
        "Mean ± 1 SEM over finished seeds; ◆ individual seeds",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    path = out_dir / "q_sweep_mae500_stab_vs_meanpool.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def plot_k_sweep_mae500(summary: pd.DataFrame, out_dir: Path) -> None:
    sub = _finished(summary[summary["k"].notna() & (summary["ssl_ep"] == 500)])
    sub = sub[sub["k"].astype(int).isin(K_VALUES)]
    mp = _finished(summary[(summary["head"] == "meanpool") & (summary["ssl_ep"] == 500)])
    if sub.empty:
        print("SKIP plot_k_sweep: no finished DivSpaceTime K runs")
        return

    sub["k_int"] = sub["k"].astype(int)
    agg_k = aggregate_sem(sub, VAL_COL, ["k_int"])
    agg_mp = aggregate_sem(mp, VAL_COL, ["head"]) if not mp.empty else pd.DataFrame()

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    color = "#7c2d12"
    ax.errorbar(
        agg_k["k_int"],
        agg_k["mean"],
        yerr=agg_k["sem"],
        fmt="-",
        color=color,
        capsize=4,
        linewidth=2.2,
        label="DivSpaceTime stab (by K)",
    )
    for _, row in sub.iterrows():
        ax.scatter(row["k_int"], row[VAL_COL], marker="D", s=55, c=color, edgecolors="white", linewidths=0.5, zorder=5)
    for _, row in agg_k.iterrows():
        if row["n"] < 3:
            ax.annotate(
                f"n={int(row['n'])}",
                (row["k_int"], row["mean"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                color=color,
            )

    if not agg_mp.empty:
        m = agg_mp.iloc[0]
        ax.axhline(m["mean"], color="#2563eb", linestyle="--", linewidth=2, label="Mean pool MAE500")
        k_min, k_max = int(agg_k["k_int"].min()), int(agg_k["k_int"].max())
        ax.fill_between(
            [k_min - 0.5, k_max + 0.5],
            m["mean"] - m["sem"],
            m["mean"] + m["sem"],
            color="#2563eb",
            alpha=0.15,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel("DivSpaceTime K (log₂ scale)")
    ax.set_ylabel(VAL_YLABEL)
    ax.set_title(
        "K sweep @ MAE500 SSL (DivSpaceTime stab)\n"
        "Mean ± 1 SEM over finished seeds; ◆ individual seeds",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    path = out_dir / "k_sweep_mae500_stab_vs_meanpool.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def plot_ssl_epoch_q8_vs_meanpool(summary: pd.DataFrame, out_dir: Path) -> None:
    q8 = _finished(
        summary[
            (summary["head"] == "perceiverQ8")
            & summary["is_stab"]
            & summary["ssl_ep"].isin(Q8_STAB_SSL)
        ]
    )
    mp = _finished(
        summary[
            (summary["head"] == "meanpool")
            & summary["ssl_ep"].isin(Q8_STAB_SSL)
        ]
    )
    if q8.empty and mp.empty:
        print("SKIP plot_ssl_epoch: no Q8 stab or meanpool data")
        return

    eps = sorted(set(q8["ssl_ep"].unique()) | set(mp["ssl_ep"].unique()))
    curves: list[pd.DataFrame] = []
    for head, sub, color, label in (
        ("perceiverQ8", q8, "#dc2626", "Perceiver Q8 stab"),
        ("meanpool", mp, "#2563eb", "Mean pool"),
    ):
        if sub.empty:
            continue
        agg = aggregate_sem(sub, VAL_COL, ["ssl_ep"])
        agg["head"] = head
        agg["color"] = color
        agg["label"] = label
        curves.append(agg)

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    fig, ax = plt.subplots(figsize=(10, 6))
    for agg in curves:
        ax.errorbar(
            agg["ssl_ep"],
            agg["mean"],
            yerr=agg["sem"],
            fmt="o-",
            color=agg["color"].iloc[0],
            capsize=4,
            linewidth=2.2,
            markersize=8,
            label=agg["label"].iloc[0],
        )
        sub = q8 if agg["head"].iloc[0] == "perceiverQ8" else mp
        for ssl_ep in agg["ssl_ep"]:
            pts = sub[sub["ssl_ep"] == ssl_ep]
            ax.scatter(
                pts["ssl_ep"],
                pts[VAL_COL],
                marker="D" if agg["head"].iloc[0] == "perceiverQ8" else "o",
                s=50,
                c=agg["color"].iloc[0],
                edgecolors="white",
                linewidths=0.5,
                zorder=5,
                alpha=0.85,
            )
            n = len(pts)
            if n < 3:
                row = agg[agg["ssl_ep"] == ssl_ep].iloc[0]
                ax.annotate(
                    f"n={n}",
                    (ssl_ep, row["mean"]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=9,
                    color=agg["color"].iloc[0],
                )

    ax.set_xlabel("SSL pretrain epoch (MAE backbone)")
    ax.set_ylabel(VAL_YLABEL)
    ax.set_title(
        "SSL epoch sweep: Perceiver Q8 stab vs mean pool\n"
        "Mean ± 1 SEM over finished seeds at each backbone epoch",
        fontsize=11,
    )
    ax.set_xticks(eps)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = out_dir / "ssl_epoch_q8_stab_vs_meanpool.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wandb-summary", type=Path, default=WANDB_SUMMARY)
    ap.add_argument("--fleet-summary", type=Path, default=FLEET_SUMMARY)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--refresh-wandb", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_merged_summary(args.wandb_summary, args.fleet_summary, refresh_wandb=args.refresh_wandb)
    # Mark Q8 SSL fleet runs as stab (recipe=stab on W&B; fleet is arch2 stab preset).
    q8_ssl = summary["run_name"].str.match(r"perceiverQ8-mae\d+-s\d+") & (summary["ssl_ep"] != 500)
    summary.loc[q8_ssl, "is_stab"] = True

    audit = args.out_dir / "plot_inputs_merged_summary.csv"
    summary.to_csv(audit, index=False)
    print(f"Wrote {audit} ({len(summary)} rows)")

    plot_q_sweep_mae500(summary, args.out_dir)
    plot_k_sweep_mae500(summary, args.out_dir)
    plot_ssl_epoch_q8_vs_meanpool(summary, args.out_dir)


if __name__ == "__main__":
    main()
