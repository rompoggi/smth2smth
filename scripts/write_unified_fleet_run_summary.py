#!/usr/bin/env python3
"""Write per-run summary table (best/last val top-1, FT epochs) from unified fleet CSV."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
UNIFIED = REPO / "outputs/unified_fleet/unified_run_summary.csv"
WANDB = REPO / "outputs/ft_mae_scaling_thesis/wandb_ft_mae_scaling_run_summary.csv"
DEFAULT_OUT = REPO / "outputs/unified_fleet"

MEANPOOL_RE = re.compile(r"^meanpool-mae0*(\d+)-s(\d+)$")
Q_RE = re.compile(r"^perceiverQ(\d+)-mae0*(\d+)-s(\d+)$")
DIV_RE = re.compile(r"^DivSpaceTimeK(\d+)-mae0*(\d+)-s(\d+)$")


def _describe_run(name: str) -> str:
    m = MEANPOOL_RE.match(name)
    if m:
        return f"meanpool, SSL={m.group(1)}, seed={m.group(2)}"
    m = Q_RE.match(name)
    if m:
        return f"Perceiver Q={m.group(1)}, SSL={m.group(2)}, seed={m.group(3)}"
    m = DIV_RE.match(name)
    if m:
        return f"DivSpaceTime K={m.group(1)}, SSL={m.group(2)}, seed={m.group(3)}"
    return name


def _family(name: str) -> str:
    if name.startswith("meanpool"):
        return "meanpool"
    if name.startswith("perceiverQ8"):
        return "q8_ssl"
    if name.startswith("perceiverQ16"):
        return "q16_ssl"
    if name.startswith("perceiverQ"):
        return "q500_sweep"
    if name.startswith("DivSpaceTime"):
        return "divspace"
    return "other"


def load_summary(unified: Path, wandb: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if unified.is_file():
        u = pd.read_csv(unified)
        u["source"] = "unified"
        frames.append(u)
    if wandb.is_file():
        w = pd.read_csv(wandb)
        w = w.rename(columns={"ssl_pretrain_epoch": "ssl_ep"})
        w["source"] = "wandb"
        w = w[w["run_name"].str.startswith("meanpool-mae")]
        frames.append(w)
    if not frames:
        raise FileNotFoundError("No summary CSV found")
    df = pd.concat(frames, ignore_index=True)
    df["_prio"] = df["source"].map({"wandb": 0, "unified": 1}).fillna(0)
    df = df.sort_values(["run_name", "_prio"]).drop_duplicates("run_name", keep="last")
    df["params"] = df["run_name"].map(_describe_run)
    df["family"] = df["run_name"].map(_family)
    df["best_val_top1"] = df["best_val_top1"].combine_first(df.get("wandb_best_val_top1"))
    df["last_val_top1"] = df["last_val_top1"].combine_first(df.get("wandb_last_val_top1"))
    df["ft_epochs"] = df.get("wandb_last_epoch")
    if "last_ft_epoch" in df.columns:
        df["ft_epochs"] = df["ft_epochs"].combine_first(df["last_ft_epoch"])
    if "finished_50" in df.columns:
        df.loc[df["finished_50"].fillna(False) & df["ft_epochs"].isna(), "ft_epochs"] = 50
    df["stab"] = df.get("wandb_recipe", pd.Series(dtype=object)).eq("stab") | df.get(
        "stab", pd.Series(False, index=df.index)
    ).fillna(False)
    return df.sort_values(["family", "run_name"])


def to_markdown(df: pd.DataFrame) -> str:
    lines = [
        "# Unified fleet run summary",
        "",
        f"**{len(df)}** runs (unified fleet + meanpool from W&B). Val top-1 = honest official val.",
        "",
        "| Run | Parameters | Best val top-1 | Last val top-1 | FT epochs |",
        "|-----|------------|--------------:|---------------:|----------:|",
    ]
    for _, r in df.iterrows():
        best = r["best_val_top1"]
        last = r["last_val_top1"]
        ep = r["ft_epochs"]
        best_s = f"{100 * best:.2f}%" if pd.notna(best) else "—"
        last_s = f"{100 * last:.2f}%" if pd.notna(last) else "—"
        ep_s = str(int(ep)) if pd.notna(ep) else "—"
        lines.append(
            f"| `{r['run_name']}` | {r['params']} | {best_s} | {last_s} | {ep_s} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unified", type=Path, default=UNIFIED)
    ap.add_argument("--wandb", type=Path, default=WANDB)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_summary(args.unified, args.wandb)
    out_cols = [
        "run_name",
        "family",
        "params",
        "ssl_ep",
        "seed",
        "q",
        "k",
        "stab",
        "best_val_top1",
        "last_val_top1",
        "ft_epochs",
        "host",
        "wandb_id",
        "finished_50",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    table = df[out_cols].copy()
    csv_path = args.out_dir / "run_summary_table.csv"
    md_path = args.out_dir / "run_summary_table.md"
    table.to_csv(csv_path, index=False)
    md_path.write_text(to_markdown(table), encoding="utf-8")
    print(f"Wrote {csv_path} ({len(table)} rows)")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
