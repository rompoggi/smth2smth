#!/usr/bin/env python3
"""Collect gymnote fleet run status + W&B epoch metrics; write CSVs and print summary."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "gymnote_fleet"
ENTITY = "romain-poggi-ecole-polytechnique"
PROJECT = "smth2smth-frame-ablation"
GROUP = "ft-mae-scaling"

# host, run, kind, wandb_id (from resume_fleet_gymnote.sh)
FLEET = [
    ("ablette", "DivSpaceTimeK1-mae500-s43", "divspace", "ntuwnv5r"),
    ("silure", "DivSpaceTimeK3-mae500-s43", "divspace", "i7ado1gd"),
    ("barbue", "DivSpaceTimeK6-mae500-s43", "divspace", "txwwef5b"),
    ("carrelet", "DivSpaceTimeK9-mae500-s43", "divspace", "0sguvl8z"),
    ("piranha", "DivSpaceTimeK1-mae500-s44", "divspace", "9cdrb7z6"),
    ("raie", "DivSpaceTimeK3-mae500-s44", "divspace", "z0zwj43l"),
    ("requin", "DivSpaceTimeK6-mae500-s44", "divspace", "01gnvhqm"),
    ("roussette", "DivSpaceTimeK9-mae500-s44", "divspace", "533jwygf"),
    ("gardon", "DivSpaceTimeK1-mae500-s42", "divspace", "pu325vfh"),
    ("lieu", "perceiverQ16-mae500-s43", "q16", "6idgo6j5"),
    ("brochet", "perceiverQ16-mae500-s44", "q16", "24fxahew"),
    ("anchois", "perceiverQ8-mae100-s42", "q8", "715f76zo"),
    ("labre", "perceiverQ8-mae100-s43", "q8", "vs4mcp2v"),
    ("truite", "perceiverQ8-mae200-s42", "q8", "culr1ccl"),
    ("thon", "perceiverQ8-mae200-s43", "q8", "4wte56ck"),
    ("rouget", "perceiverQ8-mae300-s42", "q8", "n6ibgn26"),
    ("sole", "perceiverQ8-mae300-s43", "q8", "jscsvpkp"),
    ("mulet", "perceiverQ8-mae400-s42", "q8", "x20b72jl"),
    ("murene", "perceiverQ8-mae400-s43", "q8", "zufhx731"),
]

FLEET_NAMES = {r[1] for r in FLEET}
REMOTE_SCRIPT = REPO / "scripts" / "remote_host_status.py"


def ssh_host_state(host: str, run: str, kind: str) -> dict:
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        host,
        f"{REPO}/.venv/bin/python",
        str(REMOTE_SCRIPT),
        run,
        kind,
    ]
    out = subprocess.check_output(cmd, text=True, timeout=90)
    return json.loads(out.strip().splitlines()[-1])


def fetch_wandb_history() -> pd.DataFrame:
    import wandb

    api = wandb.Api()
    rows: list[dict] = []
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP}, per_page=300)
    id_to_name = {wid: rname for _, rname, _, wid in FLEET}
    for run in runs:
        cfg = run.config or {}
        name = (
            cfg.get("training.wandb.name")
            or cfg.get("training/wandb/name")
            or run.name
        )
        if name not in FLEET_NAMES and run.id not in id_to_name:
            continue
        if name not in FLEET_NAMES:
            name = id_to_name.get(run.id, name)
        try:
            hist = run.history(samples=500, keys=["epoch", "val/top1", "val/ema_top1", "_step"])
        except Exception as exc:
            print(f"WARN wandb {name}: {exc}")
            continue
        if hist is None or hist.empty:
            continue
        dfh = hist
        if dfh.empty or "val/top1" not in dfh.columns:
            continue
        dfh = dfh.dropna(subset=["epoch"], how="any")
        for _, row in dfh.iterrows():
            rows.append(
                {
                    "run_name": name,
                    "wandb_id": run.id,
                    "wandb_state": run.state,
                    "global_step": row.get("_step"),
                    "ft_epoch": int(row["epoch"]) if pd.notna(row.get("epoch")) else None,
                    "val_top1": float(row["val/top1"]) if pd.notna(row.get("val/top1")) else None,
                    "val_ema_top1": float(row["val/ema_top1"])
                    if pd.notna(row.get("val/ema_top1"))
                    else None,
                }
            )
    return pd.DataFrame(rows)


def summarize_wandb(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()
    rows = []
    for name, grp in long_df.groupby("run_name"):
        g = grp.dropna(subset=["ft_epoch", "val_top1"])
        if g.empty:
            continue
        by_ep = g.groupby("ft_epoch")["val_top1"].last()
        rows.append(
            {
                "run_name": name,
                "wandb_best_val_top1": float(by_ep.max()),
                "wandb_last_val_top1": float(by_ep.loc[int(by_ep.index.max())]),
                "wandb_last_epoch": int(by_ep.index.max()),
                "wandb_finished_50": int(by_ep.index.max()) >= 50,
                "wandb_state": grp["wandb_state"].iloc[0],
                "wandb_id": grp["wandb_id"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hosts = sorted({h for h, *_ in FLEET})
    for h in hosts:
        subprocess.run(
            ["rsync", "-az", str(REMOTE_SCRIPT), f"{h}:{REMOTE_SCRIPT}"],
            check=False,
            capture_output=True,
        )
    host_rows = []
    for host, run, kind, wid in FLEET:
        print(f"host {host} {run}...")
        try:
            st = ssh_host_state(host, run, kind)
        except Exception as exc:
            st = {"error": str(exc)}
        host_rows.append(
            {
                "host": host,
                "run_name": run,
                "kind": kind,
                "wandb_id": wid,
                **{k: v for k, v in st.items() if k != "error"},
                "error": st.get("error"),
            }
        )

    host_df = pd.DataFrame(host_rows)

    print("Fetching W&B...")
    long_df = fetch_wandb_history()
    long_path = OUT / "fleet_epoch_val_long.csv"
    long_df.to_csv(long_path, index=False)

    wandb_sum = summarize_wandb(long_df)
    summary = host_df.merge(wandb_sum, on="run_name", how="left")

    def status_row(r: pd.Series) -> str:
        err = r.get("error")
        if pd.notna(err) and str(err).strip():
            return "ERROR"
        if r.get("running"):
            return "RUNNING"
        ep = r.get("ckpt_epoch")
        wfin = r.get("wandb_finished_50")
        if r.get("log_done") or (pd.notna(ep) and int(ep) >= 50) or wfin is True or wfin == 1:
            return "DONE"
        if pd.notna(ep) and int(ep) > 0:
            return "IN_PROGRESS"
        return "UNKNOWN"

    summary["status"] = summary.apply(status_row, axis=1)
    summary["best_val_top1"] = summary[
        ["ckpt_best_val_top1", "log_best_val_top1", "wandb_best_val_top1"]
    ].max(axis=1)
    summary["last_val_top1"] = summary[
        ["ckpt_last_val_top1", "log_last_val_top1", "wandb_last_val_top1"]
    ].bfill(axis=1).iloc[:, 0]
    summary["last_epoch"] = summary[
        ["ckpt_epoch", "log_last_epoch", "wandb_last_epoch"]
    ].max(axis=1)

    summary_path = OUT / "fleet_run_summary.csv"
    summary.to_csv(summary_path, index=False)

    done = summary[summary["status"] == "DONE"]
    running = summary[summary["status"] == "RUNNING"]
    prog = summary[summary["status"] == "IN_PROGRESS"]

    print(f"\nWrote {long_path} ({len(long_df)} rows)")
    print(f"Wrote {summary_path}")
    print(f"DONE: {len(done)}  RUNNING: {len(running)}  IN_PROGRESS: {len(prog)}")
    if len(running):
        print("RUNNING:", ", ".join(running["run_name"].tolist()))
    if len(prog):
        print("IN_PROGRESS:", ", ".join(prog["run_name"].tolist()))
    not_done = summary[~summary["status"].isin(["DONE"])]
    if len(not_done):
        print("\nNot DONE:")
        print(not_done[["host", "run_name", "status", "last_epoch", "best_val_top1"]].to_string())


if __name__ == "__main__":
    main()
