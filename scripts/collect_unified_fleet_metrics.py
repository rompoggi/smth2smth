#!/usr/bin/env python3
"""Audit + collect unified fleet metrics (logs, ckpts, W&B) for stab scaling runs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "unified_fleet"
ENTITY = "romain-poggi-ecole-polytechnique"
PROJECT = "smth2smth-frame-ablation"
GROUP = "ft-mae-scaling"

HOSTS = [
    "ablette", "anchois", "anguille", "barbeau", "barbue", "baudroie", "carrelet",
    "gardon", "labre", "lotte", "mulet", "murene", "piranha", "raie", "requin",
    "rouget", "roussette", "saumon", "silure", "sole", "thon", "truite", "lieu",
    "brochet", "gymnote",
]

SSL_EPOCHS = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
SEEDS = [42, 43, 44]
REMOTE_STATUS = REPO / "scripts" / "remote_host_status.py"

Q8_RE = re.compile(r"^perceiverQ8-mae(\d+)-s(\d+)$")
Q16_RE = re.compile(r"^perceiverQ16-mae(\d+)-s(\d+)$")
Q500_RE = re.compile(r"^perceiverQ(\d+)-mae500-s(\d+)$")
DIV_RE = re.compile(r"^DivSpaceTimeK(\d+)-mae500-s(\d+)$")


@dataclass(frozen=True)
class RunSpec:
    run_name: str
    family: str
    kind: str  # q8 | q16 | divspace | q500
    ssl_ep: int
    seed: int
    q: int | None = None
    k: int | None = None
    stab: bool = True


def expected_runs() -> list[RunSpec]:
    out: list[RunSpec] = []
    for ep in SSL_EPOCHS:
        for seed in SEEDS:
            out.append(RunSpec(f"perceiverQ8-mae{ep}-s{seed}", "q8_ssl", "q8", ep, seed, q=8))
            out.append(RunSpec(f"perceiverQ16-mae{ep}-s{seed}", "q16_ssl", "q16", ep, seed, q=16))
    for k in (1, 3, 6, 9):
        for seed in SEEDS:
            out.append(
                RunSpec(
                    f"DivSpaceTimeK{k}-mae500-s{seed}",
                    "divspace",
                    "divspace",
                    500,
                    seed,
                    k=k,
                )
            )
    for q in (2, 4, 8, 16, 32, 64):
        for seed in SEEDS:
            out.append(
                RunSpec(f"perceiverQ{q}-mae500-s{seed}", "q500_sweep", "q8", 500, seed, q=q)
            )
    # Dedupe by run_name (q8 mae500 appears twice)
    seen: dict[str, RunSpec] = {}
    for r in out:
        if r.run_name not in seen:
            seen[r.run_name] = r
        elif r.family == "q500_sweep" and seen[r.run_name].family == "q8_ssl":
            pass  # keep q8_ssl label
    return list(seen.values())


def parse_run_meta(name: str) -> dict:
    for rx, fam in ((Q8_RE, "q8_ssl"), (Q16_RE, "q16_ssl"), (DIV_RE, "divspace"), (Q500_RE, "q500_sweep")):
        m = rx.match(name)
        if m:
            if fam == "divspace":
                return {"family": fam, "ssl_ep": 500, "seed": int(m.group(2)), "q": None, "k": int(m.group(1))}
            return {"family": fam, "ssl_ep": int(m.group(1)), "seed": int(m.group(2)), "q": int(m.group(1)), "k": None}
    return {}


def ssh(host: str, cmd: str, timeout: int = 60) -> str:
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return (r.stdout or "").strip()


def remote_file_size(host: str, path: str) -> int:
    """Remote log size in bytes (0 if missing)."""
    out = ssh(host, f"stat -c %s '{path}' 2>/dev/null || echo 0", timeout=15)
    try:
        return int(out.strip() or "0")
    except ValueError:
        return 0


def scan_host_logs(host: str) -> dict[str, str]:
    cmd = (
        f"find {REPO}/logs/track_a "
        r"\( -name 'perceiverQ*.log' -o -name 'DivSpaceTime*.log' \) -type f 2>/dev/null"
    )
    out = ssh(host, cmd, timeout=180)
    mapping: dict[str, str] = {}
    for path in out.splitlines():
        if not path:
            continue
        base = Path(path).name
        m = re.match(r"^(.+)_\d{8}\.log$", base)
        if m:
            run = m.group(1)
            # Keep newest log per run (paths sorted lexicographically; later dates win)
            mapping[run] = path
    return mapping


def parse_log_metrics(log_text: str) -> dict:
    """Parse completion and val metrics from a training log tail."""
    tail = log_text[-200000:]
    done = "Done. Best val" in tail[-15000:]
    log_best = None
    m = re.search(r"Done\. Best val honest top1: ([0-9]+\.[0-9]+)", tail)
    if m:
        log_best = float(m.group(1))
    epochs = re.findall(
        r"Epoch (\d+)/50 \|.*?val (?:holdout|honest)[^\n]*top1 ([0-9.]+)",
        tail,
    )
    log_last_epoch = int(epochs[-1][0]) if epochs else None
    log_last_val = float(epochs[-1][1]) if epochs else None
    if log_last_epoch is None:
        steps = re.findall(r"\[\d+:\d+:\d+\] step (\d+)/5625", tail[-80000:])
        if steps:
            total_steps = int(steps[-1])
            log_last_epoch = (total_steps - 1) // 5625 + 1
    return {
        "log_done": done,
        "log_best_val_top1": log_best,
        "log_last_epoch": log_last_epoch,
        "log_last_val_top1": log_last_val,
    }


def remote_status(host: str, run: str, kind: str) -> dict:
    cmd = [str(REPO / ".venv/bin/python"), str(REMOTE_STATUS), run, kind]
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, *cmd],
        capture_output=True,
        text=True,
        timeout=90,
    )
    line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "{}"
    return json.loads(line)


def fetch_wandb_long(names: set[str]) -> pd.DataFrame:
    import wandb

    api = wandb.Api()
    rows: list[dict] = []
    for run in api.runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP}, per_page=500):
        cfg = run.config or {}
        name = (cfg.get("training.wandb.name") or run.name).replace("-NoStab", "")
        if name not in names:
            continue
        try:
            hist = run.history(samples=500, keys=["epoch", "val/top1", "val/ema_top1", "_step"])
        except Exception:
            continue
        if hist is None or hist.empty:
            continue
        for _, row in hist.dropna(subset=["epoch"]).iterrows():
            rows.append(
                {
                    "run_name": name,
                    "wandb_id": run.id,
                    "ft_epoch": int(row["epoch"]) if pd.notna(row.get("epoch")) else None,
                    "global_step": row.get("_step"),
                    "val_top1": float(row["val/top1"]) if pd.notna(row.get("val/top1")) else None,
                    "val_ema_top1": float(row["val/ema_top1"])
                    if pd.notna(row.get("val/ema_top1"))
                    else None,
                }
            )
    return pd.DataFrame(rows)


def fetch_wandb_summary(names: set[str]) -> pd.DataFrame:
    import wandb

    api = wandb.Api()
    rows: list[dict] = []
    for run in api.runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP}, per_page=500):
        cfg = run.config or {}
        name = (cfg.get("training.wandb.name") or run.name).replace("-NoStab", "")
        if name not in names:
            continue
        try:
            hist = run.history(samples=300, keys=["epoch", "val/top1", "val/ema_top1", "_step"])
        except Exception as exc:
            rows.append(
                {
                    "run_name": name,
                    "wandb_id": run.id,
                    "wandb_state": run.state,
                    "wandb_error": str(exc),
                }
            )
            continue
        if hist is None or hist.empty or "val/top1" not in hist.columns:
            continue
        hist = hist.dropna(subset=["epoch", "val/top1"])
        if hist.empty:
            continue
        hist["epoch"] = hist["epoch"].astype(int)
        by_ep = hist.groupby("epoch")["val/top1"].last()
        last_ep = int(by_ep.index.max())
        rows.append(
            {
                "run_name": name,
                "wandb_id": run.id,
                "wandb_state": run.state,
                "wandb_best_val_top1": float(by_ep.max()),
                "wandb_last_val_top1": float(by_ep.loc[last_ep]),
                "wandb_last_epoch": last_ep,
                "wandb_finished_50": last_ep >= 50,
                "wandb_recipe": cfg.get("recipe"),
            }
        )
    return pd.DataFrame(rows)


def rsync_file(host: str, remote: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["rsync", "-az", f"{host}:{remote}", str(local)],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


def ckpt_paths(run: str, kind: str) -> list[str]:
    if kind == "divspace":
        seed_m = re.search(r"-s(\d+)$", run)
        seed = seed_m.group(1) if seed_m else "42"
        base = REPO / f"checkpoints/track_a/divspace_s{seed}"
        return [str(base / f"{run}.pt"), str(base / f"{run}.last.pt")]
    base = REPO / "checkpoints/track_a/videomaev2+ft"
    return [str(base / f"{run}.pt"), str(base / f"{run}.last.pt")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true", help="rsync logs+ckpts to outputs/unified_fleet/collected/")
    ap.add_argument("--skip-wandb", action="store_true")
    args = ap.parse_args()

    specs = expected_runs()
    names = {s.run_name for s in specs}
    print(f"Expected unique runs: {len(specs)}")

    # Index logs per host
    host_logs: dict[str, dict[str, str]] = {}
    run_host_log: dict[str, tuple[str, str]] = {}
    for host in HOSTS:
        try:
            host_logs[host] = scan_host_logs(host)
            print(f"  {host}: {len(host_logs[host])} logs")
        except Exception as exc:
            print(f"  WARN {host}: {exc}")
            host_logs[host] = {}
    for host, logs in host_logs.items():
        for run, path in logs.items():
            if run not in names:
                continue
            prev = run_host_log.get(run)
            if prev is None:
                run_host_log[run] = (host, path)
                continue
            new_name = Path(path).name
            old_name = Path(prev[1]).name
            if new_name > old_name:
                run_host_log[run] = (host, path)
            elif new_name == old_name and remote_file_size(host, path) > remote_file_size(
                prev[0], prev[1]
            ):
                # Same dated log copied/resumed on another host — keep the larger file.
                run_host_log[run] = (host, path)

    rows: list[dict] = []
    for spec in sorted(specs, key=lambda s: s.run_name):
        host, log_remote = run_host_log.get(spec.run_name, (None, None))
        status: dict = {}
        if host:
            try:
                status = remote_status(host, spec.run_name, spec.kind)
            except Exception as exc:
                status = {"error": str(exc)}

        log_done = bool(status.get("log_done"))
        ckpt_ep = status.get("ckpt_epoch")
        finished = log_done or (ckpt_ep is not None and int(ckpt_ep) >= 50)

        best = status.get("log_best_val_top1") or status.get("ckpt_best_val_top1")
        last = status.get("log_last_val_top1") or status.get("ckpt_last_val_top1")
        last_ep = status.get("log_last_epoch") or ckpt_ep

        collected_log = None
        collected_ckpt_best = None
        collected_ckpt_last = None
        if host and log_remote and host != "gymnote":
            try:
                log_body = ssh(host, f"tail -c 250000 '{log_remote}'", timeout=60)
                lp = parse_log_metrics(log_body)
                if lp.get("log_done"):
                    log_done = True
                    finished = True
                if lp.get("log_best_val_top1") is not None:
                    status["log_best_val_top1"] = lp["log_best_val_top1"]
                    best = lp["log_best_val_top1"]
                if lp.get("log_last_val_top1") is not None:
                    status["log_last_val_top1"] = lp["log_last_val_top1"]
                    last = lp["log_last_val_top1"]
                if lp.get("log_last_epoch") is not None:
                    last_ep = lp["log_last_epoch"]
            except Exception:
                pass
        elif host == "gymnote" and log_remote:
            try:
                lp = parse_log_metrics(Path(log_remote).read_text(errors="replace"))
                if lp.get("log_done"):
                    log_done = True
                    finished = True
                if lp.get("log_best_val_top1") is not None:
                    best = lp["log_best_val_top1"]
                if lp.get("log_last_val_top1") is not None:
                    last = lp["log_last_val_top1"]
                if lp.get("log_last_epoch") is not None:
                    last_ep = lp["log_last_epoch"]
            except Exception:
                pass

        if args.collect and host and log_remote:
            dest_host = host
            cl = OUT / "collected" / "logs" / dest_host / Path(log_remote).name
            if host == "gymnote":
                cl.parent.mkdir(parents=True, exist_ok=True)
                cl.write_bytes(Path(log_remote).read_bytes())
                collected_log = str(cl.relative_to(REPO))
            elif rsync_file(host, log_remote, cl):
                collected_log = str(cl.relative_to(REPO))
            for ck in ckpt_paths(spec.run_name, spec.kind):
                ck_name = Path(ck).name
                local_ck = OUT / "collected" / "checkpoints" / spec.family / ck_name
                if rsync_file(host, ck, local_ck):
                    if ck.endswith(".last.pt"):
                        collected_ckpt_last = str(local_ck.relative_to(REPO))
                    else:
                        collected_ckpt_best = str(local_ck.relative_to(REPO))

        rows.append(
            {
                "run_name": spec.run_name,
                "family": spec.family,
                "kind": spec.kind,
                "ssl_ep": spec.ssl_ep,
                "seed": spec.seed,
                "q": spec.q,
                "k": spec.k,
                "stab": spec.stab,
                "host": host,
                "log_path_remote": log_remote,
                "collected_log": collected_log,
                "collected_ckpt_best": collected_ckpt_best,
                "collected_ckpt_last": collected_ckpt_last,
                "log_done": log_done,
                "finished_50": finished,
                "running": status.get("running"),
                "last_ft_epoch": last_ep,
                "log_best_val_top1": status.get("log_best_val_top1"),
                "log_last_val_top1": status.get("log_last_val_top1"),
                "ckpt_best_val_top1": status.get("ckpt_best_val_top1"),
                "ckpt_last_val_top1": status.get("ckpt_last_val_top1"),
                "best_val_top1": best,
                "last_val_top1": last,
            }
        )

    df = pd.DataFrame(rows)

    if not args.skip_wandb:
        long_df = fetch_wandb_long(names)
        if not long_df.empty:
            long_path = OUT / "unified_epoch_val_long.csv"
            long_df.to_csv(long_path, index=False)
            print(f"Wrote {long_path} ({len(long_df)} rows)")
        wb = fetch_wandb_summary(names)
        if not wb.empty:
            wb = wb.sort_values("wandb_last_epoch", ascending=False, na_position="last")
            wb = wb.drop_duplicates(subset=["run_name"], keep="first")
            df = df.merge(wb, on="run_name", how="left")
            df["best_val_top1"] = df["best_val_top1"].combine_first(df["wandb_best_val_top1"])
            df["last_val_top1"] = df["last_val_top1"].combine_first(df["wandb_last_val_top1"])

    if "wandb_finished_50" not in df.columns:
        df["wandb_finished_50"] = False
    df["finished_50"] = (
        df["log_done"].fillna(False)
        | (df["last_ft_epoch"].fillna(0) >= 50)
        | df["wandb_finished_50"].fillna(False)
    )

    OUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT / "unified_run_summary.csv"
    df.to_csv(summary_path, index=False)

    # Audit report
    no_log = df[df["host"].isna()]
    not_done = df[~df["finished_50"].fillna(False)]
    done = df[df["finished_50"].fillna(False)]

    print(f"\nWrote {summary_path} ({len(df)} rows)")
    print(f"DONE (ep50 / log Done): {len(done)}")
    print(f"NOT DONE: {len(not_done)}")
    print(f"NO LOG FOUND: {len(no_log)}")

    if len(no_log):
        print("\n--- NO LOG ---")
        for n in no_log["run_name"].tolist():
            print(f"  {n}")

    if len(not_done):
        print("\n--- NOT DONE ---")
        for _, r in not_done.iterrows():
            print(
                f"  {r['run_name']:34} host={r['host']} ep={r['last_ft_epoch']} "
                f"log_done={r['log_done']} wandb_ep={r.get('wandb_last_epoch', '')}"
            )

    audit_path = OUT / "completion_audit.txt"
    audit_path.write_text(
        f"expected={len(df)} done={len(done)} not_done={len(not_done)} no_log={len(no_log)}\n"
        + "\n".join(not_done["run_name"].tolist()),
        encoding="utf-8",
    )
    print(f"Wrote {audit_path}")


if __name__ == "__main__":
    main()
