#!/usr/bin/env python3
"""Usage: remote_host_status.py <run> <kind:divspace|q8|q16>"""
import json
import re
import glob
import os
import sys
import torch
from subprocess import run as sprun

repo = "/Data/romain.poggi/smth2smth"
run = sys.argv[1]
kind = sys.argv[2]
subs = []
if kind == "divspace":
    for s in ("divspace_s43", "divspace_s44", "divspace_s42"):
        subs.append(f"{repo}/checkpoints/track_a/{s}/{run}.last.pt")
        subs.append(f"{repo}/checkpoints/track_a/{s}/{run}.pt")
else:
    subs.append(f"{repo}/checkpoints/track_a/videomaev2+ft/{run}.last.pt")
    subs.append(f"{repo}/checkpoints/track_a/videomaev2+ft/{run}.pt")
last_pt = None
best_pt = None
for p in subs:
    if not os.path.isfile(p):
        continue
    if p.endswith(".last.pt"):
        last_pt = p
    elif p.endswith(".pt"):
        best_pt = p
ckpt_path = last_pt or best_pt
ex = {}
if ckpt_path:
    c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ex = c.get("extra") or {}
logs = sorted(
    glob.glob(repo + "/logs/track_a/**/" + run + "_*.log", recursive=True)
    + glob.glob(repo + "/logs/track_a/" + run + "_*.log")
)
log_path = logs[-1] if logs else ""
done = False
log_best = None
log_last_epoch = None
log_last_val = None
if log_path:
    t = open(log_path).read()
    done = "Done. Best val" in t[-15000:]
    m = re.search(r"Done\. Best val honest top1: ([0-9]+\.[0-9]+)", t)
    if m:
        log_best = float(m.group(1))
    epochs = re.findall(
        r"Epoch (\d+)/50 \|.*?val (?:holdout|honest)[^\n]*top1 ([0-9.]+)", t
    )
    if epochs:
        log_last_epoch = int(epochs[-1][0])
        log_last_val = float(epochs[-1][1])
procs = sprun(
    ["pgrep", "-af", f"training.wandb.name={run}"],
    capture_output=True,
    text=True,
)
running = "python" in procs.stdout and "smth2smth.pipelines.train" in procs.stdout
print(
    json.dumps(
        {
            "ckpt_epoch": ex.get("epoch"),
            "ckpt_global_step": ex.get("global_step"),
            "ckpt_best_val_top1": ex.get("val_top1"),
            "ckpt_last_val_top1": ex.get("latest_val_top1"),
            "log_path": log_path,
            "log_done": done,
            "log_best_val_top1": log_best,
            "log_last_epoch": log_last_epoch,
            "log_last_val_top1": log_last_val,
            "running": running,
        }
    )
)
