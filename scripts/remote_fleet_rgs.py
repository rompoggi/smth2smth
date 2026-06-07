#!/usr/bin/env python3
"""Usage: remote_fleet_rgs.py <run> <kind:divspace|q8|q16>"""
import json
import re
import glob
import os
import sys
import torch

repo = "/Data/romain.poggi/smth2smth"
run = sys.argv[1]
kind = sys.argv[2]
subs = []
if kind == "divspace":
    for s in ("divspace_s43", "divspace_s44", "divspace_s42"):
        subs.append(f"{repo}/checkpoints/track_a/{s}/{run}.last.pt")
else:
    subs.append(f"{repo}/checkpoints/track_a/videomaev2+ft/{run}.last.pt")
ckpt_gs = 0
ep = None
has_ckpt = False
for p in subs:
    if os.path.isfile(p):
        c = torch.load(p, map_location="cpu", weights_only=False)
        ex = c.get("extra") or {}
        ckpt_gs = int(ex.get("global_step") or 0)
        ep = ex.get("epoch")
        has_ckpt = True
        break
logs = sorted(
    glob.glob(repo + "/logs/track_a/**/" + run + "_*.log", recursive=True)
    + glob.glob(repo + "/logs/track_a/" + run + "_*.log")
)
wandb_gs = 0
done = False
log_tail_gs = 0
if logs:
    t = open(logs[-1]).read()
    ws = [int(x) for x in re.findall(r"current step (\d+)", t)]
    wandb_gs = max(ws) if ws else 0
    done = "Done. Best val" in t[-12000:]
    # Infer global step from last [HH:MM:SS] step S/5625 in tail + ckpt epoch
    step_m = re.findall(r"\[\d+:\d+:\d+\] step (\d+)/5625", t[-80000:])
    if step_m and ep is not None:
        log_tail_gs = int(ep) * 5625 + int(step_m[-1])
rgs = max(ckpt_gs, wandb_gs, log_tail_gs)
print(
    json.dumps(
        {
            "ckpt_gs": ckpt_gs,
            "wandb_gs": wandb_gs,
            "rgs": rgs,
            "ep": ep,
            "done": done,
            "has_ckpt": has_ckpt,
        }
    )
)
