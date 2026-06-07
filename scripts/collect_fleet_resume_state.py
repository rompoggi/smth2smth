#!/usr/bin/env python3
"""Print resume state for active fleet (host:run:kind:params)."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path("/Data/romain.poggi/smth2smth")

# host:run:kind — kind = divspace|q8|q16 ; gymnote assignments (2026-06-03)
FLEET = [
    ("ablette", "DivSpaceTimeK1-mae500-s43", "divspace", "1:43:ntuwnv5r"),
    ("silure", "DivSpaceTimeK3-mae500-s43", "divspace", "3:43:i7ado1gd"),
    ("barbue", "DivSpaceTimeK6-mae500-s43", "divspace", "6:43:txwwef5b"),
    ("carrelet", "DivSpaceTimeK9-mae500-s43", "divspace", "9:43:0sguvl8z"),
    ("piranha", "DivSpaceTimeK1-mae500-s44", "divspace", "1:44:9cdrb7z6"),
    ("raie", "DivSpaceTimeK3-mae500-s44", "divspace", "3:44:z0zwj43l"),
    ("requin", "DivSpaceTimeK6-mae500-s44", "divspace", "6:44:01gnvhqm"),
    ("roussette", "DivSpaceTimeK9-mae500-s44", "divspace", "9:44:533jwygf"),
    ("gardon", "DivSpaceTimeK1-mae500-s42", "divspace", "1:42:pu325vfh"),
    ("lieu", "perceiverQ16-mae500-s43", "q16", "43:6idgo6j5"),
    ("brochet", "perceiverQ16-mae500-s44", "q16", "44:24fxahew"),
    ("anchois", "perceiverQ8-mae100-s42", "q8", "42:100:8:715f76zo"),
    ("labre", "perceiverQ8-mae100-s43", "q8", "43:100:8:vs4mcp2v"),
    ("truite", "perceiverQ8-mae200-s42", "q8", "42:200:8:culr1ccl"),
    ("thon", "perceiverQ8-mae200-s43", "q8", "43:200:8:4wte56ck"),
    ("rouget", "perceiverQ8-mae300-s42", "q8", "42:300:8:n6ibgn26"),
    ("sole", "perceiverQ8-mae300-s43", "q8", "43:300:8:jscsvpkp"),
    ("mulet", "perceiverQ8-mae400-s42", "q8", "42:400:8:x20b72jl"),
    ("murene", "perceiverQ8-mae400-s43", "q8", "43:400:8:zufhx731"),
]

SCRIPT = r'''
import re, glob, os, torch, json
repo = "/Data/romain.poggi/smth2smth"
run = "{run}"
kind = "{kind}"
subs = []
if kind == "divspace":
    for s in ("divspace_s43", "divspace_s44", "divspace_s42"):
        subs.append(f"{{repo}}/checkpoints/track_a/{{s}}/{{run}}.last.pt")
else:
    subs.append(f"{{repo}}/checkpoints/track_a/videomaev2+ft/{{run}}.last.pt")
ckpt_gs = 0
ep = None
has_ckpt = False
for p in subs:
    if os.path.isfile(p):
        c = torch.load(p, map_location="cpu", weights_only=False)
        ex = c.get("extra") or {{}}
        ckpt_gs = int(ex.get("global_step") or 0)
        ep = ex.get("epoch")
        has_ckpt = True
        break
logs = sorted(glob.glob(repo + "/logs/track_a/**/" + run + "_*.log", recursive=True)
              + glob.glob(repo + "/logs/track_a/" + run + "_*.log"))
wandb_gs = 0
wid = ""
done = False
if logs:
    t = open(logs[-1]).read()
    ws = re.findall(r"current step (\d+)", t)
    wandb_gs = int(ws[-1]) if ws else 0
    ids = re.findall(r"/runs/([a-z0-9]+)", t)
    wid = ids[-1] if ids else ""
    done = "Done. Best val" in t[-12000:]
rgs = max(ckpt_gs, wandb_gs)
print(json.dumps({{"ckpt_gs": ckpt_gs, "wandb_gs": wandb_gs, "rgs": rgs, "ep": ep,
                  "done": done, "has_ckpt": has_ckpt, "wid": wid}}))
'''


def main() -> None:
    for host, run, kind, params in FLEET:
        cmd = ["ssh", "-o", "BatchMode=yes", host, f"{REPO}/.venv/bin/python", "-c",
               SCRIPT.format(run=run, kind=kind)]
        try:
            out = subprocess.check_output(cmd, text=True, timeout=60).strip()
            import json
            st = json.loads(out.splitlines()[-1])
        except Exception as e:
            print(f"{host}:{run}:{kind}:{params}:ERR:{e}", file=sys.stderr)
            continue
        if st.get("done") or (st.get("ep") is not None and int(st["ep"]) >= 50):
            status = "SKIP_DONE"
        elif not st.get("has_ckpt"):
            status = "SKIP_NOCKPT"
        else:
            status = "RESUME"
        print(
            f"{host}:{run}:{kind}:{params}:{status}:"
            f"rgs={st['rgs']}:ckpt={st['ckpt_gs']}:wandb={st['wandb_gs']}:ep={st.get('ep')}"
        )


if __name__ == "__main__":
    main()
