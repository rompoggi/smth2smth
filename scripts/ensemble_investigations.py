#!/usr/bin/env python3
"""Ensembling investigations #1, #2, #4 (offline on cached val logits).

#1 Fair diversity test  — OOF gain +/- CI for size-matched member sets.
#2 Weak-member ablation — drop mean-pool / DivST from diverse-4 and arch-only.
#4 Val->LB inversion    — residual (val - public-basic LB) vs mp/DivST weight share.

All ensemble scores are 5-fold stratified softmax OOF over the full val set,
with paired bootstrap CIs (resample clips, recompute ens - best-single).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from smth2smth.ensemble.combiners import sanitize_logits
from smth2smth.ensemble.optimize import metrics_from_logits

from ensemble_heads_val_offline import BOOTSTRAP, SPLIT_SEED, _oof_scores, _top1_ci  # noqa: E402

CACHE = REPO / "outputs/ensemble/mae500_stab_val"
Y = np.load(CACHE / "labels_val.npy")


def load(m: str) -> np.ndarray:
    return sanitize_logits(np.load(CACHE / "logits" / f"{m}.npy"))


def single_top1(m: str) -> float:
    return float(metrics_from_logits(load(m), Y)[0])


def oof_top1(members: list[str]) -> tuple[np.ndarray, float]:
    oof = _oof_scores("softmax", [load(m) for m in members], Y)
    return oof, float(metrics_from_logits(oof, Y)[0])


def gain_ci(ens_scores: np.ndarray, best_single: str, n_boot: int = BOOTSTRAP) -> tuple[float, float]:
    ens_c = (ens_scores.argmax(1) == Y).astype(float)
    sg_c = (load(best_single).argmax(1) == Y).astype(float)
    rng = np.random.default_rng(SPLIT_SEED)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(len(Y), len(Y), replace=True)
        boots.append(100.0 * (ens_c[idx].mean() - sg_c[idx].mean()))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def investigate_1() -> pd.DataFrame:
    sets = {
        "seed-only (Q8 x3 seeds)": ["perceiverQ8-mae500-s42", "perceiverQ8-mae500-s43", "perceiverQ8-mae500-s44"],
        "arch-matched (Q2/Q4/Q8 @s42)": ["perceiverQ2-mae500-s42", "perceiverQ4-mae500-s42", "perceiverQ8-mae500-s42"],
        "arch-weak (mp/Q8/DivST @s42)": ["meanpool-mae500-s42", "perceiverQ8-mae500-s42", "DivSpaceTimeK9-mae500-s42"],
        "arch-3seed (mp/Q8/DivST x3)": [
            f"{h}-mae500-{s}"
            for h in ("meanpool", "perceiverQ8", "DivSpaceTimeK9")
            for s in ("s42", "s43", "s44")
        ],
        "all-axes (4 heads, 3 seeds)": ["meanpool-mae500-s42", "perceiverQ8-mae500-s44", "DivSpaceTimeK9-mae500-s43", "perceiverQ16-mae500-s43"],
    }
    rows = []
    for name, members in sets.items():
        oof, ens = oof_top1(members)
        singles = {m: single_top1(m) for m in members}
        best_m = max(singles, key=singles.get)
        lo, hi = _top1_ci(oof, Y)
        glo, ghi = gain_ci(oof, best_m)
        rows.append({
            "set": name, "n": len(members), "ens_top1": ens, "ens_ci": f"[{lo:.2f},{hi:.2f}]",
            "best_single": singles[best_m], "best_member": best_m.replace("-mae500", ""),
            "gain_pp": ens - singles[best_m], "gain_ci": f"[{glo:+.2f},{ghi:+.2f}]",
        })
    df = pd.DataFrame(rows)
    print("\n===== #1 Fair diversity test (softmax OOF, full val) =====")
    print(df.to_string(index=False))
    df.to_csv(CACHE / "plots" / "investig1_fair_diversity.csv", index=False)
    return df


def investigate_2() -> pd.DataFrame:
    base = {
        "diverse-4": ["meanpool-mae500-s42", "perceiverQ8-mae500-s42", "DivSpaceTimeK9-mae500-s42", "perceiverQ16-mae500-s43"],
        "arch-only": ["meanpool-mae500-s42", "perceiverQ8-mae500-s42", "DivSpaceTimeK9-mae500-s42"],
    }
    rows = []
    for set_name, members in base.items():
        full_top1 = oof_top1(members)[1]
        ablations = {"full": members}
        ablations["drop mean-pool"] = [m for m in members if "meanpool" not in m]
        ablations["drop DivST-K9"] = [m for m in members if "DivSpaceTime" not in m]
        ablations["drop both"] = [m for m in members if "meanpool" not in m and "DivSpaceTime" not in m]
        for abl, mem in ablations.items():
            if len(mem) == 1:
                t1 = single_top1(mem[0])
            else:
                t1 = oof_top1(mem)[1]
            rows.append({"set": set_name, "ablation": abl, "n": len(mem),
                         "members": "+".join(m.split("-mae500")[0] for m in mem),
                         "top1": t1, "delta_vs_full": t1 - full_top1})
    df = pd.DataFrame(rows)
    print("\n===== #2 Weak-member ablation (softmax OOF) =====")
    print(df.to_string(index=False))
    df.to_csv(CACHE / "plots" / "investig2_ablation.csv", index=False)
    return df


def investigate_4() -> pd.DataFrame:
    pts = pd.read_csv(CACHE / "val_lb_points.csv")
    members_of = {
        "ens-seed-Q8x3": ["Q8", "Q8", "Q8"],
        "ens-arch-s42": ["mp", "Q8", "DivST"],
        "ens-allaxes": ["mp", "Q8", "DivST", "Q16"],
        "ens-diverse4-s42": ["mp", "Q8", "DivST", "Q16"],
    }

    def share(name: str) -> float:
        if name in members_of:
            ms = members_of[name]
            return sum(m in ("mp", "DivST") for m in ms) / len(ms)
        return 1.0 if ("meanpool" in name or "DivSpaceTime" in name) else 0.0

    pts["weak_share"] = pts["name"].map(share)
    pts["residual"] = pts["val_top1"] - pts["lb_public"]  # >0 => val > LB (under-transfer)
    r = float(np.corrcoef(pts["weak_share"], pts["residual"])[0, 1])
    print("\n===== #4 Val->LB inversion =====")
    print(pts[["name", "kind", "val_top1", "lb_public", "residual", "weak_share"]].to_string(index=False))
    print(f"\nPearson r(weak_share, residual) = {r:+.3f}")
    weak = pts[pts.weak_share >= 0.5]["residual"]
    lean = pts[pts.weak_share < 0.5]["residual"]
    print(f"mean residual: weak-heavy (>=0.5) = {weak.mean():+.2f}pp  vs  lean (<0.5) = {lean.mean():+.2f}pp")
    verdict = ("weak members under-transfer on public" if r > 0.4 and weak.mean() - lean.mean() > 0.4
               else "uniform noise (no weak-member transfer effect)")
    print(f"VERDICT: {verdict}")
    pts.to_csv(CACHE / "plots" / "investig4_residuals.csv", index=False)
    return pts


if __name__ == "__main__":
    investigate_1()
    investigate_2()
    investigate_4()
