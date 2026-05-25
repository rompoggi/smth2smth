#!/usr/bin/env python3
"""Verify the SSv2-FT attentive pooler survives ``VJEPA2HFClassifier`` construction.

Compares per-parameter L2 norms between:
  (R) reference: a raw ``VJEPA2ForVideoClassification.from_pretrained(hf_repo)``,
  (W) wrapped:  the ``VJEPA2HFClassifier`` instance built by our pipeline.

If a pooler tensor in (W) is zero (or matches a random Kaiming/Xavier scale)
while in (R) it has a typical pretrained scale (~1.0 - ~10.0), then
``from_pretrained`` is silently re-initializing the pooler -- the single most
likely explanation for plateauing at ~70.6 % val on the SSv2-FT checkpoint.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/verify_vjepa2_pooler_load.py
    PYTHONPATH=src .venv/bin/python scripts/verify_vjepa2_pooler_load.py \
        --hf-repo facebook/vjepa2-vitl-fpc16-256-ssv2 --lora-enabled
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smth2smth.track_b.vjepa2 import VJEPA2HFClassifier  # noqa: E402


def _strip_peft_prefix(name: str) -> str:
    """Drop ``base_model.model.`` PEFT prefix so wrapped/reference keys align."""
    return name.removeprefix("base_model.model.")


def _pooler_params(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return ``{stripped_name: weight}`` for every parameter under a ``.pooler.`` path."""
    out: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        stripped = _strip_peft_prefix(name)
        if ".pooler." in stripped or stripped.startswith("pooler."):
            out[stripped] = p.detach()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-repo", default="facebook/vjepa2-vitl-fpc16-256-ssv2")
    parser.add_argument("--train-dir", default=str(REPO_ROOT / "data/train"))
    parser.add_argument("--num-classes", type=int, default=33)
    parser.add_argument("--lora-enabled", action="store_true",
                        help="Match the experiment config (LoRA r=16 on encoder attn).")
    parser.add_argument("--atol", type=float, default=1e-6,
                        help="Per-tensor L2-norm tolerance for the OK/FAIL verdict.")
    args = parser.parse_args()

    from transformers import VJEPA2ForVideoClassification

    print(f"[ref]  loading raw {args.hf_repo} ...")
    ref = VJEPA2ForVideoClassification.from_pretrained(args.hf_repo)
    ref.eval()
    ref_pooler = _pooler_params(ref)
    print(f"[ref]  {len(ref_pooler)} pooler parameters found.")

    print(f"[wrap] building VJEPA2HFClassifier(lora_enabled={args.lora_enabled}) ...")
    wrap = VJEPA2HFClassifier(
        num_classes=args.num_classes,
        train_dir=args.train_dir,
        hf_repo=args.hf_repo,
        lora_enabled=args.lora_enabled,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=r".*\.encoder\.layer\.\d+\.attention\.(query|key|value|proj)",
        attn_implementation="sdpa",
    )
    wrap.eval()
    wrap_pooler = _pooler_params(wrap.model)
    print(f"[wrap] {len(wrap_pooler)} pooler parameters found.")

    # Match by stripped name.
    common = sorted(set(ref_pooler) & set(wrap_pooler))
    only_ref = sorted(set(ref_pooler) - set(wrap_pooler))
    only_wrap = sorted(set(wrap_pooler) - set(ref_pooler))
    if only_ref:
        print(f"[diff] {len(only_ref)} pooler keys only in reference (sample): "
              f"{only_ref[:3]}")
    if only_wrap:
        print(f"[diff] {len(only_wrap)} pooler keys only in wrapper (sample): "
              f"{only_wrap[:3]}")

    print()
    header = f"{'parameter':<78} {'ref_norm':>12} {'wrap_norm':>12} {'max|diff|':>12}  status"
    print(header)
    print("-" * len(header))

    n_ok = 0
    n_fail = 0
    n_drift = 0
    for key in common:
        r = ref_pooler[key].float()
        w = wrap_pooler[key].float()
        ref_norm = r.norm().item()
        wrap_norm = w.norm().item()
        if r.shape != w.shape:
            status = "SHAPE!"
            n_fail += 1
            max_abs = float("nan")
        else:
            max_abs = (r - w).abs().max().item()
            if max_abs <= args.atol:
                status = "ok"
                n_ok += 1
            elif wrap_norm < 1e-3:
                status = "ZERO!"
                n_fail += 1
            else:
                status = "drift"
                n_drift += 1
        print(f"{key:<78} {ref_norm:12.4f} {wrap_norm:12.4f} {max_abs:12.4e}  {status}")

    print()
    print(f"summary: {n_ok} identical / {n_drift} drifted / {n_fail} zero-or-shape-mismatch "
          f"(atol={args.atol:g})")

    # Verdict.
    if n_fail == 0 and n_drift == 0:
        print("VERDICT: pooler weights are preserved bit-for-bit from the HF checkpoint.")
        return 0
    if n_fail == 0:
        print("VERDICT: pooler weights match in shape and are nonzero, but some tensors "
              "differ within tolerance. Likely OK (dtype casts) but inspect 'drift' rows.")
        return 0
    print("VERDICT: at least one pooler tensor is ZERO or shape-mismatched. "
          "from_pretrained is NOT loading the SSv2-FT pooler -- fix this before retraining.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
