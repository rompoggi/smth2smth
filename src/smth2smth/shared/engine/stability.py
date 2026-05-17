"""Per-step stability logging for HC/mHC ablation runs.

The HC/mHC stability claim is a **variance claim**: gradient-norm oscillation
and loss-curve smoothness, not (only) end-of-training accuracy. That requires
per-step (not per-epoch) logging.

This module dumps a CSV with one row per optimizer step:

    step, epoch, lr, loss, grad_norm_global, grad_norm_block_0, ...,
    hc_M_max_abs_block_0, ..., hc_M_off_diag_mass_block_0, ...,
    mhc_sk_doubly_stochastic_dev_block_0, ...

Hooked from ``train_one_epoch`` between ``backward`` and ``optimizer.step()``,
so the recorded grad norms are pre-clip (the experiment recipe does not clip).
For fp16 AMP the GradScaler is unscaled in-place before measurement; for bf16
the scaler is a no-op so this is a free pass-through.

The per-block grad norm covers every encoder transformer block (Pre-Norm or
HC/mHC). The HC drift metrics (``M_max_abs``, off-diagonal mass) are emitted
for every :class:`HCRouter` found in the model — i.e. 24 entries per ViT-B
block stack (12 blocks × 2 sublayers). For mHC the projected SK matrix's
deviation from doubly-stochastic is also emitted (max(|row_sum - 1|, |col_sum - 1|)).

All scalars are written as Python floats. The CSV is opened in append mode so
resumed training keeps appending without truncating the header.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn as nn

from smth2smth.shared.models.video_mae import HCRouter, sinkhorn_knopp


class StabilityLogger:
    """Per-step CSV logger for grad norms and HC drift metrics."""

    def __init__(
        self,
        csv_path: Path,
        *,
        model: nn.Module,
        log_every: int = 1,
        include_per_block: bool = True,
        include_hc_drift: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_every = max(1, int(log_every))
        self.include_per_block = bool(include_per_block)
        self.include_hc_drift = bool(include_hc_drift)
        self._step = 0
        self._epoch = 0

        # Cache references to the named encoder blocks and HC routers, in stable
        # order. The trainer keeps the same model object across epochs, so the
        # cache stays valid for the full run.
        self._blocks: list[tuple[str, nn.Module]] = []
        self._hc_routers: list[tuple[str, HCRouter]] = []
        for name, module in model.named_modules():
            if name.startswith("encoder.blocks.") and name.count(".") == 2:
                # Match exactly "encoder.blocks.<i>" — not descendants.
                self._blocks.append((name, module))
            if isinstance(module, HCRouter):
                self._hc_routers.append((name, module))

        # Persist any HC variant for later mHC-only header decisions.
        self._has_mhc = any(r.is_mhc for _, r in self._hc_routers)

        self._fh = self.csv_path.open("a", newline="")
        self._writer = csv.writer(self._fh)
        if self.csv_path.stat().st_size == 0:
            self._writer.writerow(self._header())
            self._fh.flush()

    def _header(self) -> list[str]:
        cols = ["step", "epoch", "lr", "loss", "grad_norm_global"]
        if self.include_per_block:
            for name, _ in self._blocks:
                cols.append(f"grad_norm_{name}")
        if self.include_hc_drift:
            for name, _ in self._hc_routers:
                cols.append(f"M_max_abs_{name}")
                cols.append(f"M_off_diag_mass_{name}")
            if self._has_mhc:
                for name, router in self._hc_routers:
                    if router.is_mhc:
                        cols.append(f"sk_dev_{name}")
        return cols

    def set_epoch(self, epoch_zero_indexed: int) -> None:
        """Record the current epoch so each row carries it."""
        self._epoch = int(epoch_zero_indexed)

    @torch.no_grad()
    def __call__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler | None,
        loss_value: float,
    ) -> None:
        """Log one step. Call after ``backward``, before ``optimizer.step()``.

        For fp16 AMP this unscales the optimizer's gradients in place so the
        norms are in real units; ``optimizer.step()`` will detect this and
        avoid double-unscale.
        """
        self._step += 1
        if self._step % self.log_every != 0:
            return

        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(optimizer)

        # Per-block gradient norms (encoder.blocks.<i> sum of squares).
        per_block_norms: list[float] = []
        if self.include_per_block:
            for _, block in self._blocks:
                sq = 0.0
                for p in block.parameters():
                    if p.grad is not None:
                        sq += float(p.grad.detach().pow(2).sum().item())
                per_block_norms.append(sq**0.5)

        # Global L2 norm across *all* trainable params.
        global_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                global_sq += float(p.grad.detach().pow(2).sum().item())
        global_norm = global_sq**0.5

        # HC drift metrics.
        m_max_abs: list[float] = []
        m_off_mass: list[float] = []
        sk_devs: list[float] = []
        if self.include_hc_drift:
            for _, router in self._hc_routers:
                m = router.mixing_matrix(torch.float32).detach()
                n = m.shape[0]
                m_max_abs.append(float(m.abs().max().item()))
                off_mass = float(
                    (m - torch.eye(n, device=m.device, dtype=m.dtype))
                    .pow(2).sum().sqrt().item()
                )
                m_off_mass.append(off_mass)
                if router.is_mhc:
                    # mHC: how far from doubly-stochastic is the SK projection?
                    row_dev = float((m.sum(dim=1) - 1.0).abs().max().item())
                    col_dev = float((m.sum(dim=0) - 1.0).abs().max().item())
                    sk_devs.append(max(row_dev, col_dev))

        # Use the first param group's LR as a representative LR.
        lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0

        row: list[float | int] = [
            self._step, self._epoch, lr, float(loss_value), global_norm,
        ]
        if self.include_per_block:
            row.extend(per_block_norms)
        if self.include_hc_drift:
            for ma, om in zip(m_max_abs, m_off_mass):
                row.append(ma)
                row.append(om)
            if self._has_mhc:
                row.extend(sk_devs)
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
