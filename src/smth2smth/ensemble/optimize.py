"""Mix (learned weights) and Equal combiners on cached logits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EnsembleResult:
    """Metrics and weights from one combiner on holdout logits."""

    name: str
    weights: np.ndarray
    top1: float
    top5: float
    cross_entropy: float


def combine_logits(
    logit_arrays: list[np.ndarray | torch.Tensor],
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted sum of member logits, shape ``(N, C)``."""
    w = np.asarray(weights, dtype=np.float64)
    if len(logit_arrays) != len(w):
        raise ValueError("len(logit_arrays) must match len(weights).")
    out = np.zeros_like(np.asarray(logit_arrays[0], dtype=np.float64))
    for arr, wi in zip(logit_arrays, w, strict=True):
        out = out + wi * np.asarray(arr, dtype=np.float64)
    return out


def metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """Top-1, top-5 (percent), mean cross-entropy."""
    x = torch.from_numpy(logits.astype(np.float32))
    y = torch.from_numpy(labels.astype(np.int64))
    ce = float(F.cross_entropy(x, y, reduction="mean").item())
    top1 = float((x.argmax(dim=1) == y).float().mean().item() * 100.0)
    top5 = float(
        (x.topk(min(5, x.shape[1]), dim=1).indices == y.unsqueeze(1))
        .any(dim=1)
        .float()
        .mean()
        .item()
        * 100.0
    )
    return top1, top5, ce


def optimize_mix_weights(
    logit_arrays: list[np.ndarray],
    labels: np.ndarray,
    *,
    constrained: str = "simplex",
    name: str = "mix",
) -> EnsembleResult:
    """Find scalar weights minimizing holdout cross-entropy.

    Args:
        logit_arrays: List of ``(N, C)`` arrays, one per ensemble member.
        labels: ``(N,)`` integer class indices.
        constrained: ``simplex`` (non-negative, sum to 1), ``nonneg``, or ``unconstrained``.
        name: Label for the result row.

    Returns:
        :class:`EnsembleResult` with optimal weights and holdout metrics.
    """
    from scipy.optimize import minimize

    n_models = len(logit_arrays)
    if n_models == 0:
        raise ValueError("At least one logit array required.")
    stacked = np.stack(logit_arrays, axis=0).astype(np.float64)  # (M, N, C)
    labels = np.asarray(labels, dtype=np.int64)

    def loss_raw(w: np.ndarray) -> float:
        combined = np.tensordot(w, stacked, axes=(0, 0))  # (N, C)
        x = torch.from_numpy(combined.astype(np.float32))
        y = torch.from_numpy(labels)
        return float(F.cross_entropy(x, y, reduction="mean").item())

    x0 = np.full(n_models, 1.0 / n_models, dtype=np.float64)

    if constrained == "simplex":

        def loss(w: np.ndarray) -> float:
            w = np.maximum(w, 0.0)
            s = w.sum()
            if s <= 0:
                return loss_raw(x0)
            return loss_raw(w / s)

        res = minimize(
            loss,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-8},
        )
        w = np.maximum(res.x, 0.0)
        w = w / w.sum()
    elif constrained == "nonneg":
        bounds = [(0.0, None)] * n_models
        res = minimize(loss_raw, x0, method="L-BFGS-B", bounds=bounds)
        w = np.maximum(res.x, 0.0)
        w = w / w.sum() if w.sum() > 0 else x0
    elif constrained == "unconstrained":
        res = minimize(loss_raw, x0, method="BFGS")
        w = res.x
        w = w / w.sum() if np.abs(w.sum()) > 1e-12 else x0
    else:
        raise ValueError(f"Unknown constrained={constrained!r}")

    combined = combine_logits(logit_arrays, w)
    top1, top5, ce = metrics_from_logits(combined, labels)
    return EnsembleResult(name=name, weights=w, top1=top1, top5=top5, cross_entropy=ce)


def equal_weights_result(
    logit_arrays: list[np.ndarray],
    labels: np.ndarray,
    *,
    name: str = "equal",
) -> EnsembleResult:
    """Uniform ``1/M`` weights (no optimization)."""
    n = len(logit_arrays)
    w = np.full(n, 1.0 / n, dtype=np.float64)
    combined = combine_logits(logit_arrays, w)
    top1, top5, ce = metrics_from_logits(combined, labels)
    return EnsembleResult(name=name, weights=w, top1=top1, top5=top5, cross_entropy=ce)


def single_member_result(
    logit_arrays: list[np.ndarray],
    labels: np.ndarray,
    index: int,
    *,
    name: str,
) -> EnsembleResult:
    """Metrics for one member alone."""
    w = np.zeros(len(logit_arrays), dtype=np.float64)
    w[index] = 1.0
    combined = combine_logits(logit_arrays, w)
    top1, top5, ce = metrics_from_logits(combined, labels)
    return EnsembleResult(name=name, weights=w, top1=top1, top5=top5, cross_entropy=ce)
