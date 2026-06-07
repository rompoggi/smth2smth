"""Ensemble combiners for cached member logits (v2 grid)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from smth2smth.ensemble.optimize import (
    EnsembleResult,
    combine_logits,
    metrics_from_logits,
    optimize_mix_weights,
)


def sanitize_logits(arr: np.ndarray) -> np.ndarray:
    """Replace masked ``-inf`` class slots with a large negative finite value."""
    out = np.asarray(arr, dtype=np.float64)
    if not np.isfinite(out).all():
        out = out.copy()
        out[np.isneginf(out)] = -1e9
        out[np.isposinf(out)] = 1e9
        out[np.isnan(out)] = 0.0
    return out


def _fit_lsg(
    x_fit: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    *,
    c_reg: float = 1.0,
) -> np.ndarray:
    """L2 linear stacker on one-hot targets; keeps all ``n_classes`` outputs."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import label_binarize

    y = np.asarray(labels, dtype=np.int64)
    y_onehot = label_binarize(y, classes=np.arange(n_classes))
    if y_onehot.shape[1] < n_classes:
        pad = np.zeros((y_onehot.shape[0], n_classes - y_onehot.shape[1]))
        y_onehot = np.concatenate([y_onehot, pad], axis=1)
    alpha = 1.0 / max(float(c_reg), 1e-6)
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(x_fit, y_onehot)
    return reg.coef_.astype(np.float64)


def _predict_lsg(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return x @ coef.T


def logits_to_probs(arr: np.ndarray) -> np.ndarray:
    """Row-wise softmax probabilities from logits."""
    x = torch.from_numpy(sanitize_logits(arr).astype(np.float32))
    return F.softmax(x, dim=1).cpu().numpy()


def softmax_mean_result(
    prob_arrays: list[np.ndarray],
    labels: np.ndarray,
    *,
    name: str,
) -> EnsembleResult:
    """Average member probability vectors, then argmax (softmax-mean)."""
    n = len(prob_arrays)
    w = np.full(n, 1.0 / n, dtype=np.float64)
    combined = np.zeros_like(prob_arrays[0], dtype=np.float64)
    for arr in prob_arrays:
        combined += np.asarray(arr, dtype=np.float64)
    combined /= float(n)
    # Treat as logits via log for CE metric (numerically stable)
    logits = np.log(np.clip(combined, 1e-12, 1.0))
    top1, top5, ce = metrics_from_logits(logits, labels)
    return EnsembleResult(name=name, weights=w, top1=top1, top5=top5, cross_entropy=ce)


def majority_vote_result(
    logit_arrays: list[np.ndarray],
    labels: np.ndarray,
    *,
    name: str,
) -> EnsembleResult:
    """Per-sample majority vote over member argmax predictions."""
    preds = np.stack([np.argmax(arr, axis=1) for arr in logit_arrays], axis=1)
    n_models = preds.shape[1]
    combined_preds = np.zeros(preds.shape[0], dtype=np.int64)
    for i in range(preds.shape[0]):
        votes = preds[i]
        counts = np.bincount(votes, minlength=int(logit_arrays[0].shape[1]))
        combined_preds[i] = int(np.argmax(counts))
    y = np.asarray(labels, dtype=np.int64)
    top1 = float((combined_preds == y).mean() * 100.0)
    top5 = top1  # vote does not define top-5; duplicate for table
    w = np.full(n_models, 1.0 / n_models, dtype=np.float64)
    return EnsembleResult(name=name, weights=w, top1=top1, top5=top5, cross_entropy=0.0)


def optimize_cws_weights(
    logit_arrays: list[np.ndarray],
    labels: np.ndarray,
    *,
    name: str,
) -> EnsembleResult:
    """Class-dependent weights: per class c, theta[:,c] on the simplex."""
    from scipy.optimize import minimize

    m_models = len(logit_arrays)
    stacked = np.stack(logit_arrays, axis=0).astype(np.float64)  # (M, N, C)
    labels = np.asarray(labels, dtype=np.int64)
    n_classes = stacked.shape[2]

    def loss_flat(flat_w: np.ndarray) -> float:
        w = flat_w.reshape(m_models, n_classes)
        w = np.maximum(w, 0.0)
        w = w / np.maximum(w.sum(axis=0, keepdims=True), 1e-12)
        combined = np.einsum("mnc,mc->nc", stacked, w)
        x = torch.from_numpy(combined.astype(np.float32))
        y = torch.from_numpy(labels)
        return float(F.cross_entropy(x, y, reduction="mean").item())

    x0 = np.full(m_models * n_classes, 1.0 / m_models, dtype=np.float64)
    res = minimize(
        loss_flat,
        x0,
        method="Nelder-Mead",
        options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8},
    )
    w = np.maximum(res.x, 0.0).reshape(m_models, n_classes)
    w = w / np.maximum(w.sum(axis=0, keepdims=True), 1e-12)
    combined = np.einsum("mnc,mc->nc", stacked, w)
    top1, top5, ce = metrics_from_logits(combined, labels)
    return EnsembleResult(name=name, weights=w, top1=top1, top5=top5, cross_entropy=ce)


def optimize_lsg_weights(
    logit_arrays: list[np.ndarray],
    labels: np.ndarray,
    *,
    name: str,
    c_reg: float = 1.0,
) -> EnsembleResult:
    """Linear stacked generalization via L2 ridge on one-hot labels."""
    x = np.concatenate(logit_arrays, axis=1).astype(np.float64)
    y = np.asarray(labels, dtype=np.int64)
    n_classes = int(logit_arrays[0].shape[1])
    coef = _fit_lsg(x, y, n_classes, c_reg=c_reg)
    combined = _predict_lsg(x, coef)
    top1, top5, ce = metrics_from_logits(combined, labels)
    return EnsembleResult(name=name, weights=coef, top1=top1, top5=top5, cross_entropy=ce)


def run_combiner(
    combiner: str,
    fit_arrays: list[np.ndarray],
    eval_arrays: list[np.ndarray],
    eval_prob_arrays: list[np.ndarray] | None,
    labels: np.ndarray,
    *,
    name: str,
    fit_labels: np.ndarray | None = None,
) -> EnsembleResult:
    """Fit on ``fit_*`` (labelled by ``fit_labels``) when learned; always score on ``eval_*`` with ``labels``.

    ``fit_labels`` defaults to ``labels`` when ``None`` (legacy fit==eval behaviour).
    """
    fit_arrays = [sanitize_logits(a) for a in fit_arrays]
    eval_arrays = [sanitize_logits(a) for a in eval_arrays]
    fit_labels = labels if fit_labels is None else fit_labels
    combiner = combiner.lower()
    if combiner == "mean":
        w = np.full(len(fit_arrays), 1.0 / len(fit_arrays))
        return _score_logits(eval_arrays, labels, w, name=name)
    if combiner == "softmax":
        probs = eval_prob_arrays
        if probs is None:
            probs = [logits_to_probs(a) for a in eval_arrays]
        return softmax_mean_result(probs, labels, name=name)
    if combiner == "vote":
        return majority_vote_result(eval_arrays, labels, name=name)
    if combiner == "ws":
        r_fit = optimize_mix_weights(fit_arrays, fit_labels, name=name)
        return _score_logits(eval_arrays, labels, r_fit.weights, name=name)
    if combiner == "cws":
        r_fit = optimize_cws_weights(fit_arrays, fit_labels, name=name)
        w = r_fit.weights
        if w.ndim == 2:
            stacked = np.stack(eval_arrays, axis=0)
            combined = np.einsum("mnc,mc->nc", stacked, w)
            top1, top5, ce = metrics_from_logits(combined, labels)
            return EnsembleResult(name=name, weights=w, top1=top1, top5=top5, cross_entropy=ce)
        return _score_logits(eval_arrays, labels, w, name=name)
    if combiner == "lsg":
        y = np.asarray(fit_labels, dtype=np.int64)
        n_classes = int(fit_arrays[0].shape[1])
        x_fit = np.concatenate(fit_arrays, axis=1).astype(np.float64)
        x_eval = np.concatenate(eval_arrays, axis=1).astype(np.float64)
        coef = _fit_lsg(x_fit, y, n_classes, c_reg=1.0)
        combined = _predict_lsg(x_eval, coef)
        top1, top5, ce = metrics_from_logits(combined, labels)
        return EnsembleResult(name=name, weights=coef, top1=top1, top5=top5, cross_entropy=ce)
    raise ValueError(f"Unknown combiner: {combiner}")


def _score_logits(
    arrays: list[np.ndarray],
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    name: str,
) -> EnsembleResult:
    if weights.ndim == 1:
        combined = combine_logits(arrays, weights)
    else:
        stacked = np.stack(arrays, axis=0)
        combined = np.einsum("mnc,mc->nc", stacked, weights)
    top1, top5, ce = metrics_from_logits(combined, labels)
    return EnsembleResult(name=name, weights=weights, top1=top1, top5=top5, cross_entropy=ce)
