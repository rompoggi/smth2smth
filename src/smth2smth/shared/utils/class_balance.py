"""Class-imbalance helpers for the SSv2 Track-A pipeline.

The Track-A training set has a ~20x ratio between the smallest and largest
class (162 vs 3170 samples). Two complementary fixes are wired in here:

1. ``compute_sample_weights`` produces per-sample weights for use with
   :class:`torch.utils.data.WeightedRandomSampler`. Both a ``sqrt_inverse``
   policy (1 / sqrt(n_c) -- recommended; tames the imbalance without
   destroying head-class accuracy) and a pure ``inverse`` policy (1 / n_c)
   are supported.

2. ``compute_class_weights`` returns a vector of per-class loss weights
   suitable for :class:`torch.nn.CrossEntropyLoss(weight=...)` and for the
   soft-target CE path used by the video-mixing augmentations. We support:

   * ``inverse``    -- 1 / n_c, normalised to mean 1.
   * ``sqrt_inverse`` -- 1 / sqrt(n_c), normalised to mean 1.
   * ``cb`` (Cui et al. 2019, class-balanced loss): w_c = (1 - beta) /
     (1 - beta^n_c), normalised to mean 1. ``beta = 0.999`` is the paper's
     recommended default; ``beta -> 0`` recovers uniform weights and
     ``beta -> 1`` recovers ``inverse``.

Never-trained classes (no samples in ``train_samples``) receive weight 0 so
they cannot win a softmax-weighted argmax at training time. The existing
:mod:`smth2smth.pipelines.submit` untrained-mask still applies at inference
time.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import torch

VideoSampleLike = tuple[object, int]


_VALID_SAMPLER_POLICIES = frozenset({"none", "inverse", "sqrt_inverse"})
_VALID_LOSS_POLICIES = frozenset({"none", "inverse", "sqrt_inverse", "cb"})


def class_counts(samples: Iterable[VideoSampleLike], num_classes: int) -> list[int]:
    """Count training samples per class.

    Args:
        samples: Iterable of ``(video_dir, label)`` pairs (label is an int).
        num_classes: Width of the classifier head. Classes with index in
            ``[0, num_classes)`` that never appear get count 0.

    Returns:
        List of length ``num_classes``; entry ``c`` is the number of samples
        with ``label == c``.
    """
    if num_classes <= 0:
        raise ValueError(f"num_classes must be > 0, got {num_classes}.")
    counts = [0] * int(num_classes)
    counter: Counter[int] = Counter(int(label) for _, label in samples)
    for c, n in counter.items():
        if 0 <= c < num_classes:
            counts[c] = int(n)
    return counts


def compute_sample_weights(
    samples: list[VideoSampleLike],
    num_classes: int,
    policy: str = "sqrt_inverse",
) -> torch.Tensor:
    """Per-sample weights for :class:`WeightedRandomSampler`.

    The same per-class factor is broadcast to every sample of that class.

    Args:
        samples: ``(video_dir, label)`` pairs, one entry per training sample.
        num_classes: Width of the classifier head.
        policy: One of ``{"none", "inverse", "sqrt_inverse"}``. ``"none"`` is
            shorthand for uniform 1.0 weights; useful so callers can pass the
            config knob through unchanged.

    Returns:
        1-D float32 ``Tensor`` of length ``len(samples)``.

    Raises:
        ValueError: On unknown ``policy``.
    """
    if policy not in _VALID_SAMPLER_POLICIES:
        raise ValueError(
            f"policy must be one of {sorted(_VALID_SAMPLER_POLICIES)}, got {policy!r}"
        )
    n_per_class = class_counts(samples, num_classes=num_classes)
    if policy == "none":
        return torch.ones(len(samples), dtype=torch.float32)
    factors = _class_factors(n_per_class, policy=policy)
    weights = torch.tensor(
        [factors[int(label)] for _, label in samples], dtype=torch.float32
    )
    return weights


def compute_class_weights(
    samples: list[VideoSampleLike],
    num_classes: int,
    policy: str = "cb",
    beta: float = 0.999,
) -> torch.Tensor | None:
    """Per-class loss weights for cross-entropy.

    Args:
        samples: ``(video_dir, label)`` pairs.
        num_classes: Width of the classifier head.
        policy: One of ``{"none", "inverse", "sqrt_inverse", "cb"}``. Returns
            ``None`` when ``policy == "none"`` so callers can skip the
            ``weight=...`` keyword entirely.
        beta: Only used when ``policy == "cb"``. Cui et al. 2019 recommend
            0.999; values near 1.0 emphasise rare classes, values near 0.0
            recover uniform weights.

    Returns:
        1-D float32 ``Tensor`` of length ``num_classes`` normalised so that
        ``mean(weights[trained_classes]) == 1.0``. Never-trained classes
        receive weight 0. Returns ``None`` if ``policy == "none"``.

    Raises:
        ValueError: On unknown ``policy`` or invalid ``beta``.
    """
    if policy not in _VALID_LOSS_POLICIES:
        raise ValueError(
            f"policy must be one of {sorted(_VALID_LOSS_POLICIES)}, got {policy!r}"
        )
    if policy == "none":
        return None
    if not 0.0 < beta < 1.0:
        raise ValueError(f"beta must be in (0, 1), got {beta}.")
    n_per_class = class_counts(samples, num_classes=num_classes)
    factors = _class_factors(n_per_class, policy=policy, beta=beta)
    weights = torch.tensor(factors, dtype=torch.float32)
    trained = weights > 0
    if trained.any():
        weights = weights / float(weights[trained].mean().item())
    return weights


def _class_factors(
    n_per_class: list[int],
    policy: str,
    beta: float = 0.999,
) -> list[float]:
    """Compute per-class scalar factors before sample-weight broadcasting."""
    factors: list[float] = []
    for n in n_per_class:
        if n <= 0:
            factors.append(0.0)
            continue
        if policy == "inverse":
            factors.append(1.0 / float(n))
        elif policy == "sqrt_inverse":
            factors.append(1.0 / (float(n) ** 0.5))
        elif policy == "cb":
            effective_num = 1.0 - (beta ** float(n))
            factors.append((1.0 - beta) / max(1e-12, effective_num))
        else:
            raise ValueError(f"Unsupported policy {policy!r}")
    return factors


__all__ = [
    "class_counts",
    "compute_class_weights",
    "compute_sample_weights",
]
