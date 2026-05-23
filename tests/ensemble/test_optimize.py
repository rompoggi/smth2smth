"""Tests for learned-weight ensemble optimization."""

from __future__ import annotations

import numpy as np

from smth2smth.ensemble.optimize import equal_weights_result, optimize_mix_weights


def test_mix_beats_or_matches_equal_on_synthetic() -> None:
    """With one member much better, Mix should put weight on it."""
    rng = np.random.default_rng(0)
    n, c = 200, 33
    labels = rng.integers(0, c, size=n)
    good = rng.standard_normal((n, c))
    good[np.arange(n), labels] += 3.0
    bad = rng.standard_normal((n, c))
    bad[np.arange(n), labels] += 0.5
    mix = optimize_mix_weights([good, bad], labels)
    eq = equal_weights_result([good, bad], labels)
    assert mix.top1 >= eq.top1 - 0.5
    assert mix.weights[0] > mix.weights[1]
