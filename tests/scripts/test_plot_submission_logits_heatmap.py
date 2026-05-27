"""Tests for ``scripts/plot_submission_logits_heatmap.py`` (pure helpers)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "plot_submission_logits_heatmap.py"
    spec = importlib.util.spec_from_file_location("plot_submission_logits_heatmap", script_path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["plot_submission_logits_heatmap"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture(scope="module")
def heatmap_mod():
    return _load_module()


class TestLogitsHeatmapHelpers:
    def test_softmax_rows_sum_to_one(self, heatmap_mod) -> None:
        x = np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 5.0]])
        p = heatmap_mod.scale_matrix(x, "softmax")
        np.testing.assert_allclose(p.sum(axis=1), [1.0, 1.0], rtol=1e-6)

    def test_argmax_sort_groups_classes(self, heatmap_mod) -> None:
        mat = np.eye(4, dtype=np.float64)
        order = heatmap_mod.row_sort_order(mat, "argmax")
        assert list(order) == [0, 1, 2, 3]

    def test_subsample_stride(self, heatmap_mod) -> None:
        x = np.arange(10).reshape(10, 1)
        y = heatmap_mod.subsample_rows(x, 3)
        assert y.shape == (4, 1)
        assert y[0, 0] == 0
        assert y[-1, 0] == 9

    def test_drop_class_columns(self, heatmap_mod) -> None:
        x = np.arange(12, dtype=np.float64).reshape(3, 4)
        y = heatmap_mod.drop_class_columns(x, [1])
        assert y.shape == (3, 3)
        np.testing.assert_array_equal(y[:, 0], x[:, 0])
        np.testing.assert_array_equal(y[:, 1], x[:, 2])

    def test_reference_sort_matches_argmax_on_reference(self, heatmap_mod) -> None:
        ref = np.array([[3.0, 1.0, 0.0], [0.0, 2.0, 0.0], [1.0, 0.0, 4.0]])
        ref_scaled = heatmap_mod.scale_matrix(ref, "softmax")
        other = ref_scaled.copy()
        order_ref = heatmap_mod.row_sort_order(ref_scaled, "reference")
        order_direct = heatmap_mod.row_sort_order(ref_scaled, "argmax")
        assert list(order_ref) == list(order_direct)
        aligned, _, _ = heatmap_mod._prepare_matrix(
            other,
            scale="softmax",
            exclude=[],
            sort_rows="reference",
            sort_cols="index",
            reference_raw=ref,
        )
        assert aligned.shape == ref_scaled.shape
