"""Tests for ``scripts/compare_submissions.py``.

Imports the ``compare`` helper and runs it against synthetic mappings.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_compare_module():
    """Import ``scripts/compare_submissions.py`` as a module for testing."""
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "compare_submissions.py"
    spec = importlib.util.spec_from_file_location("compare_submissions", script_path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["compare_submissions"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture(scope="module")
def compare_mod():
    return _load_compare_module()


class TestCompare:
    def test_full_agreement(self, compare_mod) -> None:
        baseline: dict[str, int] = {"a": 1, "b": 2, "c": 3}
        ours: dict[str, int] = {"a": 1, "b": 2, "c": 3}
        result = compare_mod.compare(baseline, ours)
        assert result.num_videos == 3
        assert result.agreement == 1.0
        assert result.agree_count == 3
        assert result.disagree_count == 0
        assert result.top_disagreement_pairs == []

    def test_partial_disagreement_is_counted_per_pair(self, compare_mod) -> None:
        baseline = {"a": 0, "b": 0, "c": 0, "d": 1, "e": 2}
        ours = {"a": 1, "b": 1, "c": 0, "d": 0, "e": 2}
        result = compare_mod.compare(baseline, ours, top_k=2)
        assert result.num_videos == 5
        assert result.agree_count == 2  # c, e
        assert result.disagree_count == 3
        # Most common disagreement direction: 0 -> 1 (a, b), then 1 -> 0 (d).
        assert result.top_disagreement_pairs[0] == ((0, 1), 2)
        assert ((1, 0), 1) in result.top_disagreement_pairs

    def test_video_name_mismatch_raises(self, compare_mod) -> None:
        with pytest.raises(ValueError):
            compare_mod.compare({"a": 0}, {"b": 0})

    def test_empty_mappings_yield_zero_videos(self, compare_mod) -> None:
        result = compare_mod.compare({}, {})
        assert result.num_videos == 0
        assert result.agreement == 0.0
