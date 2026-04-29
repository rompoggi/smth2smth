"""Tests for ``smth2smth.shared.engine.metrics``."""

from __future__ import annotations

import pytest
import torch

from smth2smth.shared.engine.metrics import accuracy_topk


class TestAccuracyTopK:
    """Top-k accuracy correctness."""

    def test_perfect_predictions_yield_one(self) -> None:
        # Logits whose argmax equals targets: top-1 must be 1.0.
        targets = torch.tensor([0, 1, 2, 3])
        logits = torch.eye(4) * 5
        top1, top5 = accuracy_topk(logits, targets, topk=(1, 5))
        assert float(top1) == pytest.approx(1.0)
        assert float(top5) == pytest.approx(1.0)

    def test_all_wrong_top1_zero(self) -> None:
        # Predict class 0 for every sample, but targets are 1..4 -> top-1 = 0.
        targets = torch.tensor([1, 2, 3, 4])
        logits = torch.zeros(4, 5)
        logits[:, 0] = 10.0
        top1, top5 = accuracy_topk(logits, targets, topk=(1, 5))
        assert float(top1) == pytest.approx(0.0)
        # Top-5 covers all 5 classes, so all matches succeed.
        assert float(top5) == pytest.approx(1.0)

    def test_topk_consistent_with_topk_argsort(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(20, 33)
        targets = torch.randint(0, 33, (20,))
        top1, top5 = accuracy_topk(logits, targets, topk=(1, 5))
        # Manual reference for top-1: argmax matches.
        manual_top1 = (logits.argmax(dim=1) == targets).float().mean()
        assert float(top1) == pytest.approx(float(manual_top1))
        # Top-5 is monotone: >= top-1.
        assert float(top5) >= float(top1) - 1e-6

    @pytest.mark.parametrize(
        "bad_logits",
        [torch.zeros(4), torch.zeros(2, 2, 2)],
    )
    def test_invalid_logits_shape_raises(self, bad_logits: torch.Tensor) -> None:
        with pytest.raises(ValueError):
            accuracy_topk(bad_logits, torch.tensor([0, 0, 0, 0]))

    def test_empty_batch_returns_zeros(self) -> None:
        logits = torch.zeros(0, 3)
        targets = torch.zeros(0, dtype=torch.long)
        top1, top5 = accuracy_topk(logits, targets, topk=(1, 5))
        assert float(top1) == 0.0
        assert float(top5) == 0.0
