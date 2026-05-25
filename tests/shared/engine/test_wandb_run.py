"""Tests for optional W&B logging."""

from __future__ import annotations

from omegaconf import OmegaConf

from types import SimpleNamespace

from smth2smth.shared.engine.wandb_run import WandbTracker
from smth2smth.shared.utils.wandb_run import log_epoch_summary


def test_wandb_tracker_disabled_by_default() -> None:
    cfg = OmegaConf.create({"training": {"wandb_enabled": False}})
    tracker = WandbTracker(cfg)
    assert not tracker.enabled
    tracker.log({"train/loss": 1.0}, step=1)
    tracker.finish()


def _epoch_stats(top1: float) -> SimpleNamespace:
    return SimpleNamespace(loss=1.0 - top1, top1=top1, top5=min(1.0, top1 + 0.1))


class _CapturingTracker:
    enabled = True

    def __init__(self) -> None:
        self.last: dict[str, float] | None = None

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.last = metrics


def test_log_epoch_summary_holdout_and_honest_metrics() -> None:
    tracker = _CapturingTracker()
    train = _epoch_stats(0.5)
    holdout = _epoch_stats(0.2)
    honest = _epoch_stats(0.35)
    log_epoch_summary(
        tracker,
        epoch_one_indexed=3,
        steps_per_epoch=100,
        train_stats=train,
        val_holdout_stats=holdout,
        val_honest_stats=honest,
        lr=5e-4,
        best_top1=0.25,
        use_val_holdout=True,
    )
    assert tracker.last is not None
    assert tracker.last["val/holdout_top1"] == 0.2
    assert tracker.last["val/honest_top1"] == 0.35
    assert tracker.last["val/top1"] == 0.2


def test_log_epoch_summary_honest_only_aliases() -> None:
    tracker = _CapturingTracker()
    honest = _epoch_stats(0.45)
    log_epoch_summary(
        tracker,
        epoch_one_indexed=1,
        steps_per_epoch=50,
        train_stats=_epoch_stats(0.6),
        val_holdout_stats=honest,
        lr=1e-3,
        best_top1=0.45,
        use_val_holdout=False,
    )
    assert tracker.last is not None
    assert tracker.last["val/top1"] == 0.45
    assert tracker.last["val/honest_top1"] == 0.45
