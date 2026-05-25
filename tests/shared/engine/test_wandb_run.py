"""Tests for optional W&B logging."""

from __future__ import annotations

from omegaconf import OmegaConf

from smth2smth.shared.engine.wandb_run import WandbTracker


def test_wandb_tracker_disabled_by_default() -> None:
    cfg = OmegaConf.create({"training": {"wandb_enabled": False}})
    tracker = WandbTracker(cfg)
    assert not tracker.enabled
    tracker.log({"train/loss": 1.0}, step=1)
    tracker.finish()
