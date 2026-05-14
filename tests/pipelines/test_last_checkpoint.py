"""Tests for the per-epoch resume-safe checkpoint added in Phase 4.3.

The trainer historically only wrote ``best_model.pt`` on val_top1
improvements -- a crash on epoch 12 of a 15-epoch run could lose all
the work since the last val improvement (often epoch 3-5 for our
frozen-encoder probes). ``training.save_last_checkpoint`` writes a
companion ``<best>.last.pt`` at the end of *every* epoch with the live
model state, optimizer/scheduler/scaler state, and ``best_top1`` so
the next launch can pass ``training.resume_from=<best>.last.pt`` and
pick up at the next epoch boundary.

These tests pin both pieces of the contract:

1. ``last.pt`` is written for every epoch (eval or skip-eval).
2. The saved payload contains the optimizer state and the running
   ``best_top1``, so resume restores a meaningful state.
3. The ``save_last_checkpoint=false`` opt-out preserves the legacy
   behaviour (no ``last.pt`` on disk).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from PIL import Image

pytestmark = pytest.mark.slow


def _write_video(video_dir: Path, num_frames: int, color: tuple[int, int, int]) -> None:
    """Create ``num_frames`` solid-colour 32x32 JPEG frames in ``video_dir``."""
    video_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_frames):
        Image.new("RGB", (32, 32), color).save(
            video_dir / f"frame_{i:03d}.jpg", format="JPEG", quality=70
        )


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """Build train/val/test layout reused by the slim-eval and last-ckpt tests."""
    root = tmp_path / "data"
    layout = {
        "train": [
            ("000_ClassA", "video_1", (200, 50, 50)),
            ("000_ClassA", "video_2", (200, 60, 60)),
            ("001_ClassB", "video_1", (50, 200, 50)),
            ("001_ClassB", "video_2", (60, 200, 60)),
        ],
        "val": [
            ("000_ClassA", "video_3", (200, 70, 70)),
            ("001_ClassB", "video_3", (70, 200, 70)),
        ],
        "test": [
            ("video_test_1", (100, 100, 200)),
        ],
    }
    for split in ("train", "val"):
        split_root = root / split
        for class_name, video_name, color in layout[split]:  # type: ignore[misc]
            _write_video(split_root / class_name / video_name, num_frames=4, color=color)
    test_root = root / "test"
    for video_name, color in layout["test"]:  # type: ignore[misc]
        _write_video(test_root / video_name, num_frames=4, color=color)
    return root


def _make_cfg(
    data_root: Path,
    checkpoint_path: Path,
    *,
    epochs: int,
    eval_every_n_epochs: int,
    save_last_checkpoint: bool,
    last_checkpoint_path: str | None = None,
) -> OmegaConf:
    """Minimal Hydra-like cfg that exercises the last-checkpoint logic."""
    training: dict[str, object] = {
        "batch_size": 2,
        "lr": 0.001,
        "epochs": int(epochs),
        "num_workers": 0,
        "checkpoint_path": str(checkpoint_path),
        "device": "cpu",
        "eval_every_n_epochs": int(eval_every_n_epochs),
        "eval_ema": True,
        "save_last_checkpoint": bool(save_last_checkpoint),
        "last_checkpoint_path": last_checkpoint_path,
        # Cosine + warmup + scaler stay off so the saved extras stay simple.
        "scheduler_cosine": False,
        "warmup_epochs": 0,
        "amp": False,
    }
    return OmegaConf.create(
        {
            "seed": 0,
            "num_classes": 2,
            "dataset": {
                "root": str(data_root),
                "train_dir": str(data_root / "train"),
                "val_dir": str(data_root / "val"),
                "test_dir": str(data_root / "test"),
                "augmented_dirs": [],
                "test_manifest": None,
                "submission_output": str(checkpoint_path.parent / "sub.csv"),
                "num_frames": 4,
                "val_ratio": 0.5,
                "seed": 0,
                "max_samples": None,
                "image_size": 64,
            },
            "model": {
                "name": "cnn_baseline",
                "pretrained": False,
                "num_classes": 2,
            },
            "training": training,
            "track": {"name": "a", "description": "last-checkpoint"},
            "experiment": {},
        }
    )


def _run_train(cfg: OmegaConf) -> None:
    """Run the train pipeline preserving cwd (the pipeline does ``os.chdir``)."""
    from smth2smth.pipelines.train import run as train_run

    cwd_before = os.getcwd()
    try:
        train_run(cfg)
    finally:
        os.chdir(cwd_before)


def test_last_checkpoint_written_every_epoch_including_skip_eval(
    tiny_dataset: Path,
    tmp_path: Path,
) -> None:
    """``last.pt`` exists after a run even when eval is skipped on epoch 1.

    With ``eval_every_n_epochs=2`` over 3 epochs, epoch 1 is a skip-eval
    epoch and epochs 2 + 3 evaluate. The last.pt file must still be
    refreshed on every epoch -- after the full run it should reflect
    the last completed epoch (3) regardless of which path the trainer
    took on each iteration.
    """
    from smth2smth.shared.io.checkpoints import load_checkpoint

    checkpoint_path = tmp_path / "best_model.pt"
    cfg = _make_cfg(
        tiny_dataset,
        checkpoint_path,
        epochs=3,
        eval_every_n_epochs=2,
        save_last_checkpoint=True,
    )
    _run_train(cfg)

    expected_last = checkpoint_path.with_name(
        checkpoint_path.stem + ".last" + checkpoint_path.suffix
    )
    assert expected_last.is_file(), f"Last checkpoint not written at {expected_last}"

    payload = load_checkpoint(expected_last, map_location="cpu")
    extra = payload["extra"]
    assert int(extra["epoch"]) == 3, (
        f"Last checkpoint should record epoch 3 (final), got {extra.get('epoch')!r}"
    )
    assert extra.get("checkpoint_kind") == "last"
    assert "optimizer_state_dict" in extra, "optimizer state must be persisted"
    assert "val_top1" in extra, "running best_top1 must be persisted for resume"


def test_save_last_checkpoint_off_preserves_legacy_behavior(
    tiny_dataset: Path,
    tmp_path: Path,
) -> None:
    """When the opt-out is set, no ``last.pt`` is written and only ``best`` exists."""
    checkpoint_path = tmp_path / "best_model.pt"
    cfg = _make_cfg(
        tiny_dataset,
        checkpoint_path,
        epochs=2,
        eval_every_n_epochs=1,
        save_last_checkpoint=False,
    )
    _run_train(cfg)

    assert checkpoint_path.is_file(), "best_model.pt must still be produced"
    derived_last = checkpoint_path.with_name(
        checkpoint_path.stem + ".last" + checkpoint_path.suffix
    )
    assert not derived_last.exists(), (
        "save_last_checkpoint=False must not write the derived last.pt"
    )


def test_last_checkpoint_path_override_is_honoured(
    tiny_dataset: Path,
    tmp_path: Path,
) -> None:
    """An explicit ``last_checkpoint_path`` is written instead of the derived one."""
    checkpoint_path = tmp_path / "best_model.pt"
    explicit_last = tmp_path / "elsewhere" / "snap.pt"
    cfg = _make_cfg(
        tiny_dataset,
        checkpoint_path,
        epochs=1,
        eval_every_n_epochs=1,
        save_last_checkpoint=True,
        last_checkpoint_path=str(explicit_last),
    )
    _run_train(cfg)

    assert explicit_last.is_file(), f"Override path was not written: {explicit_last}"
    derived_last = checkpoint_path.with_name(
        checkpoint_path.stem + ".last" + checkpoint_path.suffix
    )
    assert not derived_last.exists(), (
        "Override must take precedence over the derived ``<best>.last.pt`` path"
    )
