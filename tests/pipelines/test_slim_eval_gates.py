"""Tests for the Phase 4.2 slim-eval gates in :mod:`pipelines.train`.

The two knobs added to ``configs/train/default.yaml`` are:

* ``training.eval_every_n_epochs`` -- skip val on epochs whose index
  ``(epoch + 1)`` is not a multiple of ``N``. The last epoch always
  evaluates so the recorded best checkpoint is never stale.
* ``training.eval_ema`` -- when ``False``, do not evaluate the EMA
  snapshot at the end of each (selected) epoch, even if EMA is enabled.

Both default to the legacy behaviour (``eval_every_n_epochs=1`` and
``eval_ema=True``) so all earlier experiments are byte-for-byte
unchanged. These tests pin the new behaviour without booting any GPU
backbone -- they reuse the small synthetic fixture from
``test_smoke_end_to_end.py``.
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
    """Build the same train/val/test layout as the end-to-end smoke test."""
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
    ema_enabled: bool,
    eval_ema: bool,
) -> OmegaConf:
    """Compose a minimal Hydra-like cfg that exercises the slim-eval gates."""
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
            "training": {
                "batch_size": 2,
                "lr": 0.001,
                "epochs": int(epochs),
                "num_workers": 0,
                "checkpoint_path": str(checkpoint_path),
                "device": "cpu",
                "ema_enabled": ema_enabled,
                "ema_decay": 0.999,
                "eval_every_n_epochs": int(eval_every_n_epochs),
                "eval_ema": bool(eval_ema),
            },
            "track": {"name": "a", "description": "slim-eval-gates"},
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


def test_eval_every_n_epochs_skips_intermediate_evals(
    tiny_dataset: Path,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """With ``eval_every_n_epochs=2`` and ``epochs=3``, only epochs 2 and 3 evaluate.

    The first epoch must print a "val skipped" message; the second and the
    third (last) must both run the full val loop. Last-epoch eval is
    unconditional so the recorded best is never stale.
    """
    checkpoint_path = tmp_path / "best_model.pt"
    cfg = _make_cfg(
        tiny_dataset,
        checkpoint_path,
        epochs=3,
        eval_every_n_epochs=2,
        ema_enabled=False,
        eval_ema=True,
    )
    _run_train(cfg)
    out, _ = capfd.readouterr()

    assert "Epoch 1/3" in out and "val skipped" in out, (
        "Epoch 1 should be skipped under eval_every_n_epochs=2:\n" + out
    )
    assert (
        "Epoch 2/3 |" in out and "val loss" in out.split("Epoch 2/3 |", 1)[1].split("Epoch 3/3")[0]
    ), "Epoch 2 should run a full val pass:\n" + out
    assert "Epoch 3/3 |" in out and "val loss" in out.split("Epoch 3/3 |", 1)[1], (
        "Epoch 3 (final) should always evaluate:\n" + out
    )
    assert checkpoint_path.is_file()


def test_eval_ema_off_skips_ema_pass_but_keeps_live(
    tiny_dataset: Path,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """``eval_ema=False`` runs the live val pass but skips the EMA val pass.

    With EMA enabled and ``eval_ema=False``, the printed line for the
    epoch must report the live val numbers and the "ema eval skipped"
    suffix, and must NOT contain "ema val top1" (which is only printed
    when both EMA and ``eval_ema`` are on).
    """
    checkpoint_path = tmp_path / "best_model.pt"
    cfg = _make_cfg(
        tiny_dataset,
        checkpoint_path,
        epochs=1,
        eval_every_n_epochs=1,
        ema_enabled=True,
        eval_ema=False,
    )
    _run_train(cfg)
    out, _ = capfd.readouterr()

    assert "Epoch 1/1" in out
    assert "val loss" in out, "Live val pass must still run:\n" + out
    assert "ema eval skipped" in out, "eval_ema=False should annotate the skip:\n" + out
    assert "ema val top1" not in out, "eval_ema=False must not print the EMA val numbers:\n" + out
    assert checkpoint_path.is_file()


def test_eval_ema_on_runs_both_passes(
    tiny_dataset: Path,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Legacy default (``eval_ema=True``) keeps both val + EMA val passes.

    Sanity-check: when EMA is enabled and ``eval_ema=True`` we get the
    "ema holdout top1" line, i.e. nothing changed for callers who never
    touch the new knob.
    """
    checkpoint_path = tmp_path / "best_model.pt"
    cfg = _make_cfg(
        tiny_dataset,
        checkpoint_path,
        epochs=1,
        eval_every_n_epochs=1,
        ema_enabled=True,
        eval_ema=True,
    )
    _run_train(cfg)
    out, _ = capfd.readouterr()

    assert "Epoch 1/1" in out
    assert "val loss" in out
    assert "ema holdout top1" in out, "Default eval_ema=True must run the EMA pass:\n" + out
    assert checkpoint_path.is_file()
