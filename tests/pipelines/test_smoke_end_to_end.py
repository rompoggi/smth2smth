"""End-to-end smoke test for the full train -> evaluate -> submit pipeline.

Uses a tiny synthetic frame dataset (2 classes x 2 videos x 4 frames) so the
whole loop runs in a few seconds on CPU. Exercises:
    * pipelines.train.run with from-scratch ResNet18 (no weight download)
    * pipelines.evaluate.run on the saved checkpoint
    * pipelines.submit.run writing a valid submission CSV

The test marks itself as ``slow`` so contributors can deselect with
``pytest -m 'not slow'`` if desired.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from PIL import Image

pytestmark = pytest.mark.slow


def _write_video(video_dir: Path, num_frames: int, color: tuple[int, int, int]) -> None:
    video_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_frames):
        Image.new("RGB", (32, 32), color).save(
            video_dir / f"frame_{i:03d}.jpg", format="JPEG", quality=70
        )


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """Build train/val/test splits with two classes and a few short videos."""
    root = tmp_path / "data"
    splits = {
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
            ("video_test_2", (110, 110, 210)),
        ],
    }

    for split_name in ("train", "val"):
        split_root = root / split_name
        for class_name, video_name, color in splits[split_name]:  # type: ignore[misc]
            _write_video(split_root / class_name / video_name, num_frames=4, color=color)

    test_root = root / "test"
    for video_name, color in splits["test"]:  # type: ignore[misc]
        _write_video(test_root / video_name, num_frames=4, color=color)

    return root


def _make_cfg(data_root: Path, checkpoint_path: Path, submission_path: Path) -> OmegaConf:
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
                "submission_output": str(submission_path),
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
                "epochs": 1,
                "num_workers": 0,
                "checkpoint_path": str(checkpoint_path),
                "device": "cpu",
            },
            "track": {"name": "a", "description": "smoke"},
            "experiment": {},
        }
    )


def test_train_evaluate_submit_round_trip(synthetic_dataset: Path, tmp_path: Path) -> None:
    """Full pipeline: train -> evaluate -> submit, all on synthetic data."""
    from smth2smth.pipelines.evaluate import run as evaluate_run
    from smth2smth.pipelines.submit import run as submit_run
    from smth2smth.pipelines.train import run as train_run
    from smth2smth.shared.io import validate_submission_csv

    checkpoint_path = tmp_path / "best_model.pt"
    submission_path = tmp_path / "submission.csv"
    cfg = _make_cfg(synthetic_dataset, checkpoint_path, submission_path)

    cwd_before = os.getcwd()
    try:
        # train_run prints OmegaConf.to_yaml; ensure it doesn't change cwd.
        saved_path = train_run(cfg)
        assert saved_path is not None and saved_path.is_file()
        assert checkpoint_path.is_file()

        report = evaluate_run(cfg)
        assert report.num_samples > 0
        assert 0.0 <= report.top1 <= 1.0

        out = submit_run(cfg)
        assert out == submission_path
        assert submission_path.is_file()

        # Format validation: header, integer predictions, value range, unique names.
        validation = validate_submission_csv(
            submission_path,
            num_classes=int(cfg.num_classes),
            expected_video_names={"video_test_1", "video_test_2"},
        )
        assert validation.num_rows == 2
        assert validation.unique_videos == 2

        with submission_path.open(encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["video_name", "predicted_class"]
        assert len(rows) == 1 + 2  # header + 2 test videos
    finally:
        os.chdir(cwd_before)
