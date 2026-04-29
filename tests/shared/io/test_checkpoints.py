"""Tests for ``smth2smth.shared.io.checkpoints``."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from smth2smth.shared.io.checkpoints import (
    CURRENT_SCHEMA_VERSION,
    CheckpointSchemaError,
    cfg_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


@pytest.fixture
def tiny_model() -> nn.Module:
    model = nn.Linear(4, 3)
    return model


@pytest.fixture
def sample_cfg():
    return OmegaConf.create(
        {
            "model": {"name": "cnn_baseline", "num_classes": 3, "pretrained": False},
            "dataset": {"num_frames": 4},
            "training": {"lr": 0.001},
        }
    )


class TestRoundTrip:
    def test_save_then_load_recovers_state_and_config(
        self, tmp_path: Path, tiny_model: nn.Module, sample_cfg
    ) -> None:
        ckpt_path = save_checkpoint(
            tmp_path / "ckpt.pt", tiny_model, sample_cfg, extra={"val_top1": 0.5}
        )
        assert ckpt_path.is_file()
        loaded = load_checkpoint(ckpt_path)
        assert loaded["schema_version"] == CURRENT_SCHEMA_VERSION
        assert "model_state_dict" in loaded
        assert loaded["config"]["model"]["name"] == "cnn_baseline"
        assert loaded["extra"]["val_top1"] == 0.5

    def test_state_dict_can_reload_into_fresh_model(
        self, tmp_path: Path, tiny_model: nn.Module, sample_cfg
    ) -> None:
        ckpt_path = save_checkpoint(tmp_path / "ckpt.pt", tiny_model, sample_cfg)
        loaded = load_checkpoint(ckpt_path)
        fresh = nn.Linear(4, 3)
        fresh.load_state_dict(loaded["model_state_dict"])
        assert torch.allclose(fresh.weight, tiny_model.weight)
        assert torch.allclose(fresh.bias, tiny_model.bias)

    def test_cfg_from_checkpoint(self, tmp_path: Path, tiny_model: nn.Module, sample_cfg) -> None:
        ckpt_path = save_checkpoint(tmp_path / "ckpt.pt", tiny_model, sample_cfg)
        cfg = cfg_from_checkpoint(load_checkpoint(ckpt_path))
        assert cfg.model.num_classes == 3


class TestSchemaErrors:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "no_such.pt")

    def test_unsupported_schema_version_raises(self, tmp_path: Path, tiny_model: nn.Module) -> None:
        bad_path = tmp_path / "bad.pt"
        torch.save(
            {
                "schema_version": 99,
                "model_state_dict": tiny_model.state_dict(),
                "config": {"model": {"name": "cnn_baseline"}},
            },
            bad_path,
        )
        with pytest.raises(CheckpointSchemaError):
            load_checkpoint(bad_path)

    def test_missing_state_dict_raises(self, tmp_path: Path) -> None:
        bad_path = tmp_path / "bad.pt"
        torch.save(
            {"schema_version": CURRENT_SCHEMA_VERSION, "config": {"foo": 1}},
            bad_path,
        )
        with pytest.raises(CheckpointSchemaError):
            load_checkpoint(bad_path)

    def test_non_dict_payload_raises(self, tmp_path: Path) -> None:
        bad_path = tmp_path / "bad.pt"
        torch.save([1, 2, 3], bad_path)
        with pytest.raises(CheckpointSchemaError):
            load_checkpoint(bad_path)
