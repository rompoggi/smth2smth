"""Tests for the model registry and the bundled models."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from smth2smth.shared.models import (
    CNNLSTM,
    MODEL_REGISTRY,
    CNNBaseline,
    ModelAlreadyRegisteredError,
    UnknownModelError,
    build_model,
    list_registered_models,
    register_model,
)

# Both bundled models share the same input contract; this lets us run forward
# tests in a parametrized loop without duplicating boilerplate.
_BUNDLED_MODELS = [
    ("cnn_baseline", {"num_classes": 7, "pretrained": False}),
    (
        "cnn_lstm",
        {"num_classes": 7, "pretrained": False, "lstm_hidden_size": 32},
    ),
]


def _make_cfg(model_overrides: dict) -> OmegaConf:
    return OmegaConf.create({"model": model_overrides | {"name": "placeholder"}})


class TestRegistry:
    """Registration semantics."""

    def test_bundled_models_are_registered_on_import(self) -> None:
        registered = list_registered_models()
        assert "cnn_baseline" in registered
        assert "cnn_lstm" in registered

    def test_build_model_returns_correct_class(self) -> None:
        cfg = OmegaConf.create(
            {"model": {"name": "cnn_baseline", "num_classes": 5, "pretrained": False}}
        )
        model = build_model(cfg)
        assert isinstance(model, CNNBaseline)

        cfg = OmegaConf.create(
            {
                "model": {
                    "name": "cnn_lstm",
                    "num_classes": 5,
                    "pretrained": False,
                    "lstm_hidden_size": 16,
                }
            }
        )
        model = build_model(cfg)
        assert isinstance(model, CNNLSTM)

    def test_unknown_model_raises_explicit_error(self) -> None:
        cfg = OmegaConf.create(
            {"model": {"name": "does_not_exist", "num_classes": 1, "pretrained": False}}
        )
        with pytest.raises(UnknownModelError):
            build_model(cfg)

    def test_register_model_decorator_blocks_duplicates(self) -> None:
        unique_name = "_test_dup_model_xyz"
        try:

            @register_model(unique_name)
            def _builder(_cfg):  # type: ignore[no-redef]
                return torch.nn.Linear(1, 1)

            with pytest.raises(ModelAlreadyRegisteredError):

                @register_model(unique_name)
                def _builder2(_cfg):  # type: ignore[no-redef]
                    return torch.nn.Linear(1, 1)

        finally:
            MODEL_REGISTRY.pop(unique_name, None)


class TestForwardShapes:
    """Forward-pass contracts for bundled models."""

    @pytest.mark.parametrize("name,overrides", _BUNDLED_MODELS)
    def test_forward_returns_correct_logits_shape(self, name: str, overrides: dict) -> None:
        cfg = OmegaConf.create({"model": overrides | {"name": name}})
        model = build_model(cfg)
        model.eval()
        batch_size, num_frames = 2, 4
        dummy = torch.randn(batch_size, num_frames, 3, 64, 64)
        with torch.no_grad():
            logits = model(dummy)
        assert logits.shape == (batch_size, overrides["num_classes"])
        assert logits.dtype == torch.float32

    @pytest.mark.parametrize("num_frames", [2, 6, 8])
    def test_cnn_lstm_handles_varying_sequence_length(self, num_frames: int) -> None:
        cfg = OmegaConf.create(
            {
                "model": {
                    "name": "cnn_lstm",
                    "num_classes": 4,
                    "pretrained": False,
                    "lstm_hidden_size": 16,
                }
            }
        )
        model = build_model(cfg)
        model.eval()
        with torch.no_grad():
            logits = model(torch.randn(2, num_frames, 3, 64, 64))
        assert logits.shape == (2, 4)


class TestCnnLstmDefaults:
    """Defaults for cnn_lstm.lstm_hidden_size are forgiving."""

    def test_missing_hidden_size_uses_default(self) -> None:
        cfg = OmegaConf.create(
            {"model": {"name": "cnn_lstm", "num_classes": 3, "pretrained": False}}
        )
        model = build_model(cfg)
        assert isinstance(model, CNNLSTM)
        # Default hidden size; verified indirectly via classifier in_features.
        assert model.classifier.in_features in {16, 32, 128, 256, 512}


# Make sure _make_cfg is referenced (silences unused warnings without changing API).
_ = _make_cfg
