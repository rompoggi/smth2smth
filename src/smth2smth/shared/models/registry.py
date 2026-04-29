"""Model registry.

Each model module registers itself by decorating a builder function with
``@register_model("name")``. ``build_model(cfg)`` looks up the requested
``cfg.model.name`` in the registry and calls its builder.

This replaces the ``if/elif`` chain found in the professor's ``train.py``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn
from omegaconf import DictConfig

ModelBuilder = Callable[[DictConfig], nn.Module]

MODEL_REGISTRY: dict[str, ModelBuilder] = {}


class UnknownModelError(KeyError):
    """Raised when ``cfg.model.name`` does not match any registered builder."""


class ModelAlreadyRegisteredError(ValueError):
    """Raised when two builders try to claim the same name."""


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    """Decorator that registers a builder under ``name``.

    Args:
        name: The unique key matched against ``cfg.model.name``.

    Returns:
        A decorator that registers and returns the builder unchanged.

    Raises:
        ModelAlreadyRegisteredError: If another builder already uses ``name``.
    """

    def decorator(builder: ModelBuilder) -> ModelBuilder:
        if name in MODEL_REGISTRY:
            raise ModelAlreadyRegisteredError(f"Model name already registered: {name!r}")
        MODEL_REGISTRY[name] = builder
        return builder

    return decorator


def build_model(cfg: DictConfig) -> nn.Module:
    """Construct the model described by ``cfg.model.name``.

    Args:
        cfg: Hydra config. Must expose ``cfg.model.name``.

    Returns:
        The instantiated model on the CPU. Caller is responsible for ``.to(device)``.

    Raises:
        UnknownModelError: If ``cfg.model.name`` is not in :data:`MODEL_REGISTRY`.
    """
    name = str(cfg.model.name)
    if name not in MODEL_REGISTRY:
        known = ", ".join(sorted(MODEL_REGISTRY)) or "<empty>"
        raise UnknownModelError(f"Unknown model name: {name!r}. Registered: {known}")
    return MODEL_REGISTRY[name](cfg)


def list_registered_models() -> list[str]:
    """Return the sorted list of currently registered model names."""
    return sorted(MODEL_REGISTRY)
