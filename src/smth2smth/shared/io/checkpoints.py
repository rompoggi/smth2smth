"""Versioned checkpoint save/load.

A checkpoint stores enough information to reconstruct the model and resume
or evaluate independently of the original training script.

Schema (``schema_version=1``)::

    {
        "schema_version": 1,
        "model_state_dict": <state dict>,
        "config": <merged Hydra config dict>,
        "extra": <free-form dict, e.g. metrics>,
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

CURRENT_SCHEMA_VERSION: int = 1


class CheckpointSchemaError(ValueError):
    """Raised when a checkpoint cannot be interpreted under the current schema."""


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    cfg: DictConfig,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist ``model`` state and ``cfg`` to ``path`` under the current schema.

    Args:
        path: Destination file (parent directories are created if missing).
        model: Network whose ``state_dict`` is saved.
        cfg: Hydra config; serialized via ``OmegaConf.to_container(resolve=True)``.
        extra: Optional free-form metadata (e.g. best validation metrics).

    Returns:
        Resolved absolute :class:`Path` written to disk.
    """
    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "extra": dict(extra) if extra else {},
    }
    torch.save(payload, out_path)
    return out_path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint and validate it against the current schema.

    Args:
        path: Path written by :func:`save_checkpoint`.
        map_location: Forwarded to :func:`torch.load`.

    Returns:
        Dict with keys ``schema_version``, ``model_state_dict``, ``config``, ``extra``.

    Raises:
        CheckpointSchemaError: If the file is missing required fields or has an
            unsupported schema version.
    """
    in_path = Path(path).resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {in_path}")

    payload = torch.load(in_path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise CheckpointSchemaError(
            f"Checkpoint at {in_path} is not a dict; got {type(payload).__name__}"
        )

    schema_version = payload.get("schema_version")
    if schema_version != CURRENT_SCHEMA_VERSION:
        raise CheckpointSchemaError(
            f"Unsupported checkpoint schema_version={schema_version!r} "
            f"(expected {CURRENT_SCHEMA_VERSION})."
        )

    for required in ("model_state_dict", "config"):
        if required not in payload:
            raise CheckpointSchemaError(
                f"Checkpoint at {in_path} is missing required key {required!r}."
            )

    payload.setdefault("extra", {})
    return payload


def cfg_from_checkpoint(checkpoint: dict[str, Any]) -> DictConfig:
    """Re-hydrate the ``OmegaConf`` config saved inside a checkpoint."""
    if "config" not in checkpoint or checkpoint["config"] is None:
        raise CheckpointSchemaError("Checkpoint has no 'config' entry.")
    return OmegaConf.create(checkpoint["config"])
