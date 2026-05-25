"""Optional Weights & Biases logging for training runs."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from omegaconf import DictConfig, OmegaConf


class WandbRun:
    """Thin wrapper around ``wandb`` so training code stays import-safe when W&B is off."""

    def __init__(self, cfg: DictConfig) -> None:
        wb_cfg = cfg.training.get("wandb") if hasattr(cfg, "training") else None
        self._enabled = False
        self._run = None
        if wb_cfg is None or not bool(wb_cfg.get("enabled", False)):
            return
        try:
            import wandb
        except ImportError as exc:
            raise SystemExit(
                "training.wandb.enabled=true but the wandb package is not installed. "
                "Run: uv add wandb"
            ) from exc

        mode = str(wb_cfg.get("mode", "online"))
        if mode == "disabled":
            return

        project = str(wb_cfg.get("project", "smth2smth"))
        entity = wb_cfg.get("entity")
        name = wb_cfg.get("name")
        if not name:
            name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tags = list(wb_cfg.get("tags") or [])
        config = OmegaConf.to_container(cfg, resolve=True)
        init_kwargs: dict[str, Any] = {
            "project": project,
            "name": str(name),
            "config": config,
            "mode": mode,
            "tags": tags,
        }
        if entity is not None:
            init_kwargs["entity"] = str(entity)
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key, relogin=True)
        self._run = wandb.init(**init_kwargs)
        self._wandb = wandb
        self._enabled = True
        self._log_interval_epochs = max(1, int(wb_cfg.get("log_interval_epochs", 1)))

    @property
    def enabled(self) -> bool:
        return self._enabled

    def should_log_epoch(self, epoch_one_indexed: int, total_epochs: int) -> bool:
        if not self._enabled:
            return False
        if epoch_one_indexed == total_epochs:
            return True
        return epoch_one_indexed % self._log_interval_epochs == 0

    def log(
        self,
        metrics: dict[str, float | int],
        *,
        step: int | None = None,
        commit: bool = True,
    ) -> None:
        if self._enabled and self._wandb is not None:
            self._wandb.log(metrics, step=step, commit=commit)

    def log_summary(self, metrics: dict[str, float | int]) -> None:
        if self._enabled and self._run is not None:
            for key, value in metrics.items():
                self._run.summary[key] = value

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None
            self._enabled = False
