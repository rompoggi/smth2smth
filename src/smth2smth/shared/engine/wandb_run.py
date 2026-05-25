"""Optional Weights & Biases logging for training runs."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf


class WandbTracker:
    """Thin wrapper around ``wandb.init`` / ``wandb.log`` / ``wandb.finish``.

    When disabled (or ``wandb`` is not installed), all methods are no-ops.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._enabled = bool(cfg.training.get("wandb_enabled", False))
        self._run: Any = None
        if not self._enabled:
            return
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "training.wandb_enabled=true but wandb is not installed. "
                "Install with: uv pip install wandb"
            ) from exc

        project = str(cfg.training.get("wandb_project", "smth2smth"))
        entity = cfg.training.get("wandb_entity")
        name = cfg.training.get("wandb_name")
        group = cfg.training.get("wandb_group")
        tags_cfg = cfg.training.get("wandb_tags")
        tags = [str(t) for t in tags_cfg] if tags_cfg else None
        mode = str(cfg.training.get("wandb_mode", "online"))

        config_dict = OmegaConf.to_container(cfg, resolve=True)
        init_kwargs: dict[str, Any] = {
            "project": project,
            "config": config_dict,
            "mode": mode,
        }
        if entity is not None:
            init_kwargs["entity"] = str(entity)
        if name is not None:
            init_kwargs["name"] = str(name)
        if group is not None:
            init_kwargs["group"] = str(group)
        if tags:
            init_kwargs["tags"] = tags

        self._run = wandb.init(**init_kwargs)
        print(
            f"[wandb] run started: project={project!r} "
            f"name={getattr(self._run, 'name', name)!r} "
            f"url={getattr(self._run, 'url', 'n/a')}"
        )

    @property
    def enabled(self) -> bool:
        """Return whether a live W&B run is active."""
        return self._run is not None

    def log(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        """Log scalar metrics to the active run.

        Args:
            metrics: Metric names and values (e.g. ``train/loss``).
            step: Optional global step index.
        """
        if self._run is None:
            return
        import wandb

        wandb.log(metrics, step=step)

    def finish(self) -> None:
        """Close the W&B run if one was opened."""
        if self._run is None:
            return
        import wandb

        wandb.finish()
        self._run = None
