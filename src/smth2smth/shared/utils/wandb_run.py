"""Optional Weights & Biases logging for training pipelines."""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_repo_dotenv() -> None:
    """Load ``.env`` from the repository root when ``python-dotenv`` is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    repo_root = Path(__file__).resolve().parents[4]
    load_dotenv(repo_root / ".env", override=False)


def _resolve_run_name(wb_cfg: Any) -> str | None:
    """Hydra may set ``name`` or ``run_name``; auto-suffix dry-run names with a timestamp."""
    raw = wb_cfg.get("name")
    if raw is None:
        raw = wb_cfg.get("run_name")
    if raw is None:
        return None
    name = str(raw)
    if name == "espadon_official_dryrun" or name.endswith("_dryrun"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{name}_{stamp}"
    return name


class WandbTracker:
    """Thin wrapper around ``wandb.init`` / ``wandb.log`` for the train pipeline.

    Logging contract (single global step axis):

    * **Step metrics** (every ``log_interval_steps`` batches): ``train/loss``,
      ``train/top1``, ``train/top5`` — running averages since the start of the
      current epoch.  Step index: ``epoch_offset + step_in_epoch``.
    * **Epoch metrics** (after train, and val when enabled): ``train/epoch_*``,
      ``val/*``, ``val/best_top1``, ``lr``, optional ``val/ema_*``.  Logged at
      step ``(epoch_index + 1) * steps_per_epoch`` (same as the last train step
      of that epoch).
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._run: Any | None = None
        self.log_interval_epochs = 1
        wb_cfg = cfg.training.get("wandb") if hasattr(cfg, "training") else None
        if wb_cfg is None or not bool(wb_cfg.get("enabled", False)):
            return

        import wandb

        project = str(wb_cfg.get("project") or os.environ.get("WANDB_PROJECT") or "smth2smth")
        entity = wb_cfg.get("entity") or os.environ.get("WANDB_ENTITY")
        run_name = _resolve_run_name(wb_cfg)
        tags = list(wb_cfg.get("tags") or [])
        mode = wb_cfg.get("mode") or os.environ.get("WANDB_MODE")
        self.log_interval_epochs = max(1, int(wb_cfg.get("log_interval_epochs", 1)))

        config_dict = OmegaConf.to_container(cfg, resolve=True)
        init_kwargs: dict[str, Any] = {
            "project": project,
            "config": config_dict,
            "tags": tags,
        }
        if entity:
            init_kwargs["entity"] = str(entity)
        if run_name:
            init_kwargs["name"] = run_name
        if mode:
            init_kwargs["mode"] = str(mode)

        self._run = wandb.init(**init_kwargs)
        print(f"[wandb] run started: {self._run.url if self._run.url else self._run.id}")

    @property
    def enabled(self) -> bool:
        """True when a W&B run is active."""
        return self._run is not None

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log scalar metrics at a global step index."""
        if self._run is not None:
            self._run.log(metrics, step=int(step))

    def finish(self) -> None:
        """End the W&B run."""
        if self._run is not None:
            import wandb

            wandb.finish()
            self._run = None

    def should_log_epoch(self, epoch_one_indexed: int, total_epochs: int) -> bool:
        """Whether to push the epoch summary block this epoch."""
        if self.log_interval_epochs <= 1:
            return True
        if epoch_one_indexed == total_epochs:
            return True
        return epoch_one_indexed % self.log_interval_epochs == 0


def build_step_metrics_callback(
    tracker: WandbTracker,
    *,
    step_offset: int,
) -> Callable[[dict[str, float], int], None] | None:
    """Build a trainer callback that logs step metrics via ``tracker``."""
    if not tracker.enabled:
        return None

    def _callback(metrics: dict[str, float], step_in_epoch: int) -> None:
        tracker.log(metrics, step=step_offset + step_in_epoch)

    return _callback


def log_epoch_summary(
    tracker: WandbTracker,
    *,
    epoch_one_indexed: int,
    steps_per_epoch: int,
    train_stats: Any,
    val_stats: Any | None = None,
    ema_val_stats: Any | None = None,
    lr: float,
    best_top1: float,
) -> None:
    """Log end-of-epoch train / val scalars at the epoch boundary step."""
    if not tracker.enabled:
        return

    metrics: dict[str, float] = {
        "epoch": float(epoch_one_indexed),
        "train/epoch_loss": float(train_stats.loss),
        "train/epoch_top1": float(train_stats.top1),
        "train/epoch_top5": float(train_stats.top5),
        "lr": float(lr),
        "val/best_top1": float(best_top1),
    }
    if val_stats is not None:
        metrics.update(
            {
                "val/loss": float(val_stats.loss),
                "val/top1": float(val_stats.top1),
                "val/top5": float(val_stats.top5),
            }
        )
    if ema_val_stats is not None:
        metrics.update(
            {
                "val/ema_top1": float(ema_val_stats.top1),
                "val/ema_top5": float(ema_val_stats.top5),
            }
        )
    tracker.log(metrics, step=epoch_one_indexed * steps_per_epoch)
