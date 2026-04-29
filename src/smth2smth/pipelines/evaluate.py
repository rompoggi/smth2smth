"""Evaluation pipeline.

Loads a checkpoint produced by :mod:`smth2smth.pipelines.train` and reports
top-1 / top-5 accuracy on the *full* validation directory (no random split).

Run from the repo root::

    PYTHONPATH=src uv run python -m smth2smth.pipelines.evaluate \\
        training.checkpoint_path=best_model.pt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import hydra
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from smth2smth.pipelines.train import CONFIGS_DIR, _resolve_device
from smth2smth.shared.data import (
    VideoFrameDataset,
    build_transforms,
    collect_video_samples,
)
from smth2smth.shared.engine import evaluate_epoch
from smth2smth.shared.io.checkpoints import cfg_from_checkpoint, load_checkpoint
from smth2smth.shared.models import build_model
from smth2smth.shared.utils import set_seed


@dataclass
class EvalReport:
    """Summary returned by :func:`run`."""

    num_samples: int
    top1: float
    top5: float
    loss: float


def run(cfg: DictConfig) -> EvalReport:
    """Evaluate a saved checkpoint on the full validation directory.

    Args:
        cfg: Hydra configuration. Must have ``cfg.training.checkpoint_path`` set.

    Returns:
        :class:`EvalReport` with the number of validation samples and metrics.
    """
    print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.seed))
    device = _resolve_device(str(cfg.training.device))

    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    saved_cfg = cfg_from_checkpoint(checkpoint)

    # Use the saved config to rebuild the model so architecture matches the weights.
    model = build_model(saved_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    use_imagenet_norm = bool(saved_cfg.model.pretrained)
    # Honor the augment policy that was used at training time (e.g. crop padding)
    # so the eval tensor size and pre-processing matches the trained model.
    augment_cfg = saved_cfg.get("augment") if hasattr(saved_cfg, "get") else None
    eval_transform = build_transforms(
        image_size=int(cfg.dataset.image_size),
        is_training=False,
        use_imagenet_norm=use_imagenet_norm,
        augment=augment_cfg,
    )

    val_dir = Path(cfg.dataset.val_dir).resolve()
    val_samples = collect_video_samples(val_dir)

    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        val_samples = val_samples[: int(max_samples)]

    num_frames = (
        int(saved_cfg.dataset.num_frames) if "dataset" in saved_cfg else int(cfg.dataset.num_frames)
    )
    val_dataset = VideoFrameDataset(
        root_dir=val_dir,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=val_samples,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    loss_fn = nn.CrossEntropyLoss()
    stats = evaluate_epoch(model, val_loader, loss_fn, device)

    print(f"Validation samples: {len(val_dataset)}")
    print(f"Top-1 accuracy: {stats.top1:.4f}")
    print(f"Top-5 accuracy: {stats.top5:.4f}")
    print(f"Loss:           {stats.loss:.4f}")

    return EvalReport(
        num_samples=len(val_dataset),
        top1=stats.top1,
        top5=stats.top5,
        loss=stats.loss,
    )


@hydra.main(version_base=None, config_path=CONFIGS_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra CLI entrypoint."""
    run(cfg)


if __name__ == "__main__":
    main()
