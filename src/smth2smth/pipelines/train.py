"""Training pipeline.

Run from the repo root::

    PYTHONPATH=src uv run python -m smth2smth.pipelines.train experiment=baseline_pretrained track=a

Tests can call :func:`run` directly with a hand-built ``DictConfig``; only the
:func:`main` wrapper depends on Hydra.
"""

from __future__ import annotations

import gc
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from smth2smth.shared.data import (
    VideoFrameDataset,
    build_transforms,
    collect_video_samples,
)
from smth2smth.shared.engine import EpochStats, evaluate_epoch, train_one_epoch
from smth2smth.shared.io.checkpoints import save_checkpoint
from smth2smth.shared.models import build_model
from smth2smth.shared.utils import set_seed, split_train_val

CONFIGS_DIR = str(Path(__file__).resolve().parents[3] / "configs")


def _resolve_device(device_str: str) -> torch.device:
    """Resolve the requested device, falling back to CPU when CUDA is missing."""
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def _free_cuda_memory(reason: str = "") -> None:
    """Release Python references and empty the CUDA caching allocator.

    Important for Hydra multirun, where N back-to-back jobs share a single
    Python process: PyTorch holds GPU memory in its caching allocator after a
    job finishes, so the next job can hit OOM even though the previous model
    has gone out of scope. Calling this between jobs (and after an OOM) keeps
    the GPU clean.

    Args:
        reason: Optional string included in the log line for traceability.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        suffix = f" ({reason})" if reason else ""
        free, total = torch.cuda.mem_get_info()
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        print(f"[cuda] cache emptied{suffix}. Free: {free_gb:.2f} / {total_gb:.2f} GiB")


def run(cfg: DictConfig) -> Path | None:
    """Train a model and write the best-by-val-top1 checkpoint.

    Args:
        cfg: Composed Hydra configuration.

    Returns:
        Path to the saved checkpoint, or ``None`` if training never improved
        the initial accuracy (no checkpoint written).
    """
    print(OmegaConf.to_yaml(cfg))

    _free_cuda_memory(reason="run-start")

    set_seed(int(cfg.seed))
    device = _resolve_device(str(cfg.training.device))

    train_dir = Path(cfg.dataset.train_dir).resolve()
    all_samples = collect_video_samples(train_dir)

    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        all_samples = all_samples[: int(max_samples)]

    train_samples, val_samples = split_train_val(
        all_samples,
        val_ratio=float(cfg.dataset.val_ratio),
        seed=int(cfg.dataset.seed),
    )

    use_imagenet_norm = bool(cfg.model.pretrained)
    augment_cfg = cfg.get("augment") if hasattr(cfg, "get") else None
    train_transform = build_transforms(
        image_size=int(cfg.dataset.image_size),
        is_training=True,
        use_imagenet_norm=use_imagenet_norm,
        augment=augment_cfg,
    )
    eval_transform = build_transforms(
        image_size=int(cfg.dataset.image_size),
        is_training=False,
        use_imagenet_norm=use_imagenet_norm,
        augment=augment_cfg,
    )

    num_frames = int(cfg.dataset.num_frames)
    train_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=num_frames,
        transform=train_transform,
        sample_list=train_samples,
    )
    val_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=val_samples,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=int(cfg.training.num_workers),
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=pin_memory,
    )

    model = build_model(cfg).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer_name = str(cfg.training.get("optimizer", "adam")).lower()
    base_lr = float(cfg.training.lr)
    weight_decay = float(cfg.training.get("weight_decay", 0.0))
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=base_lr,
            momentum=float(cfg.training.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(cfg.training.get("nesterov", False)),
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    use_cosine = bool(cfg.training.get("scheduler_cosine", False))
    warmup_epochs = int(cfg.training.get("warmup_epochs", 0))
    cosine_scheduler = None
    if use_cosine:
        cosine_tmax = max(1, int(cfg.training.epochs) - warmup_epochs)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_tmax,
            eta_min=float(cfg.training.get("min_lr", 0.0)),
        )

    label_smoothing = float(cfg.training.get("label_smoothing", 0.0))
    videomix_alpha = float(cfg.training.get("videomix_alpha", 0.0))
    videomix_prob = float(cfg.training.get("videomix_prob", 1.0))
    log_interval_steps = int(cfg.training.get("log_interval_steps", 0))
    early_stopping_enabled = bool(cfg.training.get("early_stopping_enabled", False))
    early_stopping_patience = int(cfg.training.get("early_stopping_patience", 10))
    early_stopping_min_delta = float(cfg.training.get("early_stopping_min_delta", 0.0))

    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    best_top1 = -1.0
    best_path: Path | None = None
    epochs_without_improvement = 0

    try:
        for epoch in range(int(cfg.training.epochs)):
            if warmup_epochs > 0 and epoch < warmup_epochs:
                warm_lr = base_lr * float(epoch + 1) / float(warmup_epochs)
                for group in optimizer.param_groups:
                    group["lr"] = warm_lr
            train_stats: EpochStats = train_one_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                device,
                num_classes=int(cfg.num_classes),
                label_smoothing=label_smoothing,
                videomix_alpha=videomix_alpha,
                videomix_prob=videomix_prob,
                log_interval_steps=log_interval_steps,
            )
            val_stats: EpochStats = evaluate_epoch(model, val_loader, loss_fn, device)
            print(
                f"Epoch {epoch + 1}/{cfg.training.epochs} | "
                f"train loss {train_stats.loss:.4f} top1 {train_stats.top1:.4f} | "
                f"val loss {val_stats.loss:.4f} top1 {val_stats.top1:.4f} top5 {val_stats.top5:.4f}"
            )

            if val_stats.top1 > (best_top1 + early_stopping_min_delta):
                best_top1 = val_stats.top1
                epochs_without_improvement = 0
                best_path = save_checkpoint(
                    checkpoint_path,
                    model,
                    cfg,
                    extra={
                        "val_top1": val_stats.top1,
                        "val_top5": val_stats.top5,
                        "val_loss": val_stats.loss,
                        "epoch": epoch + 1,
                    },
                )
                print(f"  Saved new best checkpoint: {best_path} (val top1={val_stats.top1:.4f})")
            else:
                epochs_without_improvement += 1
                if early_stopping_enabled:
                    print(
                        "  No val top1 improvement "
                        f"({epochs_without_improvement}/{early_stopping_patience}); "
                        f"best remains {best_top1:.4f}"
                    )
                    if epochs_without_improvement >= early_stopping_patience:
                        print(
                            "  Early stopping triggered: "
                            f"no improvement > {early_stopping_min_delta:.6f} for "
                            f"{early_stopping_patience} consecutive epochs."
                        )
                        break
            if cosine_scheduler is not None and epoch >= warmup_epochs:
                cosine_scheduler.step()
    except torch.cuda.OutOfMemoryError as exc:
        print(f"[cuda] OOM during training: {exc}. Releasing memory and aborting this job.")
        del model, optimizer, train_loader, val_loader
        _free_cuda_memory(reason="post-OOM")
        raise
    finally:
        _free_cuda_memory(reason="run-end")

    if best_path is None:
        print("Training finished without producing a checkpoint.")
    else:
        print(f"Done. Best val top1: {best_top1:.4f}. Checkpoint: {best_path}")
    return best_path


@hydra.main(version_base=None, config_path=CONFIGS_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra CLI entrypoint."""
    run(cfg)


if __name__ == "__main__":
    main()
