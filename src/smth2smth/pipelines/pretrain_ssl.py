"""Self-supervised pretraining pipeline (DINO-v1 on still frames).

Pretrains a ResNet-50 trunk on the *unlabeled* train+val+test still frames of
the competition dataset. Track A is closed-world: this pipeline does NOT
ingest any external dataset and does NOT load any pretrained weights -- the
trunk starts from random init.

The output is a small ``.pt`` file that contains only ``trunk.state_dict()``
(keyed identically to :class:`smth2smth.shared.models.AvancedResNet50TSM`'s
``backbone`` attribute), so the supervised training pipeline can pick it up
through the ``model.init_from`` config knob.

Run from the repo root::

    PYTHONPATH=src uv run python -m smth2smth.pipelines.pretrain_ssl \\
        experiment=track_a_ssl_pretrain
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torchvision.transforms as T
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

from smth2smth.pipelines.train import CONFIGS_DIR, _free_cuda_memory, _resolve_device
from smth2smth.shared.data import (
    MultiViewStillFramesDataset,
    collect_all_frame_paths,
)
from smth2smth.shared.models.dino_ssl import (
    DinoLoss,
    DinoModel,
    update_teacher_ema,
)
from smth2smth.shared.utils import set_seed


def _build_global_transform(image_size: int) -> T.Compose:
    """Strong DINO-style augmentation for the two global views (resolution
    ``image_size`` x ``image_size``, scale 0.4-1.0)."""
    return T.Compose(
        [
            T.RandomResizedCrop(
                image_size, scale=(0.4, 1.0), interpolation=T.InterpolationMode.BICUBIC
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=23)], p=0.5),
            T.ToTensor(),
        ]
    )


def _build_local_transform(image_size: int) -> T.Compose:
    """Smaller-resolution local crops (DINO multi-crop, scale 0.05-0.4)."""
    return T.Compose(
        [
            T.RandomResizedCrop(
                image_size, scale=(0.05, 0.4), interpolation=T.InterpolationMode.BICUBIC
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=23)], p=0.5),
            T.ToTensor(),
        ]
    )


def _ssl_collate(batch: list[dict]) -> dict[str, list[torch.Tensor]]:
    """Stacks per-sample views into per-view batches.

    Returns a dict with ``global_views: list[Tensor(B, C, H, W)]`` (length 2)
    and ``local_views: list[Tensor(B, C, h, w)]`` (length L), so they can be
    fed straight into the student/teacher.
    """
    n_global = len(batch[0]["global_views"])
    n_local = len(batch[0]["local_views"])
    global_views = [
        torch.stack([item["global_views"][i] for item in batch], dim=0) for i in range(n_global)
    ]
    local_views = [
        torch.stack([item["local_views"][i] for item in batch], dim=0) for i in range(n_local)
    ]
    return {"global_views": global_views, "local_views": local_views}


def _cosine_schedule(base: float, final: float, epoch: int, total_epochs: int) -> float:
    """Cosine schedule from ``base`` (epoch 0) to ``final`` (epoch ``total_epochs - 1``)."""
    if total_epochs <= 1:
        return final
    progress = float(epoch) / float(total_epochs - 1)
    return final + 0.5 * (base - final) * (1.0 + math.cos(math.pi * progress))


def run(cfg: DictConfig) -> Path:
    """DINO pretraining loop.

    Args:
        cfg: Composed Hydra config. Must define ``cfg.pretrain.*`` (number of
            epochs, batch size, head dims, etc.) and the standard
            ``cfg.dataset.*`` paths.

    Returns:
        Path to the saved trunk-only checkpoint.
    """
    print(OmegaConf.to_yaml(cfg))

    _free_cuda_memory(reason="pretrain-start")
    set_seed(int(cfg.seed))
    device = _resolve_device(str(cfg.training.device))

    pcfg = cfg.pretrain
    image_size = int(pcfg.get("image_size", 224))
    local_size = int(pcfg.get("local_image_size", 96))
    num_local_views = int(pcfg.get("num_local_views", 6))

    train_dir = Path(cfg.dataset.train_dir)
    val_dir = Path(cfg.dataset.val_dir)
    test_dir = Path(cfg.dataset.test_dir)
    print(f"[ssl] scanning frame paths from {train_dir}, {val_dir}, {test_dir}")
    frame_paths = collect_all_frame_paths([train_dir, val_dir, test_dir])
    if len(frame_paths) == 0:
        raise SystemExit(
            "No frames found for SSL pretraining; check dataset paths."
        )
    max_frames = pcfg.get("max_frames")
    if max_frames is not None:
        frame_paths = frame_paths[: int(max_frames)]
    print(f"[ssl] using {len(frame_paths)} still frames for pretraining.")

    global_transform = _build_global_transform(image_size)
    local_transform = _build_local_transform(local_size) if num_local_views > 0 else None
    dataset = MultiViewStillFramesDataset(
        frame_paths=frame_paths,
        global_transform=global_transform,
        local_transform=local_transform,
        num_local_views=num_local_views,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(pcfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=_ssl_collate,
    )

    out_dim = int(pcfg.get("out_dim", 4096))
    student = DinoModel(
        out_dim=out_dim,
        hidden_dim=int(pcfg.get("hidden_dim", 2048)),
        bottleneck_dim=int(pcfg.get("bottleneck_dim", 256)),
        n_layers=int(pcfg.get("n_layers", 3)),
    ).to(device)
    teacher = copy.deepcopy(student).to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(
        f"[ssl] student params: {sum(p.numel() for p in student.parameters()):,d} "
        f"(trunk + head, out_dim={out_dim})"
    )

    base_lr = float(pcfg.lr)
    weight_decay = float(pcfg.get("weight_decay", 0.04))
    optimizer = torch.optim.AdamW(student.parameters(), lr=base_lr, weight_decay=weight_decay)
    epochs = int(pcfg.epochs)

    loss_fn = DinoLoss(
        out_dim=out_dim,
        teacher_temp=float(pcfg.get("teacher_temp", 0.04)),
        student_temp=float(pcfg.get("student_temp", 0.1)),
        center_momentum=float(pcfg.get("center_momentum", 0.9)),
    ).to(device)

    amp_enabled = bool(cfg.training.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if amp_enabled else None
    if amp_enabled:
        print("[ssl] AMP (fp16 + GradScaler) enabled.")

    out_path = Path(pcfg.get("checkpoint_path", "ssl_trunk.pt")).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    teacher_momentum_base = float(pcfg.get("teacher_momentum_base", 0.996))
    teacher_momentum_final = float(pcfg.get("teacher_momentum_final", 1.0))

    log_every = int(pcfg.get("log_interval_steps", 50))

    student.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch_idx, batch in enumerate(loader):
            global_views = [v.to(device, non_blocking=True) for v in batch["global_views"]]
            local_views = [v.to(device, non_blocking=True) for v in batch["local_views"]]

            with torch.amp.autocast(
                device_type="cuda", enabled=amp_enabled, dtype=torch.float16
            ):
                # Teacher: only the 2 global views, no grad.
                with torch.no_grad():
                    teacher_logits_per_global = [teacher(v) for v in global_views]
                # Student: ALL views.
                student_logits_per_view = [student(v) for v in global_views] + [
                    student(v) for v in local_views
                ]
                loss = loss_fn(student_logits_per_view, teacher_logits_per_global)

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            momentum = _cosine_schedule(
                teacher_momentum_base,
                teacher_momentum_final,
                epoch * len(loader) + batch_idx,
                epochs * max(1, len(loader)),
            )
            update_teacher_ema(student, teacher, momentum=momentum)

            epoch_loss += float(loss.item())
            n_batches += 1
            if log_every > 0 and batch_idx % log_every == 0:
                print(
                    f"[ssl] epoch {epoch + 1}/{epochs} step {batch_idx}/{len(loader)} "
                    f"loss {float(loss.item()):.4f} momentum {momentum:.5f}"
                )

        avg_loss = epoch_loss / max(1, n_batches)
        print(f"[ssl] epoch {epoch + 1}/{epochs} avg loss {avg_loss:.4f}")

        # Save trunk-only state_dict every epoch (cheap; ~95 MiB for ResNet-50).
        # Keys are stripped of the ``trunk.`` prefix so they match
        # ``AvancedResNet50TSM.backbone`` directly.
        trunk_state_dict = {
            k.removeprefix("trunk."): v
            for k, v in student.state_dict().items()
            if k.startswith("trunk.")
        }
        torch.save({"trunk_state_dict": trunk_state_dict, "epoch": epoch + 1}, out_path)
        print(f"[ssl] wrote trunk checkpoint -> {out_path}")

    _free_cuda_memory(reason="pretrain-end")
    return out_path


@hydra.main(version_base=None, config_path=CONFIGS_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()


__all__ = ["run", "main"]


# Silence unused-import warning when reading this module under static checkers.
_ = (Image, nn)
