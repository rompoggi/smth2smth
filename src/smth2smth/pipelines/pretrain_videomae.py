"""VideoMAE masked-reconstruction self-supervised pretraining pipeline (Track A).

Follows VideoMAE (Tong et al., NeurIPS 2022) adapted for our 4-frame SSv2 clips:
  - 90% tube masking, 3D cube embedding 2×16×16
  - AdamW β=(0.9, 0.95), weight_decay=0.05, cosine LR with linear warmup
  - Reconstruction target: per-cube normalized pixel MSE at masked positions
  - Output: encoder-only checkpoint (keys prefixed with ``encoder.``)

The output can be fed to the supervised trainer via ``model.init_from``.

Run from the repo root::

    PYTHONPATH=src uv run python -m smth2smth.pipelines.pretrain_videomae \\
        experiment=track_a_videomae_pretrain
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from smth2smth.pipelines.train import CONFIGS_DIR, _free_cuda_memory, _resolve_device
from smth2smth.shared.data import (
    ClipSSLDataset,
    build_transforms,
    collect_all_video_dirs,
)
from smth2smth.shared.models.video_mae import (
    VideoMAEPretrainModel,
    videomae_pixel_loss,
)
from smth2smth.shared.models.video_mae_resnet import (
    VideoMAEResNetPretrainModel,
    videomae_resnet_feature_loss,
)
from smth2smth.shared.utils import set_seed


def _warmup_cosine_lr(
    optimizer: torch.optim.Optimizer,
    step: int,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    min_lr: float = 0.0,
) -> None:
    """Set optimizer LR using linear warmup + cosine decay (in-place)."""
    if warmup_steps > 0 and step < warmup_steps:
        lr = base_lr * (step + 1) / warmup_steps
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = max(0.0, min(1.0, progress))
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def run(cfg: DictConfig) -> Path:
    """VideoMAE pretraining loop.

    Args:
        cfg: Composed Hydra config. Must define ``cfg.pretrain.*`` and the
            standard ``cfg.dataset.*`` paths.

    Returns:
        Path to the saved encoder-only checkpoint.
    """
    print(OmegaConf.to_yaml(cfg))

    _free_cuda_memory(reason="videomae-pretrain-start")
    set_seed(int(cfg.seed))
    device = _resolve_device(str(cfg.training.device))

    pcfg = cfg.pretrain
    image_size = int(pcfg.get("image_size", int(cfg.dataset.image_size)))
    num_frames = int(pcfg.get("num_frames", int(cfg.dataset.num_frames)))

    # ── Dataset: all unlabeled clips (train + val + test) ─────────────────────
    train_dir = Path(cfg.dataset.train_dir)
    val_dir = Path(cfg.dataset.val_dir)
    test_dir = Path(cfg.dataset.test_dir)
    print(f"[videomae] scanning video folders from {train_dir}, {val_dir}, {test_dir}")
    video_dirs = collect_all_video_dirs([train_dir, val_dir, test_dir])
    if len(video_dirs) == 0:
        raise SystemExit("No video folders found for VideoMAE pretraining; check dataset paths.")
    max_videos = pcfg.get("max_videos")
    if max_videos is not None:
        video_dirs = video_dirs[: int(max_videos)]
    print(f"[videomae] using {len(video_dirs)} unlabeled clips (T={num_frames}).")

    # No flip augmentation (SSv2 is direction-sensitive)
    augment_cfg = cfg.get("augment") if hasattr(cfg, "get") else None
    clip_transform = build_transforms(
        image_size=image_size,
        is_training=True,
        use_imagenet_norm=False,
        augment=augment_cfg,
    )
    dataset = ClipSSLDataset(
        video_dirs=video_dirs,
        num_frames=num_frames,
        transform=clip_transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(pcfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model_name = str(cfg.model.get("name", "video_mae_vit"))
    use_resnet = model_name == "video_mae_resnet"
    variant = str(cfg.model.get("variant", "vit_b"))

    if use_resnet:
        model = VideoMAEResNetPretrainModel(
            num_frames=num_frames,
            img_size=image_size,
            shift_div=int(cfg.model.get("shift_div", 8)),
            shift_place=str(cfg.model.get("shift_place", "blockres")),
            mask_ratio=float(pcfg.get("mask_ratio", 0.75)),
            encoder_depth=int(cfg.model.get("encoder_depth", 2)),
            encoder_heads=int(cfg.model.get("encoder_heads", 8)),
        ).to(device)
        variant = "resnet50_tsm"
        n_params = sum(p.numel() for p in model.parameters())
        n_backbone = sum(p.numel() for p in model.backbone.parameters())
        print(
            f"[videomae] model: {model_name} ({variant}), "
            f"total params={n_params:,d}, backbone={n_backbone:,d}"
        )
    else:
        _variants = {
            "vit_s": dict(embed_dim=384, depth=12, num_heads=6),
            "vit_b": dict(embed_dim=768, depth=12, num_heads=12),
            "vit_l": dict(embed_dim=1024, depth=24, num_heads=16),
        }
        if variant not in _variants:
            raise SystemExit(f"Unknown variant {variant!r}; choose from {sorted(_variants)}")
        arch = _variants[variant]

        model = VideoMAEPretrainModel(
            num_frames=num_frames,
            img_size=image_size,
            tube_t=int(cfg.model.get("tube_t", 2)),
            patch_size=int(cfg.model.get("patch_size", 16)),
            embed_dim=arch["embed_dim"],
            depth=arch["depth"],
            num_heads=arch["num_heads"],
            mlp_ratio=float(cfg.model.get("mlp_ratio", 4.0)),
            drop_path_rate=float(pcfg.get("drop_path_rate", 0.0)),
            mask_ratio=float(pcfg.get("mask_ratio", 0.75)),
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        n_enc = sum(p.numel() for p in model.encoder.parameters())
        print(f"[videomae] model: {variant}, total params={n_params:,d}, encoder={n_enc:,d}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # VideoMAE paper: AdamW, β=(0.9, 0.95), wd=0.05, base LR 1.5e-4
    base_lr = float(pcfg.lr)
    weight_decay = float(pcfg.get("weight_decay", 0.05))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )

    epochs = int(pcfg.epochs)
    steps_per_epoch = max(1, len(loader))
    total_steps = epochs * steps_per_epoch
    warmup_epochs = int(pcfg.get("warmup_epochs", 5))
    warmup_steps = warmup_epochs * steps_per_epoch
    min_lr = float(pcfg.get("min_lr", 0.0))

    # ── AMP ───────────────────────────────────────────────────────────────────
    amp_enabled = bool(cfg.training.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if amp_enabled else None
    if amp_enabled:
        print("[videomae] AMP (bfloat16) enabled.")

    # ── Checkpoint path ───────────────────────────────────────────────────────
    out_path = Path(pcfg.get("checkpoint_path", "videomae_encoder.pt")).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log_every = int(pcfg.get("log_interval_steps", 100))
    tube_t = int(cfg.model.get("tube_t", 2))
    patch_size = int(cfg.model.get("patch_size", 16))
    norm_pix = bool(pcfg.get("norm_pix", True))
    norm_feat = bool(pcfg.get("norm_feat", True))

    global_step = 0
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch_idx, clip in enumerate(loader):
            clip = clip.to(device, non_blocking=True)  # (B, T, C, H, W)

            _warmup_cosine_lr(optimizer, global_step, warmup_steps, total_steps, base_lr, min_lr)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=amp_enabled,
                dtype=torch.bfloat16,  # bfloat16 preferred: no inf for fp16 recon loss
            ):
                if use_resnet:
                    pred, target, _, _ = model(clip)
                    loss = videomae_resnet_feature_loss(pred, target, norm_feat=norm_feat)
                else:
                    pred, _, ids_mask = model(clip)
                    loss = videomae_pixel_loss(
                        pred,
                        clip,
                        ids_mask,
                        tube_t=tube_t,
                        patch_size=patch_size,
                        norm_pix=norm_pix,
                    )

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()

            global_step += 1
            loss_val = float(loss.item())
            epoch_loss += loss_val
            n_batches += 1

            if log_every > 0 and batch_idx % log_every == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[videomae] epoch {epoch + 1}/{epochs} "
                    f"step {batch_idx}/{len(loader)} "
                    f"loss {loss_val:.4f} lr {current_lr:.2e}"
                )

        avg_loss = epoch_loss / max(1, n_batches)
        ep = epoch + 1
        ts = datetime.now().isoformat(timespec="seconds")
        if ep == 1 or ep == epochs or ep % 50 == 0:
            print(f"[videomae] {ts} epoch {ep}/{epochs} avg loss {avg_loss:.4f}")
        else:
            print(f"[videomae] epoch {ep}/{epochs} avg loss {avg_loss:.4f}")

        if use_resnet:
            # Plain ResNet+TSM keys — same layout as DINO/V-JEPA SSL trunks.
            trunk_state_dict = {
                k: v for k, v in model.backbone.state_dict().items()
            }
        else:
            # ViT encoder keys prefixed with ``encoder.`` for ``video_mae_vit``.
            trunk_state_dict = {
                f"encoder.{k}": v for k, v in model.encoder.state_dict().items()
            }
        torch.save(
            {
                "trunk_state_dict": trunk_state_dict,
                "epoch": epoch + 1,
                "variant": variant,
                "architecture": model_name,
            },
            out_path,
        )
        print(f"[videomae] wrote trunk checkpoint -> {out_path} ({len(trunk_state_dict)} tensors)")

    _free_cuda_memory(reason="videomae-pretrain-end")
    return out_path


@hydra.main(version_base=None, config_path=CONFIGS_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra CLI entrypoint."""
    run(cfg)


if __name__ == "__main__":
    main()


__all__ = ["main", "run"]
