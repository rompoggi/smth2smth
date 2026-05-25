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
import torch.nn as nn
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
from smth2smth.shared.utils.augment_log import active_augment_summary


def _now() -> str:
    """ISO-8601 timestamp to second precision, for inline log prefixes."""
    return datetime.now().isoformat(timespec="seconds")


def _cosine_mask_ratio(epoch: int, total_epochs: int, start: float, end: float) -> float:
    """Cosine schedule from ``start`` (epoch 0) to ``end`` (last epoch)."""
    if total_epochs <= 1:
        return float(end)
    progress = float(epoch) / float(total_epochs - 1)
    return float(end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress)))


def _trunk_state_dict(model: nn.Module, *, use_resnet: bool) -> dict[str, torch.Tensor]:
    """Encoder-only weights for downstream ``model.init_from``."""
    if isinstance(model, torch.optim.swa_utils.AveragedModel):
        prefix = "module.backbone." if use_resnet else "module.encoder."
        sd = model.state_dict()
        trunk: dict[str, torch.Tensor] = {}
        for key, value in sd.items():
            if not key.startswith(prefix):
                continue
            rel = key[len(prefix) :]
            out_key = rel if use_resnet else f"encoder.{rel}"
            trunk[out_key] = value
        if not trunk:
            raise RuntimeError(
                f"No tensors with prefix {prefix!r} in AveragedModel state_dict."
            )
        return trunk
    if use_resnet:
        return {k: v for k, v in model.backbone.state_dict().items()}
    return {f"encoder.{k}": v for k, v in model.encoder.state_dict().items()}


def _save_trunk_checkpoint(
    path: Path,
    *,
    trunk_state_dict: dict[str, torch.Tensor],
    epoch: int,
    variant: str,
    model_name: str,
) -> None:
    """Write an encoder-only checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "trunk_state_dict": trunk_state_dict,
            "epoch": epoch,
            "variant": variant,
            "architecture": model_name,
        },
        path,
    )


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
    source_num_frames = int(pcfg.get("source_num_frames", num_frames))
    temporal_expand_mode = str(pcfg.get("temporal_expand_mode", "interpolation"))

    # ── Dataset: unlabeled clips (default train + test; val opt-in) ───────────
    train_dir = Path(cfg.dataset.train_dir)
    val_dir = Path(cfg.dataset.val_dir)
    test_dir = Path(cfg.dataset.test_dir)
    include_val = bool(pcfg.get("include_val_in_pretrain", False))
    ssl_roots = [train_dir, test_dir]
    if include_val:
        ssl_roots = [train_dir, val_dir, test_dir]
        print(f"[videomae] SSL roots: train + val + test ({len(ssl_roots)} dirs)")
    else:
        print(f"[videomae] SSL roots: train + test only (val excluded)")
    video_dirs = collect_all_video_dirs(ssl_roots)
    if len(video_dirs) == 0:
        raise SystemExit("No video folders found for VideoMAE pretraining; check dataset paths.")
    max_videos = pcfg.get("max_videos")
    if max_videos is not None:
        video_dirs = video_dirs[: int(max_videos)]
    if source_num_frames < num_frames:
        print(
            f"[videomae] using {len(video_dirs)} unlabeled clips: "
            f"load T={source_num_frames} -> expand to T={num_frames} "
            f"({temporal_expand_mode})."
        )
    else:
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
        source_num_frames=source_num_frames,
        temporal_expand_mode=temporal_expand_mode,
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

        grad_ckpt = bool(cfg.model.get("gradient_checkpointing", False)) or bool(
            pcfg.get("gradient_checkpointing", False)
        )
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
            gradient_checkpointing=grad_ckpt,
            dual_masking=bool(pcfg.get("dual_masking", False)),
            decoder_keep_ratio=float(pcfg.get("decoder_keep_ratio", 0.50)),
            decoder_cell_h=int(pcfg.get("decoder_cell_h", 2)),
            decoder_cell_w=int(pcfg.get("decoder_cell_w", 2)),
        ).to(device)
        if bool(pcfg.get("dual_masking", False)):
            print(
                f"[videomae] V2 dual masking: encoder tube ratio="
                f"{float(pcfg.get('mask_ratio', 0.75)):g}, decoder running-cell keep="
                f"{float(pcfg.get('decoder_keep_ratio', 0.50)):g} on "
                f"({int(pcfg.get('decoder_cell_h', 2))}x{int(pcfg.get('decoder_cell_w', 2))}) cells."
            )
        if grad_ckpt:
            print("[videomae] encoder gradient checkpointing enabled.")

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

    max_grad_norm = float(pcfg.get("max_grad_norm", 3.0))
    if max_grad_norm > 0.0:
        print(f"[videomae] gradient clipping enabled (max_norm={max_grad_norm:g}).")

    ema_enabled = bool(pcfg.get("ema_enabled", False))
    ema_decay = float(pcfg.get("ema_decay", 0.9999))
    ema_model: torch.optim.swa_utils.AveragedModel | None = None
    if ema_enabled and not use_resnet:

        def _ema_avg_fn(
            avg_param: torch.Tensor, model_param: torch.Tensor, num_averaged: int
        ) -> torch.Tensor:
            return ema_decay * avg_param + (1.0 - ema_decay) * model_param

        ema_model = torch.optim.swa_utils.AveragedModel(
            model, avg_fn=_ema_avg_fn, use_buffers=True
        )
        print(f"[videomae] EMA enabled (decay={ema_decay:g}).")

    milestone_epochs = sorted(
        {int(e) for e in (pcfg.get("checkpoint_milestones") or []) if int(e) > 0}
    )
    if milestone_epochs:
        print(f"[videomae] milestone encoder checkpoints at epochs: {milestone_epochs}")

    wandb_run = None
    if bool(pcfg.get("wandb_enabled", False)):
        try:
            import wandb
        except ImportError as exc:
            raise SystemExit(
                "pretrain.wandb_enabled=true but wandb is not installed. "
                "Run: uv add wandb"
            ) from exc
        augment_summary = active_augment_summary(
            augment_cfg if augment_cfg is not None else None
        )
        wandb_config = {
            "experiment": str(cfg.get("experiment", "videomae_pretrain")),
            "seed": int(cfg.seed),
            "num_frames": num_frames,
            "source_num_frames": source_num_frames,
            "temporal_expand_mode": temporal_expand_mode,
            "include_val_in_pretrain": include_val,
            "variant": variant,
            "epochs": epochs,
            "batch_size": int(pcfg.batch_size),
            "grad_accum_steps": int(pcfg.get("grad_accum_steps", 1)),
            "lr": base_lr,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "ema_enabled": ema_enabled,
            "ema_decay": ema_decay if ema_enabled else None,
            "max_grad_norm": max_grad_norm,
            "mask_ratio": float(pcfg.get("mask_ratio", 0.75)),
            "dual_masking": bool(pcfg.get("dual_masking", False)),
            "augment": augment_summary,
        }
        wandb_run = wandb.init(
            project=str(pcfg.get("wandb_project", "smth2smth")),
            entity=pcfg.get("wandb_entity"),
            name=pcfg.get("wandb_run_name"),
            group=str(pcfg.get("wandb_group", "track_a_videomae_pretrain")),
            config=wandb_config,
            resume="allow",
        )
        print(f"[wandb] run {wandb_run.url}")

    # ── Checkpoint path ───────────────────────────────────────────────────────
    out_path = Path(pcfg.get("checkpoint_path", "videomae_encoder.pt")).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Full resume state (encoder + decoder + optimizer + scaler + step) lives next
    # to the trunk checkpoint as ``<stem>.state.pt`` — auto-loaded if present so a
    # crash-and-restart resumes the run exactly where it left off.
    state_path = out_path.with_suffix(".state.pt")

    log_every = int(pcfg.get("log_interval_steps", 100))
    tube_t = int(cfg.model.get("tube_t", 2))
    patch_size = int(cfg.model.get("patch_size", 16))
    norm_pix = bool(pcfg.get("norm_pix", True))
    norm_feat = bool(pcfg.get("norm_feat", True))
    grad_accum_steps = max(1, int(pcfg.get("grad_accum_steps", 1)))
    mask_schedule = bool(pcfg.get("mask_ratio_schedule", False))
    mask_start = float(pcfg.get("mask_ratio_start", 0.90))
    mask_end = float(pcfg.get("mask_ratio_end", pcfg.get("mask_ratio", 0.75)))
    if mask_schedule:
        print(f"[videomae] mask_ratio cosine schedule {mask_start:g} -> {mask_end:g}")

    start_epoch = 0
    global_step = 0
    auto_resume = bool(pcfg.get("auto_resume", True))
    if auto_resume and not use_resnet and state_path.is_file():
        print(f"[videomae] {_now()} resuming from state {state_path}")
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if scaler is not None and state.get("scaler_state_dict") is not None:
            scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch = int(state.get("epoch", 0))
        global_step = int(state.get("global_step", 0))
        print(
            f"[videomae] {_now()} resumed at epoch {start_epoch}/{epochs}, "
            f"global_step {global_step}"
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, epochs):
        if mask_schedule and not use_resnet:
            model.mask_ratio = _cosine_mask_ratio(epoch, epochs, mask_start, mask_end)
            if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % 50 == 0:
                print(
                    f"[videomae] {_now()} epoch {epoch + 1}: "
                    f"mask_ratio={model.mask_ratio:.4f}"
                )

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
                    pred, _, ids_mask, ids_predict = model(clip)
                    loss = videomae_pixel_loss(
                        pred,
                        clip,
                        ids_predict,
                        tube_t=tube_t,
                        patch_size=patch_size,
                        norm_pix=norm_pix,
                        ids_encoder_mask=(ids_mask if model.dual_masking else None),
                    )

            scaled_loss = loss / float(grad_accum_steps)
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            is_accum_step = (
                (batch_idx + 1) % grad_accum_steps == 0
                or (batch_idx + 1) == len(loader)
            )
            if is_accum_step:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    if max_grad_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if max_grad_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema_model is not None:
                    ema_model.update_parameters(model)

            global_step += 1
            loss_val = float(loss.item())
            epoch_loss += loss_val
            n_batches += 1

            if log_every > 0 and batch_idx % log_every == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[videomae] {_now()} epoch {epoch + 1}/{epochs} "
                    f"step {batch_idx}/{len(loader)} "
                    f"loss {loss_val:.4f} lr {current_lr:.2e}"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": loss_val,
                            "train/lr": current_lr,
                            "train/epoch": epoch + 1,
                        },
                        step=global_step,
                    )

        avg_loss = epoch_loss / max(1, n_batches)
        ep = epoch + 1
        print(f"[videomae] {_now()} epoch {ep}/{epochs} avg loss {avg_loss:.4f}")
        if wandb_run is not None:
            wandb_run.log({"train/epoch_avg_loss": avg_loss, "train/epoch": ep}, step=global_step)

        ckpt_module = ema_model if ema_model is not None else model
        trunk_state_dict = _trunk_state_dict(ckpt_module, use_resnet=use_resnet)  # type: ignore[arg-type]
        _save_trunk_checkpoint(
            out_path,
            trunk_state_dict=trunk_state_dict,
            epoch=ep,
            variant=variant,
            model_name=model_name,
        )
        ckpt_kind = "ema" if ema_model is not None else "live"
        print(
            f"[videomae] {_now()} wrote trunk checkpoint ({ckpt_kind}) -> {out_path} "
            f"({len(trunk_state_dict)} tensors)"
        )

        if ep in milestone_epochs:
            milestone_path = out_path.with_name(f"{out_path.stem}_ep{ep}{out_path.suffix}")
            _save_trunk_checkpoint(
                milestone_path,
                trunk_state_dict=trunk_state_dict,
                epoch=ep,
                variant=variant,
                model_name=model_name,
            )
            print(f"[videomae] {_now()} wrote milestone checkpoint -> {milestone_path}")

        # Full resume state (skip for the ResNet-feature pretrain variant — its
        # model layout differs and the FT pipelines don't ever resume it).
        if not use_resnet:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                    "epoch": ep,
                    "global_step": global_step,
                    "variant": variant,
                    "architecture": model_name,
                },
                state_path,
            )
            print(f"[videomae] {_now()} wrote resume state -> {state_path}")

    if wandb_run is not None:
        wandb_run.finish()

    _free_cuda_memory(reason="videomae-pretrain-end")
    return out_path


@hydra.main(version_base=None, config_path=CONFIGS_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra CLI entrypoint."""
    run(cfg)


if __name__ == "__main__":
    main()


__all__ = ["main", "run"]
