"""V-JEPA-style clip-level self-supervised pretraining pipeline.

Pretrains a TSM-equipped ResNet-50 trunk on the *unlabeled* train+val+test
clips of the competition dataset. Track A is closed-world: this pipeline
does NOT ingest external data and does NOT load pretrained weights -- the
trunk starts from random init.

The output is a small ``.pt`` file that contains only the trunk
``state_dict`` (keyed identically to
:class:`smth2smth.shared.models.AvancedResNet50TSM`'s ``backbone``
attribute), so the supervised training pipeline can pick it up through
``cfg.model.init_from``. Because the V-JEPA trunk **is** the TSM-wrapped
backbone, no key renaming is needed at load time (the existing
:func:`smth2smth.pipelines.train._ssl_trunk_to_supervised_keys` is a
no-op for TSM-wrapped keys and just adds the ``backbone.`` prefix).

Run from the repo root::

    PYTHONPATH=src uv run python -m smth2smth.pipelines.pretrain_vjepa \\
        experiment=track_a_vjepa_pretrain
"""

from __future__ import annotations

import copy
import math
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
from smth2smth.shared.models import (
    VJepaModel,
    apply_frame_mask,
    make_frame_mask,
    update_vjepa_teacher_ema,
    vjepa_feature_loss,
)
from smth2smth.shared.utils import set_seed


def _cosine_schedule(base: float, final: float, step: int, total_steps: int) -> float:
    """Half-cosine interpolation from ``base`` (step 0) to ``final``
    (step ``total_steps - 1``)."""
    if total_steps <= 1:
        return final
    progress = float(step) / float(total_steps - 1)
    progress = max(0.0, min(1.0, progress))
    return final + 0.5 * (base - final) * (1.0 + math.cos(math.pi * progress))


def _build_clip_transform(image_size: int, augment_cfg: DictConfig | None) -> object:
    """Build the per-clip transform used for both student and teacher views.

    We reuse the project's standard :func:`build_transforms` (with
    ``is_training=True``) so the SSL run sees exactly the same input
    distribution as the supervised trainer -- crucial for the resulting
    weights to transfer cleanly. ``use_imagenet_norm=False`` because Track A
    forbids pretrained ImageNet weights.
    """
    return build_transforms(
        image_size=image_size,
        is_training=True,
        use_imagenet_norm=False,
        augment=augment_cfg,
    )


def run(cfg: DictConfig) -> Path:
    """V-JEPA pretraining loop.

    Args:
        cfg: Composed Hydra config. Must define ``cfg.pretrain.*`` (epochs,
            batch size, masking probabilities, etc.) and the standard
            ``cfg.dataset.*`` paths.

    Returns:
        Path to the saved trunk-only checkpoint.
    """
    print(OmegaConf.to_yaml(cfg))

    _free_cuda_memory(reason="vjepa-pretrain-start")
    set_seed(int(cfg.seed))
    device = _resolve_device(str(cfg.training.device))

    pcfg = cfg.pretrain
    image_size = int(pcfg.get("image_size", int(cfg.dataset.image_size)))
    num_frames = int(pcfg.get("num_frames", int(cfg.dataset.num_frames)))

    train_dir = Path(cfg.dataset.train_dir)
    val_dir = Path(cfg.dataset.val_dir)
    test_dir = Path(cfg.dataset.test_dir)
    print(f"[vjepa] scanning video folders from {train_dir}, {val_dir}, {test_dir}")
    video_dirs = collect_all_video_dirs([train_dir, val_dir, test_dir])
    if len(video_dirs) == 0:
        raise SystemExit("No video folders found for V-JEPA pretraining; check dataset paths.")
    max_videos = pcfg.get("max_videos")
    if max_videos is not None:
        video_dirs = video_dirs[: int(max_videos)]
    print(f"[vjepa] using {len(video_dirs)} unlabeled clips (T={num_frames}).")

    augment_cfg = cfg.get("augment") if hasattr(cfg, "get") else None
    clip_transform = _build_clip_transform(image_size=image_size, augment_cfg=augment_cfg)
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

    student = VJepaModel(
        num_frames=num_frames,
        shift_div=int(cfg.model.get("shift_div", 8)),
        predictor_hidden_dim=int(pcfg.get("predictor_hidden_dim", 2048)),
    ).to(device)
    teacher = copy.deepcopy(student).to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    n_student_params = sum(p.numel() for p in student.parameters())
    print(
        f"[vjepa] student params: {n_student_params:,d} "
        f"(trunk + predictor); predictor_hidden_dim="
        f"{int(pcfg.get('predictor_hidden_dim', 2048))}"
    )

    base_lr = float(pcfg.lr)
    weight_decay = float(pcfg.get("weight_decay", 0.04))
    optimizer = torch.optim.AdamW(student.parameters(), lr=base_lr, weight_decay=weight_decay)
    epochs = int(pcfg.epochs)

    amp_enabled = bool(cfg.training.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if amp_enabled else None
    if amp_enabled:
        print("[vjepa] AMP (fp16 + GradScaler) enabled.")

    out_path = Path(pcfg.get("checkpoint_path", "vjepa_trunk.pt")).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mask_prob = float(pcfg.get("mask_prob", 0.5))
    min_mask = int(pcfg.get("min_mask", 1))
    max_mask = pcfg.get("max_mask")
    max_mask = int(max_mask) if max_mask is not None else None

    teacher_momentum_base = float(pcfg.get("teacher_momentum_base", 0.996))
    teacher_momentum_final = float(pcfg.get("teacher_momentum_final", 1.0))
    log_every = int(pcfg.get("log_interval_steps", 50))

    total_steps = epochs * max(1, len(loader))
    global_step = 0

    student.train()
    _warned_zero_loss = False
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch_idx, clip in enumerate(loader):
            clip = clip.to(device, non_blocking=True)  # (B, T, C, H, W)

            mask = make_frame_mask(
                batch_size=clip.shape[0],
                num_frames=clip.shape[1],
                mask_prob=mask_prob,
                min_mask=min_mask,
                max_mask=max_mask,
                device=device,
            )
            masked_clip = apply_frame_mask(clip, mask)

            with torch.amp.autocast(
                device_type="cuda", enabled=amp_enabled, dtype=torch.float16
            ):
                with torch.no_grad():
                    teacher_feats = teacher.encode(clip)  # (B, T, D)
                predicted, _ = student(masked_clip)  # (B, T, D)
                # The L1 loss is well-defined under autocast and stable in
                # fp16 with the GradScaler.
                loss = vjepa_feature_loss(
                    predicted=predicted,
                    teacher=teacher_feats.detach(),
                    mask=mask,
                )

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            momentum = _cosine_schedule(
                base=teacher_momentum_base,
                final=teacher_momentum_final,
                step=global_step,
                total_steps=total_steps,
            )
            update_vjepa_teacher_ema(student, teacher, momentum=momentum)
            global_step += 1

            loss_val = float(loss.item())
            epoch_loss += loss_val
            n_batches += 1
            # First-step sanity check: if the loss is identically zero, the
            # forward path is decoupled from the gradient (e.g. an fp16
            # denominator saturating to ``+inf``, mask collapse, etc.) and
            # the run is wasting compute. Warn loudly and continue.
            if (
                not _warned_zero_loss
                and global_step <= 5
                and (loss_val == 0.0 or not math.isfinite(loss_val))
            ):
                print(
                    f"[vjepa][WARNING] suspicious loss at step {global_step}: "
                    f"{loss_val!r}. Check mask sampling and AMP precision; "
                    f"the trunk is not learning under this configuration."
                )
                _warned_zero_loss = True
            if log_every > 0 and batch_idx % log_every == 0:
                n_masked = int(mask.sum().item())
                print(
                    f"[vjepa] epoch {epoch + 1}/{epochs} step "
                    f"{batch_idx}/{len(loader)} loss {loss_val:.4f} "
                    f"momentum {momentum:.5f} masked-frames {n_masked}"
                )

        avg_loss = epoch_loss / max(1, n_batches)
        print(f"[vjepa] epoch {epoch + 1}/{epochs} avg loss {avg_loss:.4f}")

        # Save trunk-only state_dict every epoch. The path is
        #
        #   student.state_dict() ──► strip ``trunk.backbone.`` prefix
        #                       ──► save plain ResNet-50 keys
        #                           (e.g. ``layer1.0.conv1.1.weight``).
        #
        # At supervised-load time, ``_ssl_trunk_to_supervised_keys``
        # prepends ``backbone.`` so the keys match
        # :class:`AvancedResNet50TSM.backbone` exactly. This is the same
        # convention the DINO trunk in :mod:`pretrain_ssl` uses: SSL
        # checkpoints carry plain ResNet keys, the supervised loader
        # adapts them.
        trunk_state_dict: dict[str, torch.Tensor] = {}
        for k, v in student.state_dict().items():
            if k.startswith("trunk.backbone."):
                trunk_state_dict[k.removeprefix("trunk.backbone.")] = v
        torch.save(
            {"trunk_state_dict": trunk_state_dict, "epoch": epoch + 1},
            out_path,
        )
        print(
            f"[vjepa] wrote trunk checkpoint -> {out_path} "
            f"({len(trunk_state_dict)} tensors)"
        )

    _free_cuda_memory(reason="vjepa-pretrain-end")
    return out_path


@hydra.main(version_base=None, config_path=CONFIGS_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra CLI entrypoint."""
    run(cfg)


if __name__ == "__main__":
    main()


__all__ = ["main", "run"]
