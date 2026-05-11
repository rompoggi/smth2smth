"""Training pipeline.

Run from the repo root::

    PYTHONPATH=src uv run python -m smth2smth.pipelines.train experiment=baseline_pretrained track=a

Tests can call :func:`run` directly with a hand-built ``DictConfig``; only the
:func:`main` wrapper depends on Hydra.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, WeightedRandomSampler

from smth2smth.shared.data import (
    VideoFrameDataset,
    build_time_reversal_table,
    build_transforms,
    collect_video_samples,
    describe_time_reversal_table,
)
from smth2smth.shared.engine import EpochStats, evaluate_epoch, train_one_epoch
from smth2smth.shared.io.checkpoints import load_checkpoint, save_checkpoint
from smth2smth.shared.models import build_model
from smth2smth.shared.utils import (
    compute_class_weights,
    compute_sample_weights,
    set_seed,
    split_train_val,
)

CONFIGS_DIR = str(Path(__file__).resolve().parents[3] / "configs")


def _resolve_device(device_str: str) -> torch.device:
    """Resolve the requested device, falling back to CPU when CUDA is missing."""
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def _ssl_trunk_to_supervised_keys(trunk_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Adapt a plain-ResNet-50 SSL trunk state_dict to the supervised model's
    backbone state_dict layout.

    :func:`AvancedResNet50TSM` wraps every bottleneck block's ``conv1`` in a
    ``nn.Sequential(TemporalShift, conv1)``, which renames the parameter key
    from ``layerN.M.conv1.weight`` to ``layerN.M.conv1.1.weight``. SSL
    pretraining has no temporal axis so its trunk has plain ``conv1``; this
    helper performs the renaming so the SSL weights drop into the supervised
    model's ``backbone`` directly. Other keys (``conv2``, ``conv3``, batchnorms,
    stem, etc.) are passed through unchanged.

    The returned dict is also prefixed with ``backbone.`` so it can be passed
    straight to ``AvancedResNet50TSM.load_state_dict(..., strict=False)``.
    """
    import re

    block_conv1_re = re.compile(r"^(layer[1-4]\.\d+\.conv1)\.(weight|bias)$")
    out: dict[str, torch.Tensor] = {}
    for k, v in trunk_state.items():
        match = block_conv1_re.match(k)
        if match is not None:
            new_k = f"{match.group(1)}.1.{match.group(2)}"
        else:
            new_k = k
        out[f"backbone.{new_k}"] = v
    return out


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

    use_official_val = bool(cfg.dataset.get("use_official_val", False))
    if use_official_val:
        # Validate on the official held-out folder. The internal 80/20 split is
        # bypassed: training uses *all* of ``train_dir``, validation uses
        # *all* of ``val_dir``.
        val_dir_for_val = Path(cfg.dataset.val_dir).resolve()
        val_samples = collect_video_samples(val_dir_for_val)
        if max_samples is not None:
            val_samples = val_samples[: int(max_samples)]
        train_samples = all_samples
        print(
            f"[data] use_official_val=true: train={len(train_samples)} "
            f"(from {train_dir}), val={len(val_samples)} (from {val_dir_for_val})"
        )
    else:
        train_samples, val_samples = split_train_val(
            all_samples,
            val_ratio=float(cfg.dataset.val_ratio),
            seed=int(cfg.dataset.seed),
        )

    # Class indices that actually have at least one training sample. Recorded
    # in the checkpoint so :mod:`smth2smth.pipelines.submit` can mask logits
    # for never-trained classes (e.g. the missing class 27 in the Track A
    # dataset would otherwise win a fraction of test predictions by accident).
    trained_class_indices: list[int] = sorted({int(label) for _, label in train_samples})

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

    # Time-reversal augmentation (label-aware). Default ``time_reversal_prob=0``
    # is a no-op. When > 0, we build a per-class remap from the class folder
    # names and pass it to the train dataset only; the val dataset never
    # reverses (we want unbiased val numbers).
    time_reversal_prob = float(cfg.dataset.get("time_reversal_prob", 0.0))
    tr_perm: torch.Tensor | None = None
    tr_allow: torch.Tensor | None = None
    if time_reversal_prob > 0.0:
        tr_perm, tr_allow = build_time_reversal_table(
            train_dir=train_dir, num_classes=int(cfg.num_classes)
        )
        print(f"[time-reversal] prob={time_reversal_prob}. " + describe_time_reversal_table(tr_perm, tr_allow))

    train_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=num_frames,
        transform=train_transform,
        sample_list=train_samples,
        time_reversal_prob=time_reversal_prob,
        time_reversal_perm=tr_perm,
        time_reversal_allow_mask=tr_allow,
    )
    val_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=val_samples,
    )

    # Class-balanced sampler (opt-in). When the policy is ``"none"`` we keep
    # ``shuffle=True`` so the legacy run path is byte-for-byte identical.
    cb_sampler_policy = str(cfg.training.get("class_balance_sampler", "none")).lower()
    sampler: WeightedRandomSampler | None = None
    if cb_sampler_policy != "none":
        sample_weights = compute_sample_weights(
            samples=train_samples,
            num_classes=int(cfg.num_classes),
            policy=cb_sampler_policy,
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(train_samples),
            replacement=True,
        )
        n_train_by_class = sorted({int(label) for _, label in train_samples})
        print(
            f"[class-balance] sampler={cb_sampler_policy!r}; "
            f"trained classes={len(n_train_by_class)}, "
            f"effective per-class draw equalised by {cb_sampler_policy}."
        )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
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

    # Optional: warm-start the trunk from a self-supervised checkpoint produced
    # by ``pretrain_ssl.py``. Track-A safe: the SSL trunk was trained from
    # random init on the *provided* unlabeled frames only, so loading it here
    # introduces no external knowledge. The hook is opt-in (``model.init_from``
    # null by default) and the load is non-strict so slight architectural
    # mismatches (e.g. extra BN buffers) don't fail the supervised run.
    init_from = cfg.model.get("init_from") if hasattr(cfg.model, "get") else None
    if init_from:
        init_path = Path(str(init_from)).resolve()
        if not init_path.is_file():
            raise SystemExit(f"model.init_from points to a missing file: {init_path}")
        payload = torch.load(init_path, map_location=device, weights_only=False)
        trunk_state = payload.get("trunk_state_dict") if isinstance(payload, dict) else None
        if trunk_state is None:
            raise SystemExit(
                f"{init_path} does not contain a 'trunk_state_dict' key; "
                "expected an SSL checkpoint produced by pretrain_ssl.py."
            )
        prefixed = _ssl_trunk_to_supervised_keys(trunk_state)
        missing, unexpected = model.load_state_dict(prefixed, strict=False)
        # ``missing`` will include classifier / attn_pool keys -- expected.
        backbone_missing = [k for k in missing if k.startswith("backbone.")]
        print(
            f"[init_from] loaded {len(prefixed)} trunk tensors from {init_path}. "
            f"backbone-missing={len(backbone_missing)}, unexpected={len(unexpected)}"
        )
        if backbone_missing:
            # If the SSL trunk doesn't cover every backbone key, we want to know.
            print(f"[init_from] backbone keys NOT covered by SSL: {backbone_missing[:8]}...")

    # Class-balanced cross-entropy weights (opt-in). Composes with
    # label-smoothing and video-mixing: the same tensor is passed both to
    # ``nn.CrossEntropyLoss(weight=...)`` (plain-CE path) and to the
    # soft-target CE used by the mixing augmentations.
    cb_loss_policy = str(cfg.training.get("class_balance_loss", "none")).lower()
    cb_loss_beta = float(cfg.training.get("class_balance_beta", 0.999))
    class_weights = compute_class_weights(
        samples=train_samples,
        num_classes=int(cfg.num_classes),
        policy=cb_loss_policy,
        beta=cb_loss_beta,
    )
    if class_weights is not None:
        class_weights = class_weights.to(device)
        masked = (class_weights == 0).sum().item()
        print(
            f"[class-balance] loss policy={cb_loss_policy!r}, beta={cb_loss_beta}; "
            f"min={float(class_weights[class_weights > 0].min()):.3f}, "
            f"max={float(class_weights.max()):.3f}, "
            f"never-trained classes set to 0: {int(masked)}."
        )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
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
    videomix_mode = str(cfg.training.get("videomix_mode", "cube_cutmix"))
    log_interval_steps = int(cfg.training.get("log_interval_steps", 0))
    early_stopping_enabled = bool(cfg.training.get("early_stopping_enabled", False))
    early_stopping_patience = int(cfg.training.get("early_stopping_patience", 10))
    early_stopping_min_delta = float(cfg.training.get("early_stopping_min_delta", 0.0))

    amp_enabled = bool(cfg.training.get("amp", False)) and device.type == "cuda"
    scaler: torch.amp.GradScaler | None = (
        torch.amp.GradScaler("cuda", enabled=True) if amp_enabled else None
    )
    if amp_enabled:
        print("[amp] mixed-precision (fp16 + GradScaler) enabled.")

    # EMA (Exponential Moving Average) of model weights, à la PyTorch's
    # ``AveragedModel(use_buffers=True)``. When enabled, every optimizer step
    # we update the EMA copy, and at each epoch boundary we evaluate *both*
    # the live model and the EMA model. The checkpoint stores whichever
    # achieved the best val_top1; the corresponding state_dict is recorded so
    # ``submit.py`` always loads the right weights without code changes.
    ema_enabled = bool(cfg.training.get("ema_enabled", False))
    ema_decay = float(cfg.training.get("ema_decay", 0.999))
    ema_model: torch.optim.swa_utils.AveragedModel | None = None
    if ema_enabled:

        def _ema_avg_fn(
            avg_param: torch.Tensor, model_param: torch.Tensor, _num_averaged: int
        ) -> torch.Tensor:
            return ema_decay * avg_param + (1.0 - ema_decay) * model_param

        ema_model = torch.optim.swa_utils.AveragedModel(
            model, avg_fn=_ema_avg_fn, use_buffers=True
        ).to(device)
        print(f"[ema] enabled (decay={ema_decay}); will eval both live and EMA each epoch.")

    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    best_top1 = -1.0
    best_path: Path | None = None
    epochs_without_improvement = 0
    start_epoch = 0

    resume_from = cfg.training.get("resume_from")
    if resume_from:
        resume_path = Path(str(resume_from)).resolve()
        print(f"Resuming from checkpoint: {resume_path}")
        payload = load_checkpoint(resume_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        extra = payload.get("extra") or {}
        start_epoch = int(extra.get("epoch", 0))
        if "val_top1" in extra:
            best_top1 = float(extra["val_top1"])
        best_path = resume_path
        print(
            f"  Resumed at epoch {start_epoch}/{int(cfg.training.epochs)}; "
            f"best val top1 so far = {best_top1:.4f}"
        )

        # Optimizer / scheduler / scaler are stored inside ``extra`` so the
        # checkpoint schema (``schema_version=1``) stays untouched. Older
        # checkpoints that don't carry these keys fall back to the legacy
        # cosine fast-forward path so we keep resuming runs created before
        # this change.
        opt_state = extra.get("optimizer_state_dict")
        if opt_state is not None:
            try:
                optimizer.load_state_dict(opt_state)
                print("  Optimizer state restored from checkpoint.")
            except Exception as exc:
                print(f"  [warn] could not restore optimizer state: {exc}")

        sched_state = extra.get("scheduler_state_dict")
        if cosine_scheduler is not None and sched_state is not None:
            try:
                cosine_scheduler.load_state_dict(sched_state)
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"  Cosine scheduler state restored; resumed LR={current_lr:.6g}")
            except Exception as exc:
                print(f"  [warn] could not restore scheduler state: {exc}")
                sched_state = None  # trigger fast-forward fallback below

        if cosine_scheduler is not None and sched_state is None:
            ff_steps = max(0, start_epoch - warmup_epochs)
            for _ in range(ff_steps):
                cosine_scheduler.step()
            if ff_steps > 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  Cosine scheduler fast-forwarded {ff_steps} step(s); "
                    f"resumed LR={current_lr:.6g}"
                )

        scaler_state = extra.get("scaler_state_dict")
        if scaler is not None and scaler_state is not None:
            try:
                scaler.load_state_dict(scaler_state)
                print("  GradScaler state restored from checkpoint.")
            except Exception as exc:
                print(f"  [warn] could not restore GradScaler state: {exc}")

    try:
        for epoch in range(start_epoch, int(cfg.training.epochs)):
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
                videomix_mode=videomix_mode,
                log_interval_steps=log_interval_steps,
                scaler=scaler,
                ema_model=ema_model,
                class_weights=class_weights,
            )
            val_stats: EpochStats = evaluate_epoch(
                model, val_loader, loss_fn, device, amp_enabled=amp_enabled
            )
            ema_stats: EpochStats | None = None
            if ema_model is not None:
                ema_stats = evaluate_epoch(
                    ema_model, val_loader, loss_fn, device, amp_enabled=amp_enabled
                )
                print(
                    f"Epoch {epoch + 1}/{cfg.training.epochs} | "
                    f"train loss {train_stats.loss:.4f} top1 {train_stats.top1:.4f} | "
                    f"val loss {val_stats.loss:.4f} top1 {val_stats.top1:.4f} "
                    f"top5 {val_stats.top5:.4f} | "
                    f"ema val top1 {ema_stats.top1:.4f} top5 {ema_stats.top5:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{cfg.training.epochs} | "
                    f"train loss {train_stats.loss:.4f} top1 {train_stats.top1:.4f} | "
                    f"val loss {val_stats.loss:.4f} top1 {val_stats.top1:.4f} "
                    f"top5 {val_stats.top5:.4f}"
                )

            # Pick the better of (live, EMA) for checkpointing. The chosen
            # state_dict is saved as ``model_state_dict`` so ``submit.py`` /
            # ``evaluate.py`` keep working without any "is this EMA?" branching.
            if ema_stats is not None and ema_stats.top1 > val_stats.top1:
                ckpt_stats, ckpt_module, ckpt_kind = ema_stats, ema_model, "ema"
            else:
                ckpt_stats, ckpt_module, ckpt_kind = val_stats, model, "live"

            if ckpt_stats.top1 > (best_top1 + early_stopping_min_delta):
                best_top1 = ckpt_stats.top1
                epochs_without_improvement = 0
                ckpt_extra: dict[str, Any] = {
                    "val_top1": ckpt_stats.top1,
                    "val_top5": ckpt_stats.top5,
                    "val_loss": ckpt_stats.loss,
                    "epoch": epoch + 1,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "trained_class_indices": trained_class_indices,
                    "checkpoint_kind": ckpt_kind,
                }
                if cosine_scheduler is not None:
                    ckpt_extra["scheduler_state_dict"] = cosine_scheduler.state_dict()
                if scaler is not None:
                    ckpt_extra["scaler_state_dict"] = scaler.state_dict()
                # ``AveragedModel`` wraps the underlying network in a ``.module``
                # attribute. Saving ``ema_model.module`` instead of ``ema_model``
                # keeps the state_dict shape identical to the live model so
                # downstream loading (which reconstructs via :func:`build_model`)
                # works without special cases.
                module_to_save = (
                    ckpt_module.module
                    if isinstance(ckpt_module, torch.optim.swa_utils.AveragedModel)
                    else ckpt_module
                )
                best_path = save_checkpoint(
                    checkpoint_path,
                    module_to_save,
                    cfg,
                    extra=ckpt_extra,
                )
                print(
                    f"  Saved new best checkpoint ({ckpt_kind}): "
                    f"{best_path} (val top1={ckpt_stats.top1:.4f})"
                )
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
