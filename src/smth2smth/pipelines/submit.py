"""Submission pipeline.

Loads a trained checkpoint and runs inference on the test split, writing
``video_name,predicted_class`` to the configured output CSV.

Run from the repo root::

    PYTHONPATH=src uv run python -m smth2smth.pipelines.submit \\
        training.checkpoint_path=best_model.pt
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from smth2smth.pipelines.train import CONFIGS_DIR, _resolve_device
from smth2smth.shared.data import VideoFrameDataset, build_transforms
from smth2smth.shared.engine import predict_argmax
from smth2smth.shared.io.checkpoints import cfg_from_checkpoint, load_checkpoint
from smth2smth.shared.io.submission import (
    discover_all_test_videos,
    load_manifest_video_names,
    resolve_video_dirs,
    write_submission_csv,
)
from smth2smth.shared.models import build_model
from smth2smth.shared.utils import set_seed


def _resolve_test_videos(
    test_root: Path, manifest_path: Path | None
) -> tuple[list[str], list[Path]]:
    """Pick test video order from manifest if provided, otherwise from disk."""
    if manifest_path is not None:
        names = load_manifest_video_names(manifest_path)
        dirs = resolve_video_dirs(test_root, names)
        return names, dirs
    return discover_all_test_videos(test_root)


def run(cfg: DictConfig) -> Path:
    """Generate the submission CSV from a checkpoint.

    Args:
        cfg: Hydra configuration. Must have ``cfg.training.checkpoint_path``,
            ``cfg.dataset.test_dir`` and ``cfg.dataset.submission_output``.

    Returns:
        Path to the written CSV.
    """
    print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.seed))
    device = _resolve_device(str(cfg.training.device))

    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    saved_cfg = cfg_from_checkpoint(checkpoint)

    model = build_model(saved_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    use_imagenet_norm = bool(saved_cfg.model.pretrained)
    augment_cfg = saved_cfg.get("augment") if hasattr(saved_cfg, "get") else None
    eval_transform = build_transforms(
        image_size=int(cfg.dataset.image_size),
        is_training=False,
        use_imagenet_norm=use_imagenet_norm,
        augment=augment_cfg,
    )
    num_frames = (
        int(saved_cfg.dataset.num_frames) if "dataset" in saved_cfg else int(cfg.dataset.num_frames)
    )

    test_root = Path(cfg.dataset.test_dir).resolve()
    manifest_cfg = cfg.dataset.get("test_manifest")
    manifest_path = Path(str(manifest_cfg)).resolve() if manifest_cfg else None

    print(f"Indexing video folders under: {test_root}")
    video_names, video_dirs = _resolve_test_videos(test_root, manifest_path)
    print(f"Found {len(video_names)} test videos.")

    sample_list = [(p, 0) for p in video_dirs]
    dataset = VideoFrameDataset(
        root_dir=test_root,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=sample_list,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    predictions, _ = predict_argmax(model, loader, device)
    if len(predictions) != len(video_names):
        raise RuntimeError(f"Prediction count {len(predictions)} != video count {len(video_names)}")

    output_path = Path(cfg.dataset.submission_output).resolve()
    csv_path = write_submission_csv(output_path, video_names, predictions)
    print(f"Wrote {len(predictions)} rows to {csv_path}")
    return csv_path


@hydra.main(version_base=None, config_path=CONFIGS_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra CLI entrypoint."""
    run(cfg)


if __name__ == "__main__":
    main()
