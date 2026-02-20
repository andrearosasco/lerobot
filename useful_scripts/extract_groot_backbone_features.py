#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors
from lerobot.utils.utils import get_safe_torch_device


DEFAULT_GROOT_BACKBONE = "nvidia/GR00T-N1.5-3B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one Groot-backbone feature per dataset frame and save them "
            "as a NumPy array of shape (N_frames, feature_dim)."
        )
    )
    parser.add_argument("--dataset-repo-id", required=True, help="LeRobot dataset repo id (e.g. user/my_dataset)")
    parser.add_argument(
        "--output-dir",
        default="dataset_analysis",
        help="Directory where the extracted feature file is saved (default: dataset_analysis)",
    )
    parser.add_argument("--dataset-root", default=None, help="Optional local dataset root")
    parser.add_argument("--revision", default=None, help="Optional dataset revision/tag")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of episode indices to process",
    )
    parser.add_argument(
        "--pooling",
        choices=["mean", "first"],
        default="mean",
        help="How to reduce token-level backbone features to one vector per frame",
    )
    parser.add_argument("--device", default="cuda", help="Torch device (e.g. cuda, cpu, mps)")
    return parser.parse_args()


def reduce_tokens(features: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "mean":
        return features.mean(dim=1)
    return features[:, 0, :]


@torch.no_grad()
def extract_batch_features(
    batch: dict,
    preprocessor,
    policy: GrootPolicy,
    pooling: str,
    device_type: str,
    use_autocast_bf16: bool,
) -> np.ndarray:
    processed = preprocessor(batch)

    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_autocast_bf16):
        backbone_inputs = policy._groot_model.backbone.prepare_input(processed)
        backbone_outputs = policy._groot_model.backbone(backbone_inputs)

    token_features = backbone_outputs["backbone_features"]
    frame_features = reduce_tokens(token_features, pooling=pooling)
    return frame_features.float().cpu().numpy()


def main() -> None:
    args = parse_args()

    device = get_safe_torch_device(args.device)
    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        revision=args.revision,
        episodes=args.episodes,
    )

    if len(dataset.meta.camera_keys) == 0:
        raise ValueError(f"Dataset '{args.dataset_repo_id}' has no camera/image features.")

    policy = GrootPolicy.from_pretrained(DEFAULT_GROOT_BACKBONE, strict=False)
    policy.to(device)
    policy.eval()
    policy.config.device = str(device)

    preprocessor, _ = make_groot_pre_post_processors(config=policy.config, dataset_stats=None)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    total_frames = len(dataset)
    all_features: np.ndarray | None = None
    write_idx = 0
    use_autocast_bf16 = device.type == "cuda" and bool(policy.config.use_bf16)

    pbar = tqdm(total=total_frames, desc="Extracting features", unit="frame")

    for batch in dataloader:
        batch_features = extract_batch_features(
            batch=batch,
            preprocessor=preprocessor,
            policy=policy,
            pooling=args.pooling,
            device_type=device.type,
            use_autocast_bf16=use_autocast_bf16,
        )

        if all_features is None:
            feature_dim = batch_features.shape[1]
            all_features = np.empty((total_frames, feature_dim), dtype=np.float32)

        bsz = batch_features.shape[0]
        all_features[write_idx : write_idx + bsz] = batch_features
        write_idx += bsz
        pbar.update(bsz)

    pbar.close()

    if all_features is None:
        raise RuntimeError("No features were extracted.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = args.dataset_repo_id.replace("/", "__")
    out_path = output_dir / f"{dataset_name}_groot_backbone_features.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, all_features[:write_idx])

    print(f"Saved features with shape {all_features[:write_idx].shape} to {out_path}")


if __name__ == "__main__":
    main()
