#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


@dataclass
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    m_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    m_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    m_test: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a probe to predict human pose vectors from extracted Groot features and report accuracy."
        )
    )
    parser.add_argument("--features-path", required=True, help="Path to features .npy (shape: N x D)")
    parser.add_argument(
        "--poses-path",
        required=True,
        help="Path to pose .npy (object array with pose vectors or None, length N)",
    )
    parser.add_argument("--output-dir", default="dataset_analysis", help="Output directory")
    parser.add_argument("--model-name", default="pose_probe", help="Prefix name for saved outputs")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=0, help="0 means linear probe")
    parser.add_argument(
        "--vis-threshold",
        type=float,
        default=0.2,
        help="Visibility/confidence threshold below which keypoint coordinates are masked out",
    )
    parser.add_argument(
        "--mask-low-vis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mask low-visibility joints during training/evaluation (default: enabled)",
    )
    parser.add_argument(
        "--target-mode",
        choices=["all", "xy"],
        default="xy",
        help="Loss target dimensions: 'xy' focuses localization, 'all' uses all outputs",
    )
    parser.add_argument(
        "--xy-vis-weighted",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Weight xy loss by visibility confidence when target-mode=xy",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    req = requested.lower()
    if req == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if req == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_aligned_data(features_path: str, poses_path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    features = np.load(features_path)
    poses_raw = np.load(poses_path, allow_pickle=True)

    if features.ndim != 2:
        raise ValueError(f"Expected features shape (N, D), got {features.shape}")
    if len(poses_raw) != len(features):
        raise ValueError(
            f"Features and poses must have same length. Got features={len(features)}, poses={len(poses_raw)}"
        )

    first_pose = None
    for p in poses_raw:
        if p is not None:
            first_pose = np.asarray(p, dtype=np.float32)
            break

    if first_pose is None:
        raise ValueError("All poses are None. Cannot train probe.")

    pose_dim = int(first_pose.shape[0])
    valid_mask = np.array([p is not None for p in poses_raw], dtype=bool)

    x = features[valid_mask].astype(np.float32)
    y = np.stack([np.asarray(p, dtype=np.float32) for p in poses_raw[valid_mask]], axis=0)

    meta = {
        "total_frames": int(len(features)),
        "valid_pose_frames": int(valid_mask.sum()),
        "missing_pose_frames": int((~valid_mask).sum()),
        "feature_dim": int(features.shape[1]),
        "pose_dim": pose_dim,
    }
    return x, y, meta


def split_dataset(x: np.ndarray, y: np.ndarray, val_ratio: float, test_ratio: float, seed: int) -> SplitData:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("Require 0 <= val_ratio, test_ratio and val_ratio + test_ratio < 1")

    n = len(x)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_train = n - n_val - n_test

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    return SplitData(
        x_train=x[train_idx],
        y_train=y[train_idx],
        m_train=None,
        x_val=x[val_idx],
        y_val=y[val_idx],
        m_val=None,
        x_test=x[test_idx],
        y_test=y[test_idx],
        m_test=None,
    )


def standardize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (train - mean) / std, (other - mean) / std, (mean.squeeze(0), std.squeeze(0))


def build_supervision_mask(y: np.ndarray, vis_threshold: float, mask_low_vis: bool) -> np.ndarray:
    mask = np.ones_like(y, dtype=np.float32)
    if (not mask_low_vis) or (y.ndim != 2) or (y.shape[1] % 4 != 0):
        return mask

    vis = y[:, 3::4]
    low_vis = vis < vis_threshold

    # Coordinates x,y,z are masked when visibility is low; visibility dim stays supervised.
    for coord_offset in (0, 1, 2):
        coord_slice = slice(coord_offset, y.shape[1], 4)
        coord_mask = mask[:, coord_slice]
        coord_mask[low_vis] = 0.0
        mask[:, coord_slice] = coord_mask

    return mask


def build_loss_mask(
    y: np.ndarray,
    supervision_mask: np.ndarray,
    target_mode: str,
    xy_vis_weighted: bool,
) -> np.ndarray:
    mask = supervision_mask.astype(np.float32).copy()

    if (target_mode != "xy") or (y.ndim != 2) or (y.shape[1] % 4 != 0):
        return mask

    # Keep only x,y dimensions in loss; exclude z and visibility dims.
    mask[:, 2::4] = 0.0  # z
    mask[:, 3::4] = 0.0  # vis/conf target

    if xy_vis_weighted:
        vis = np.clip(y[:, 3::4], 0.0, 1.0)
        mask[:, 0::4] *= vis
        mask[:, 1::4] *= vis

    return mask


def standardize_with_mask(
    train: np.ndarray,
    other: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    mean = np.zeros((1, train.shape[1]), dtype=np.float32)
    std = np.ones((1, train.shape[1]), dtype=np.float32)

    for d in range(train.shape[1]):
        valid = train_mask[:, d] > 0.5
        if np.any(valid):
            vals = train[valid, d]
            mean[0, d] = float(vals.mean())
            s = float(vals.std())
            std[0, d] = 1.0 if s < 1e-8 else s

    train_std = (train - mean) / std
    other_std = (other - mean) / std
    return train_std, other_std, (mean.squeeze(0), std.squeeze(0))


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    err = (pred - target) ** 2
    weighted = err * mask
    denom = torch.clamp(mask.sum(), min=1.0)
    return weighted.sum() / denom


def compute_probe_metric_artifacts(
    model: nn.Module,
    in_dim: int,
    out_dim: int,
    x_std_scale: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    x_ref = torch.zeros((1, in_dim), dtype=torch.float32, device=device, requires_grad=True)
    y_ref = model(x_ref)

    jac_rows: list[np.ndarray] = []
    for i in range(out_dim):
        if x_ref.grad is not None:
            x_ref.grad.zero_()
        scalar = y_ref[0, i]
        scalar.backward(retain_graph=True)
        jac_rows.append(x_ref.grad.detach().cpu().numpy().copy().reshape(-1))

    # Jacobian wrt standardized input z = (x - mean) / std
    jac_std = np.stack(jac_rows, axis=0).astype(np.float32)  # (out_dim, in_dim)

    # Convert to raw feature space x: dy/dx = dy/dz * dz/dx, with dz/dx = 1/std
    inv_std = (1.0 / np.clip(x_std_scale.astype(np.float32), 1e-8, None)).reshape(1, -1)
    jac_raw = jac_std * inv_std

    metric_matrix = jac_raw.T @ jac_raw  # (in_dim, in_dim), PSD
    metric_vector = np.linalg.norm(jac_raw, axis=0).astype(np.float32)  # (in_dim,)
    return metric_vector, metric_matrix.astype(np.float32)


class PoseProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    diff = y_pred - y_true
    m = mask.astype(np.float32)
    denom = float(np.clip(m.sum(), 1.0, None))

    mse = float(np.sum((diff**2) * m) / denom)
    mae = float(np.sum(np.abs(diff) * m) / denom)

    y_mean = np.sum(y_true * m, axis=0, keepdims=True) / np.clip(np.sum(m, axis=0, keepdims=True), 1.0, None)
    ss_res = float(np.sum(((y_true - y_pred) ** 2) * m))
    ss_tot = float(np.sum(((y_true - y_mean) ** 2) * m))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))

    metrics = {
        "mse": mse,
        "mae": mae,
        "r2": r2,
    }

    if y_true.shape[1] % 4 == 0:
        xy_mask = np.zeros_like(mask, dtype=np.float32)
        xy_mask[:, 0::4] = mask[:, 0::4]
        xy_mask[:, 1::4] = mask[:, 1::4]

        xy_denom = float(np.clip(xy_mask.sum(), 1.0, None))
        xy_diff = y_pred - y_true
        mse_xy = float(np.sum((xy_diff**2) * xy_mask) / xy_denom)
        mae_xy = float(np.sum(np.abs(xy_diff) * xy_mask) / xy_denom)

        y_mean_xy = np.sum(y_true * xy_mask, axis=0, keepdims=True) / np.clip(
            np.sum(xy_mask, axis=0, keepdims=True), 1.0, None
        )
        ss_res_xy = float(np.sum(((y_true - y_pred) ** 2) * xy_mask))
        ss_tot_xy = float(np.sum(((y_true - y_mean_xy) ** 2) * xy_mask))
        r2_xy = float(1.0 - ss_res_xy / (ss_tot_xy + 1e-12))

        metrics.update(
            {
                "mse_xy": mse_xy,
                "mae_xy": mae_xy,
                "r2_xy": r2_xy,
            }
        )

    if y_true.shape[1] % 4 == 0:
        n_landmarks = y_true.shape[1] // 4
        true_lm = y_true.reshape(-1, n_landmarks, 4)
        pred_lm = y_pred.reshape(-1, n_landmarks, 4)

        xy_err = np.linalg.norm(pred_lm[..., :2] - true_lm[..., :2], axis=-1)
        vis = true_lm[..., 3] > 0.5
        if np.any(vis):
            vis_err = xy_err[vis]
            pck_05 = float(np.mean(vis_err < 0.05))
            pck_10 = float(np.mean(vis_err < 0.10))
            mean_xy_err = float(np.mean(vis_err))
        else:
            pck_05 = 0.0
            pck_10 = 0.0
            mean_xy_err = float(np.mean(xy_err))

        metrics.update(
            {
                "mean_xy_error": mean_xy_err,
                "pck@0.05": pck_05,
                "pck@0.10": pck_10,
            }
        )

    return metrics


def evaluate(
    model: nn.Module,
    x: np.ndarray,
    y_std: np.ndarray,
    loss_mask: np.ndarray,
    metric_mask: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    device: torch.device,
) -> tuple[float, np.ndarray, dict]:
    model.eval()
    with torch.no_grad():
        pred_std = model(torch.from_numpy(x).to(device)).cpu().numpy()

    denom = float(np.clip(loss_mask.sum(), 1.0, None))
    loss = float(np.sum(((pred_std - y_std) ** 2) * loss_mask) / denom)
    y_pred = pred_std * y_scale + y_mean
    y_true = y_std * y_scale + y_mean
    metrics = regression_metrics(y_true, y_pred, metric_mask)
    return loss, y_pred, metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    x, y, data_meta = load_aligned_data(args.features_path, args.poses_path)
    splits = split_dataset(x, y, args.val_ratio, args.test_ratio, args.seed)

    splits.m_train = build_supervision_mask(splits.y_train, args.vis_threshold, args.mask_low_vis)
    splits.m_val = build_supervision_mask(splits.y_val, args.vis_threshold, args.mask_low_vis)
    splits.m_test = build_supervision_mask(splits.y_test, args.vis_threshold, args.mask_low_vis)

    loss_m_train = build_loss_mask(splits.y_train, splits.m_train, args.target_mode, args.xy_vis_weighted)
    loss_m_val = build_loss_mask(splits.y_val, splits.m_val, args.target_mode, args.xy_vis_weighted)
    loss_m_test = build_loss_mask(splits.y_test, splits.m_test, args.target_mode, args.xy_vis_weighted)

    x_train_std, x_val_std, (x_mean, x_std) = standardize(splits.x_train, splits.x_val)
    _, x_test_std, _ = standardize(splits.x_train, splits.x_test)

    y_train_std, y_val_std, (y_mean, y_std) = standardize_with_mask(
        splits.y_train,
        splits.y_val,
        splits.m_train,
    )
    _, y_test_std, _ = standardize_with_mask(
        splits.y_train,
        splits.y_test,
        splits.m_train,
    )

    train_ds = TensorDataset(
        torch.from_numpy(x_train_std).float(),
        torch.from_numpy(y_train_std).float(),
        torch.from_numpy(loss_m_train).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = PoseProbe(in_dim=x.shape[1], out_dim=y.shape[1], hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training probe"):
        model.train()
        running = 0.0
        count = 0

        for xb, yb, mb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = masked_mse(pred, yb, mb)
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * xb.size(0)
            count += xb.size(0)

        train_loss = running / max(count, 1)
        val_loss, _, val_metrics = evaluate(
            model,
            x_val_std,
            y_val_std,
            loss_m_val,
            splits.m_val,
            y_mean,
            y_std,
            device,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch} train_mse={train_loss:.6f} val_mse={val_loss:.6f} "
                f"val_r2={val_metrics['r2']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    train_loss, _, train_metrics = evaluate(
        model,
        x_train_std,
        y_train_std,
        loss_m_train,
        splits.m_train,
        y_mean,
        y_std,
        device,
    )
    val_loss, _, val_metrics = evaluate(
        model,
        x_val_std,
        y_val_std,
        loss_m_val,
        splits.m_val,
        y_mean,
        y_std,
        device,
    )
    test_loss, test_pred, test_metrics = evaluate(
        model,
        x_test_std,
        y_test_std,
        loss_m_test,
        splits.m_test,
        y_mean,
        y_std,
        device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_dir / f"{args.model_name}.pt"

    metric_vector, metric_matrix = compute_probe_metric_artifacts(
        model=model,
        in_dim=x.shape[1],
        out_dim=y.shape[1],
        x_std_scale=x_std,
        device=device,
    )

    metric_vector_path = output_dir / f"{args.model_name}_probe_metric_vector.npy"
    metric_matrix_path = output_dir / f"{args.model_name}_probe_metric_matrix.npy"
    np.save(metric_vector_path, metric_vector)
    np.save(metric_matrix_path, metric_matrix)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "probe_metric_vector": metric_vector,
            "probe_metric_matrix": metric_matrix,
            "config": vars(args),
            "data_meta": data_meta,
        },
        ckpt_path,
    )

    pred_path = output_dir / f"{args.model_name}_test_predictions.npy"
    np.save(pred_path, test_pred)

    report = {
        "config": vars(args),
        "data": {
            **data_meta,
            "train_size": int(len(splits.x_train)),
            "val_size": int(len(splits.x_val)),
            "test_size": int(len(splits.x_test)),
            "mask_low_vis": bool(args.mask_low_vis),
            "vis_threshold": float(args.vis_threshold),
            "target_mode": args.target_mode,
            "xy_vis_weighted": bool(args.xy_vis_weighted),
        },
        "metrics": {
            "train": {"mse_std": train_loss, **train_metrics},
            "val": {"mse_std": val_loss, **val_metrics},
            "test": {"mse_std": test_loss, **test_metrics},
        },
        "artifacts": {
            "checkpoint": str(ckpt_path),
            "test_predictions": str(pred_path),
            "probe_metric_vector": str(metric_vector_path),
            "probe_metric_matrix": str(metric_matrix_path),
        },
    }

    metrics_path = output_dir / f"{args.model_name}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nFinal test metrics:")
    for key, value in report["metrics"]["test"].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    print(f"\nSaved checkpoint: {ckpt_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved test predictions: {pred_path}")
    print(f"Saved probe metric vector: {metric_vector_path}")
    print(f"Saved probe metric matrix: {metric_matrix_path}")


if __name__ == "__main__":
    main()
