#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from sklearn.manifold import TSNE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze extracted Groot features by computing a 2D t-SNE embedding "
            "and saving it as an image."
        )
    )
    parser.add_argument("--features-path", required=True, help="Path to extracted .npy features file")
    parser.add_argument(
        "--dataset-repo-id",
        default=None,
        help="Optional dataset repo id used only for default output filename.",
    )
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for t-SNE")
    parser.add_argument("--tsne-size", type=int, default=900, help="Size in pixels of the t-SNE canvas")
    parser.add_argument("--point-radius", type=int, default=1, help="Radius of each t-SNE point")
    parser.add_argument(
        "--point-color",
        type=int,
        nargs=3,
        metavar=("B", "G", "R"),
        default=(170, 170, 170),
        help="Point color in BGR format.",
    )
    parser.add_argument(
        "--probe-metric-vector",
        default=None,
        help="Optional .npy vector from train_pose_probe to reweight features before t-SNE",
    )
    parser.add_argument(
        "--normalize-weighted-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize weighted features per dimension before t-SNE (default: enabled)",
    )
    parser.add_argument(
        "--tsne-image-output",
        default=None,
        help="Optional .png output path. If not provided, a default file is created in dataset_analysis/.",
    )
    return parser.parse_args()


def compute_tsne(features: np.ndarray, perplexity: float, random_state: int) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(f"Expected features with shape (N, D), got {features.shape}")

    n_samples = features.shape[0]
    if n_samples < 3:
        raise ValueError("Need at least 3 feature vectors to compute t-SNE")

    max_valid_perplexity = max(2.0, float(n_samples - 1) / 3.0)
    perplexity = min(perplexity, max_valid_perplexity)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(features)


def tsne_to_pixels(points_2d: np.ndarray, size: int, pad: int = 24) -> np.ndarray:
    mins = points_2d.min(axis=0)
    maxs = points_2d.max(axis=0)
    span = np.maximum(maxs - mins, 1e-8)

    normalized = (points_2d - mins) / span
    xy = normalized * (size - 2 * pad) + pad
    xy = xy.astype(np.int32)
    xy[:, 1] = size - xy[:, 1]
    return xy


def build_tsne_background(
    tsne_pixels: np.ndarray,
    size: int,
    point_radius: int,
    point_color: tuple[int, int, int],
) -> np.ndarray:
    canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
    for x, y in tsne_pixels:
        cv2.circle(canvas, (int(x), int(y)), point_radius, point_color, -1, lineType=cv2.LINE_AA)

    cv2.putText(
        canvas,
        "t-SNE of extracted Groot features",
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
        lineType=cv2.LINE_AA,
    )
    return canvas


def resolve_tsne_output_path(
    dataset_repo_id: str | None,
    features_path: Path,
    tsne_image_output: str | None,
) -> Path:
    if tsne_image_output is not None:
        output_image_path = Path(tsne_image_output)
    else:
        output_dir = Path("dataset_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_name = dataset_repo_id.replace("/", "__") if dataset_repo_id else features_path.stem
        output_image_path = output_dir / f"{dataset_name}_groot_tsne.png"
    return output_image_path


def save_tsne_image(tsne_background: np.ndarray, output_image_path: Path) -> None:
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_image_path), tsne_background):
        raise RuntimeError(f"Failed to save t-SNE image to {output_image_path}")
    print(f"Saved t-SNE image to: {output_image_path}")


def apply_probe_metric_vector(
    features: np.ndarray,
    vector_path: str,
    normalize_weighted_features: bool,
) -> np.ndarray:
    vector = np.load(vector_path)
    if vector.ndim != 1:
        raise ValueError(f"Expected probe metric vector with shape (D,), got {vector.shape}")
    if vector.shape[0] != features.shape[1]:
        raise ValueError(
            f"Probe metric vector dim ({vector.shape[0]}) does not match feature dim ({features.shape[1]})."
        )

    weighted = features * vector.reshape(1, -1)
    if not normalize_weighted_features:
        return weighted

    mean = weighted.mean(axis=0, keepdims=True)
    std = weighted.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (weighted - mean) / std


def main() -> None:
    args = parse_args()

    features_path = Path(args.features_path)
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    features = np.load(features_path)
    print(f"Loaded features: shape={features.shape} from {features_path}")

    if args.probe_metric_vector is not None:
        features = apply_probe_metric_vector(
            features=features,
            vector_path=args.probe_metric_vector,
            normalize_weighted_features=args.normalize_weighted_features,
        )
        print(f"Applied probe metric vector from: {args.probe_metric_vector}")

    print("Computing t-SNE...")
    points_2d = compute_tsne(features, perplexity=args.perplexity, random_state=args.random_state)

    tsne_pixels = tsne_to_pixels(points_2d, size=args.tsne_size)
    tsne_background = build_tsne_background(
        tsne_pixels,
        size=args.tsne_size,
        point_radius=args.point_radius,
        point_color=tuple(args.point_color),
    )

    output_image_path = resolve_tsne_output_path(args.dataset_repo_id, features_path, args.tsne_image_output)
    save_tsne_image(tsne_background, output_image_path)


if __name__ == "__main__":
    main()
