#!/usr/bin/env python

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import DATA_DIR, EPISODES_DIR, STATS_PATH, INFO_PATH, DEFAULT_TASKS_PATH


def _parquet_num_rows(path: Path) -> int:
    return pq.read_metadata(path).num_rows


def _size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2)


def _list_parquet_files(root: Path) -> list[Path]:
    return sorted(root.glob("chunk-*/file-*.parquet"))


def _summarize_parquet_dir(label: str, root: Path) -> tuple[int, float, int]:
    files = _list_parquet_files(root)
    total_rows = 0
    total_size = 0.0
    for fpath in files:
        total_rows += _parquet_num_rows(fpath)
        total_size += _size_mb(fpath)
    print(f"{label}: {len(files)} files, {total_rows} rows, {total_size:.2f} MB")
    return total_rows, total_size, len(files)


def _check_frame_index_consistency(meta: LeRobotDatasetMetadata) -> None:
    data_root = meta.root / DATA_DIR
    data_files = _list_parquet_files(data_root)
    if not data_files:
        print("Frame index check: no data parquet files found")
        return

    episode_stats: dict[int, dict] = {}
    missing_columns = set()

    for fpath in data_files:
        pf = pq.ParquetFile(fpath)
        cols = set(pf.schema.names)
        required = {"episode_index", "frame_index"}
        if not required.issubset(cols):
            missing_columns.update(required - cols)
            continue

        for batch in pf.iter_batches(columns=["episode_index", "frame_index"]):
            episodes = batch.column(0).to_numpy()
            frames = batch.column(1).to_numpy()
            for ep_idx, frame_idx in zip(episodes, frames, strict=False):
                ep_idx = int(ep_idx)
                frame_idx = int(frame_idx)
                if ep_idx not in episode_stats:
                    episode_stats[ep_idx] = {
                        "count": 0,
                        "min": frame_idx,
                        "max": frame_idx,
                    }
                stats = episode_stats[ep_idx]
                stats["count"] += 1
                stats["min"] = min(stats["min"], frame_idx)
                stats["max"] = max(stats["max"], frame_idx)

    if missing_columns:
        print(f"Frame index check: missing columns in data parquet: {sorted(missing_columns)}")
        return

    if not episode_stats:
        print("Frame index check: no rows found")
        return

    print("Frame index check:")
    for ep_idx, row in meta.episodes.to_pandas().iterrows():
        episode_index = int(row["episode_index"])
        expected_len = int(row["length"]) if "length" in row else None
        stats = episode_stats.get(episode_index)
        if stats is None:
            print(f"  Episode {episode_index}: missing in data parquet")
            continue
        min_idx = stats["min"]
        max_idx = stats["max"]
        count = stats["count"]
        expected_max = expected_len - 1 if expected_len is not None else None
        issues = []
        if min_idx != 0:
            issues.append(f"min_frame_index={min_idx}")
        if expected_max is not None and max_idx != expected_max:
            issues.append(f"max_frame_index={max_idx} (expected {expected_max})")
        if expected_len is not None and count != expected_len:
            issues.append(f"count={count} (expected {expected_len})")

        if issues:
            print(f"  Episode {episode_index}: " + ", ".join(issues))

    extra_eps = sorted(set(episode_stats.keys()) - set(meta.episodes["episode_index"]))
    if extra_eps:
        print(f"  Extra episode indices in data parquet: {extra_eps}")


def _print_feature_summary(meta: LeRobotDatasetMetadata) -> None:
    print("Features:")
    for name, spec in meta.features.items():
        dtype = spec.get("dtype")
        shape = spec.get("shape")
        print(f"  - {name}: dtype={dtype}, shape={shape}")


def _print_episode_summary(meta: LeRobotDatasetMetadata) -> None:
    episodes = meta.episodes.to_pandas()
    print(f"Episode table rows: {len(episodes)}")
    print(f"Episode table columns: {list(episodes.columns)}")

    if "episode_index" in episodes.columns:
        dup_mask = episodes.duplicated(subset=["episode_index"], keep=False)
        dup_indices = sorted(episodes.loc[dup_mask, "episode_index"].unique().tolist())
        if dup_indices:
            print(f"Duplicate episode_index values: {dup_indices}")

    if "length" in episodes.columns:
        lengths = episodes["length"].tolist()
    else:
        lengths = None

    for _, row in episodes.iterrows():
        ep_idx = int(row["episode_index"]) if "episode_index" in episodes.columns else None
        from_idx = row.get("dataset_from_index")
        to_idx = row.get("dataset_to_index")
        derived_len = None
        if from_idx is not None and to_idx is not None:
            derived_len = int(to_idx) - int(from_idx)
        length = int(row["length"]) if "length" in episodes.columns else derived_len
        tasks = row.get("tasks")
        data_chunk = row.get("data/chunk_index")
        data_file = row.get("data/file_index")
        data_file_str = None
        if data_chunk is not None and data_file is not None:
            data_file_str = f"chunk-{int(data_chunk):03d}/file-{int(data_file):03d}.parquet"
        mismatch = ""
        if length is not None and derived_len is not None and length != derived_len:
            mismatch = " (length mismatch)"
        print(
            f"Episode {ep_idx}: length={length}, from={from_idx}, to={to_idx}, tasks={tasks}, "
            f"data_file={data_file_str}{mismatch}"
        )

    if lengths:
        print(f"Episode lengths: min={min(lengths)}, max={max(lengths)}, sum={sum(lengths)}")


def _print_meta_file_sizes(root: Path) -> None:
    info_path = root / INFO_PATH
    stats_path = root / STATS_PATH
    tasks_path = root / DEFAULT_TASKS_PATH

    if info_path.exists():
        print(f"meta/info.json: {_size_mb(info_path):.2f} MB")
    else:
        print("meta/info.json: missing")

    if stats_path.exists():
        print(f"meta/stats.json: {_size_mb(stats_path):.2f} MB")
    else:
        print("meta/stats.json: missing")

    if tasks_path.exists():
        print(f"meta/tasks.parquet: {_size_mb(tasks_path):.2f} MB")
    else:
        print("meta/tasks.parquet: missing")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check LeRobot dataset metadata and frame stats.")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face dataset repo id.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for dataset cache (defaults to ~/.cache/huggingface/lerobot).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download data files for the dataset (default: metadata only).",
    )
    parser.add_argument(
        "--download-videos",
        action="store_true",
        help="Download video files too (only if --download is set).",
    )
    parser.add_argument(
        "--check-frame-index",
        action="store_true",
        help="Validate frame_index against episode lengths (requires --download).",
    )
    args = parser.parse_args()

    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)

    print(f"Repo: {args.repo_id}")
    print(f"Root: {meta.root}")
    print(f"Codebase version: {meta.info.get('codebase_version')}")
    print(f"FPS: {meta.fps}")
    print(f"Robot type: {meta.robot_type}")
    print(f"Total episodes: {meta.total_episodes}")
    print(f"Total frames: {meta.total_frames}")
    print(f"Total tasks: {meta.total_tasks}")

    _print_feature_summary(meta)

    if meta.tasks is not None:
        print(f"Tasks table rows: {len(meta.tasks)}")

    if meta.stats is None:
        print("Stats: missing")
    else:
        print(f"Stats keys: {len(meta.stats.keys())}")

    print("\nMeta file sizes:")
    _print_meta_file_sizes(meta.root)

    print("\nEpisode metadata:")
    _print_episode_summary(meta)

    print("\nParquet stats:")
    episodes_rows, _, _ = _summarize_parquet_dir("meta/episodes", meta.root / EPISODES_DIR)

    if args.download:
        dataset = LeRobotDataset(args.repo_id, root=args.root)
        dataset.download(download_videos=args.download_videos)

        data_rows, _, _ = _summarize_parquet_dir("data", meta.root / DATA_DIR)
        print(f"Total frames (data parquet rows): {data_rows}")
        print(f"Total frames (meta): {meta.total_frames}")
        print(f"Total episodes (episodes parquet rows): {episodes_rows}")
        print(f"Dataset frames (hf_dataset): {dataset.num_frames}")

        if args.check_frame_index:
            _check_frame_index_consistency(meta)
    else:
        print("Skipping data download. Re-run with --download to validate data parquet rows.")


if __name__ == "__main__":
    main()
