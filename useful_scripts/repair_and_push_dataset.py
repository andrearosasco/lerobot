#!/usr/bin/env python

import argparse
import json
import logging
import re
import shutil
from pathlib import Path

import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DATA_DIR,
    DEFAULT_EPISODES_PATH,
    DEFAULT_TASKS_PATH,
    EPISODES_DIR,
    INFO_PATH,
    load_tasks,
    write_episodes,
    write_info,
)
from lerobot.utils.constants import HF_LEROBOT_HOME


CHUNK_FILE_RE = re.compile(r"chunk-(\d+)/file-(\d+)\.parquet")


def _list_parquet_files(root: Path) -> list[Path]:
    return sorted(root.glob("chunk-*/file-*.parquet"))


def _parse_chunk_file(path: Path) -> tuple[int, int]:
    match = CHUNK_FILE_RE.search(str(path))
    if not match:
        raise ValueError(f"Could not parse chunk/file index from {path}")
    return int(match.group(1)), int(match.group(2))


def _collect_episode_stats(data_files: list[Path]) -> dict[int, dict]:
    stats: dict[int, dict] = {}
    missing_columns = set()

    for fpath in data_files:
        pf = pq.ParquetFile(fpath)
        cols = set(pf.schema.names)
        required = {"episode_index", "frame_index", "index"}
        if not required.issubset(cols):
            missing_columns.update(required - cols)
            continue

        for batch in pf.iter_batches(columns=["episode_index", "frame_index", "index"]):
            ep_col = batch.column(0).to_numpy()
            frame_col = batch.column(1).to_numpy()
            index_col = batch.column(2).to_numpy()

            for ep_idx, frame_idx, global_idx in zip(ep_col, frame_col, index_col, strict=False):
                ep_idx = int(ep_idx)
                frame_idx = int(frame_idx)
                global_idx = int(global_idx)

                if ep_idx not in stats:
                    stats[ep_idx] = {
                        "count": 0,
                        "min_frame_index": frame_idx,
                        "max_frame_index": frame_idx,
                        "min_index": global_idx,
                        "max_index": global_idx,
                    }

                ep_stats = stats[ep_idx]
                ep_stats["count"] += 1
                ep_stats["min_frame_index"] = min(ep_stats["min_frame_index"], frame_idx)
                ep_stats["max_frame_index"] = max(ep_stats["max_frame_index"], frame_idx)
                ep_stats["min_index"] = min(ep_stats["min_index"], global_idx)
                ep_stats["max_index"] = max(ep_stats["max_index"], global_idx)

    if missing_columns:
        missing = sorted(missing_columns)
        raise ValueError(f"Missing required columns in data parquet: {missing}")

    return stats


def _collect_episode_file_mapping(data_files: list[Path]) -> tuple[dict[int, tuple[int, int]], list[int]]:
    mapping: dict[int, tuple[int, int]] = {}
    conflicts: list[int] = []

    for fpath in data_files:
        pf = pq.ParquetFile(fpath)
        if "episode_index" not in pf.schema.names:
            raise ValueError(f"Missing episode_index column in {fpath}")

        chunk_idx, file_idx = _parse_chunk_file(fpath)
        seen = set()
        for batch in pf.iter_batches(columns=["episode_index"]):
            for ep_idx in batch.column(0).to_numpy():
                ep_idx = int(ep_idx)
                if ep_idx in seen:
                    continue
                seen.add(ep_idx)
                if ep_idx in mapping and mapping[ep_idx] != (chunk_idx, file_idx):
                    conflicts.append(ep_idx)
                else:
                    mapping[ep_idx] = (chunk_idx, file_idx)

    return mapping, sorted(set(conflicts))


def _build_tasks_lookup(dataset_root: Path) -> dict[int, str]:
    tasks_path = dataset_root / DEFAULT_TASKS_PATH
    if not tasks_path.exists():
        return {}
    tasks = load_tasks(dataset_root)
    lookup = {int(row.task_index): idx for idx, row in tasks.iterrows()}
    return lookup


def _select_old_episode_row(old_episodes_df, ep_idx: int, length: int) -> dict | None:
    rows = old_episodes_df[old_episodes_df["episode_index"] == ep_idx]
    if rows.empty:
        return None

    exact_len = rows[rows["length"] == length]
    if not exact_len.empty:
        return exact_len.iloc[0].to_dict()
    return rows.iloc[0].to_dict()


def _build_video_timestamps(episodes: list[dict], fps: int, video_key: str) -> None:
    per_file: dict[tuple[int, int], list[dict]] = {}
    for ep in episodes:
        chunk_idx = ep[f"videos/{video_key}/chunk_index"]
        file_idx = ep[f"videos/{video_key}/file_index"]
        per_file.setdefault((chunk_idx, file_idx), []).append(ep)

    for (_, _), file_eps in per_file.items():
        file_eps.sort(key=lambda x: x["dataset_from_index"])
        cursor = 0.0
        for ep in file_eps:
            length = ep["length"]
            ep[f"videos/{video_key}/from_timestamp"] = cursor
            cursor += length / fps
            ep[f"videos/{video_key}/to_timestamp"] = cursor


def _write_episodes(meta_root: Path, episodes: list[dict]) -> None:
    import datasets

    if not episodes:
        raise ValueError("No episodes to write")

    # Clean old episodes directory
    episodes_dir = meta_root / EPISODES_DIR
    if episodes_dir.exists():
        shutil.rmtree(episodes_dir)

    ds = datasets.Dataset.from_list(episodes)
    write_episodes(ds, meta_root)


def _update_info(meta_root: Path, total_episodes: int, total_frames: int) -> None:
    info_path = meta_root / INFO_PATH
    info = json.loads(info_path.read_text())
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["splits"] = {"train": f"0:{total_episodes}"}
    write_info(info, meta_root)


def repair_dataset(
    repo_id: str,
    output_repo_id: str,
    root: Path | None,
    download_videos: bool,
    push: bool,
    private: bool,
    keep_episode_indices: list[int] | None,
) -> Path:
    logging.info("Loading metadata")
    meta = LeRobotDatasetMetadata(repo_id, root=root)

    dataset = LeRobotDataset(repo_id, root=root)
    dataset.download(download_videos=download_videos)

    src_root = meta.root
    output_root = HF_LEROBOT_HOME / output_repo_id if root is None else root / output_repo_id

    if output_root.exists():
        raise FileExistsError(f"Output directory already exists: {output_root}")

    logging.info("Copying dataset")
    shutil.copytree(src_root, output_root)

    logging.info("Building episode stats")
    data_files = _list_parquet_files(output_root / DATA_DIR)
    ep_stats = _collect_episode_stats(data_files)
    ep_file_map, conflicts = _collect_episode_file_mapping(data_files)

    if keep_episode_indices is not None:
        keep_set = set(keep_episode_indices)
        ep_stats = {k: v for k, v in ep_stats.items() if k in keep_set}
        ep_file_map = {k: v for k, v in ep_file_map.items() if k in keep_set}

    if conflicts:
        logging.warning(f"Episodes spanning multiple data files: {conflicts}")

    old_meta = LeRobotDatasetMetadata(output_repo_id, root=output_root)
    old_episodes_df = old_meta.episodes.to_pandas()
    task_lookup = _build_tasks_lookup(output_root)

    episodes: list[dict] = []
    for ep_idx in sorted(ep_stats.keys(), key=lambda x: ep_stats[x]["min_index"]):
        stats = ep_stats[ep_idx]
        length = stats["count"]
        from_index = stats["min_index"]
        to_index = stats["max_index"] + 1

        row = _select_old_episode_row(old_episodes_df, ep_idx, length)
        tasks = None
        if row is not None:
            tasks = row.get("tasks")

        if tasks is None:
            tasks = []

        if not tasks:
            tasks = [task_lookup.get(0, "")] if task_lookup else []

        if ep_idx in ep_file_map:
            data_chunk, data_file = ep_file_map[ep_idx]
        elif row is not None:
            data_chunk = int(row.get("data/chunk_index"))
            data_file = int(row.get("data/file_index"))
        else:
            raise ValueError(f"Could not determine data file for episode {ep_idx}")

        episode_row = {
            "episode_index": ep_idx,
            "tasks": tasks,
            "length": length,
            "data/chunk_index": data_chunk,
            "data/file_index": data_file,
            "dataset_from_index": from_index,
            "dataset_to_index": to_index,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }

        for video_key in old_meta.video_keys:
            if row is not None:
                episode_row[f"videos/{video_key}/chunk_index"] = int(
                    row.get(f"videos/{video_key}/chunk_index", data_chunk)
                )
                episode_row[f"videos/{video_key}/file_index"] = int(
                    row.get(f"videos/{video_key}/file_index", data_file)
                )
                if f"videos/{video_key}/from_timestamp" in row:
                    episode_row[f"videos/{video_key}/from_timestamp"] = float(
                        row.get(f"videos/{video_key}/from_timestamp")
                    )
                if f"videos/{video_key}/to_timestamp" in row:
                    episode_row[f"videos/{video_key}/to_timestamp"] = float(
                        row.get(f"videos/{video_key}/to_timestamp")
                    )
            else:
                episode_row[f"videos/{video_key}/chunk_index"] = data_chunk
                episode_row[f"videos/{video_key}/file_index"] = data_file

        episodes.append(episode_row)

    for video_key in old_meta.video_keys:
        needs_ts = any(
            f"videos/{video_key}/from_timestamp" not in ep or f"videos/{video_key}/to_timestamp" not in ep
            for ep in episodes
        )
        if needs_ts:
            _build_video_timestamps(episodes, old_meta.fps, video_key)

    _write_episodes(output_root, episodes)
    _update_info(output_root, total_episodes=len(episodes), total_frames=sum(ep["length"] for ep in episodes))

    logging.info("Repair complete")

    if push:
        fixed_dataset = LeRobotDataset(output_repo_id, root=output_root)
        fixed_dataset.push_to_hub(private=private)

    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair LeRobot dataset episode metadata and push.")
    parser.add_argument("--repo-id", type=str, required=True, help="Original dataset repo id.")
    parser.add_argument(
        "--output-repo-id",
        type=str,
        required=True,
        help="Repo id to publish the repaired dataset.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root dataset directory (default: ~/.cache/huggingface/lerobot).",
    )
    parser.add_argument(
        "--download-videos",
        action="store_true",
        help="Download videos as well (recommended for video datasets).",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push repaired dataset to the hub.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Push as a private dataset.",
    )
    parser.add_argument(
        "--keep-episode-indices",
        type=str,
        default=None,
        help="Comma-separated episode indices to keep (e.g., 0,1,2).",
    )
    args = parser.parse_args()

    keep_episode_indices = None
    if args.keep_episode_indices:
        keep_episode_indices = [int(x.strip()) for x in args.keep_episode_indices.split(",") if x.strip()]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_root = repair_dataset(
        repo_id=args.repo_id,
        output_repo_id=args.output_repo_id,
        root=args.root,
        download_videos=args.download_videos,
        push=args.push,
        private=args.private,
        keep_episode_indices=keep_episode_indices,
    )

    print(f"Repaired dataset written to: {output_root}")


if __name__ == "__main__":
    main()
