#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import mujoco
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi


REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOCASA_REPO = Path.home() / ".cache" / "huggingface" / "lerobot" / "robocasa"
MIMICGEN_REPO = Path.home() / ".cache" / "huggingface" / "lerobot" / "mimicgen"

for repo in [ROBOCASA_REPO, MIMICGEN_REPO]:
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

import robocasa  # noqa: E402
import robosuite  # noqa: E402
from robocasa.scripts.download_datasets import download_and_extract_from_box  # noqa: E402
from robocasa.utils.dataset_registry_utils import get_ds_meta  # noqa: E402
from robocasa.utils.lerobot_utils import (  # noqa: E402
    calculate_dataset_statistics,
    get_episode_meta,
    get_episode_model_xml,
    get_episode_states,
    reorder_lerobot_action,
)
from robocasa.utils.robomimic.robomimic_dataset_utils import convert_to_robomimic_format  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402


VIDEO_KEYS = [
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
]

CANONICAL_FEATURES = {
    "observation.images.robot0_eye_in_hand": {
        "dtype": "video",
        "shape": [256, 256, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.fps": 20,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.robot0_agentview_left": {
        "dtype": "video",
        "shape": [256, 256, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.fps": 20,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.robot0_agentview_right": {
        "dtype": "video",
        "shape": [256, 256, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.fps": 20,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.state": {
        "dtype": "float32",
        "shape": [16],
        "names": [
            "state.base_position.x",
            "state.base_position.y",
            "state.base_position.z",
            "state.base_rotation.quat_x",
            "state.base_rotation.quat_y",
            "state.base_rotation.quat_z",
            "state.base_rotation.quat_w",
            "state.end_effector_position_relative.x",
            "state.end_effector_position_relative.y",
            "state.end_effector_position_relative.z",
            "state.end_effector_rotation_relative.quat_x",
            "state.end_effector_rotation_relative.quat_y",
            "state.end_effector_rotation_relative.quat_z",
            "state.end_effector_rotation_relative.quat_w",
            "state.gripper_qpos.finger1_left",
            "state.gripper_qpos.finger2_right",
        ],
    },
    "action": {
        "dtype": "float32",
        "shape": [12],
        "names": [
            "action.base_motion.mobile_forward",
            "action.base_motion.mobile_side",
            "action.base_motion.mobile_yaw",
            "action.base_motion.torso_height",
            "action.control_mode.base_mode",
            "action.end_effector_position.x",
            "action.end_effector_position.y",
            "action.end_effector_position.z",
            "action.end_effector_rotation.axis_angle_x",
            "action.end_effector_rotation.axis_angle_y",
            "action.end_effector_rotation.axis_angle_z",
            "action.gripper_close.command",
        ],
    },
    "annotation.human.task_name": {"dtype": "string", "shape": [1], "names": None},
    "annotation.human.task_description": {"dtype": "string", "shape": [1], "names": None},
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}

CANONICAL_HF_METADATA = {
    "info": {
        "features": {
            "observation.state": {
                "feature": {"dtype": "float32", "_type": "Value"},
                "length": 16,
                "_type": "List",
            },
            "action": {
                "feature": {"dtype": "float32", "_type": "Value"},
                "length": 12,
                "_type": "List",
            },
            "annotation.human.task_name": {"dtype": "string", "_type": "Value"},
            "annotation.human.task_description": {"dtype": "string", "_type": "Value"},
            "timestamp": {"dtype": "float32", "_type": "Value"},
            "frame_index": {"dtype": "int64", "_type": "Value"},
            "episode_index": {"dtype": "int64", "_type": "Value"},
            "index": {"dtype": "int64", "_type": "Value"},
            "task_index": {"dtype": "int64", "_type": "Value"},
        }
    }
}

PANDA_OMRON_CONTROLLER_CONFIG = {
    "type": "HYBRID_MOBILE_BASE",
    "composite_controller_specific_configs": {
        "left_offset": [0.0, 0.0, 0.0],
        "right_offset": [-0.15, 0.0, 0.35],
        "left2arm_transform": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "right2arm_transform": [[0, 0.7071068, 0.7071068, 0], [1, 0, 0, 0], [0, 0.7071068, -0.7071068, 0], [0, 0, 0, 1]],
        "left2finger_transform": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "right2finger_transform": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "body_part_ordering": ["right", "right_gripper", "base", "torso"],
    },
    "body_parts": {
        "right": {
            "type": "OSC_POSE",
            "input_max": 1,
            "input_min": -1,
            "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
            "kp": 150,
            "damping_ratio": 1,
            "impedance_mode": "fixed",
            "kp_limits": [0, 300],
            "damping_ratio_limits": [0, 10],
            "position_limits": None,
            "orientation_limits": None,
            "uncouple_pos_ori": True,
            "input_type": "delta",
            "input_ref_frame": "base",
            "interpolation": None,
            "ramp_ratio": 0.2,
            "gripper": {"type": "GRIP"},
        },
        "torso": {"type": "JOINT_POSITION", "interpolation": "null", "kp": 2000},
        "base": {"type": "JOINT_VELOCITY", "interpolation": "null"},
    },
}

TASK_TO_MIMICGEN_CONFIG = {
    "PickPlaceCounterToCabinet": "PnPCounterToCab",
    "PickPlaceCabinetToCounter": "PnPCabToCounter",
    "PickPlaceCounterToSink": "PnPCounterToSink",
    "PickPlaceSinkToCounter": "PnPSinkToCounter",
    "PickPlaceCounterToMicrowave": "PnPCounterToMicrowave",
    "PickPlaceMicrowaveToCounter": "PnPMicrowaveToCounter",
    "PickPlaceCounterToStove": "PnPCounterToStove",
    "PickPlaceStoveToCounter": "PnPStoveToCounter",
}


@dataclass(frozen=True)
class EpisodeRef:
    source_root: Path
    episode_index: int
    task_description: str
    split: str
    source: str
    layout_id: int | None
    style_id: int | None


def slugify_task(task_name: str) -> str:
    chars = [c.lower() if c.isalnum() else "_" for c in task_name]
    slug = "".join(chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def default_repo_id(task_name: str, total_eps: int, layout_id: int | None, style_id: int | None) -> str:
    base = f"robocasa_{slugify_task(task_name)}"
    if layout_id is not None:
        base += f"_layout{layout_id:03d}"
    if style_id is not None:
        base += f"_style{style_id:03d}"
    return f"{base}_{total_eps}eps"


def resolve_output_root(args: argparse.Namespace, total_eps: int) -> Path:
    if args.output_root is not None:
        return args.output_root.resolve()
    repo_id = args.repo_id or default_repo_id(args.task_name, total_eps, args.layout_id, args.style_id)
    return (Path.home() / ".cache" / "huggingface" / "lerobot" / "local" / repo_id).resolve()


def ensure_human_dataset(task_name: str, split: str) -> Path | None:
    meta = get_ds_meta(task=task_name, split=split, source="human")
    if meta is None:
        return None
    path = Path(meta["path"])
    if not path.exists():
        download_and_extract_from_box(destination=str(path))
    return path


def find_human_roots(task_name: str) -> list[tuple[str, Path]]:
    roots: list[tuple[str, Path]] = []
    for split in ["pretrain", "target", "real"]:
        try:
            root = ensure_human_dataset(task_name, split)
        except Exception:
            root = None
        if root is not None and root.exists():
            roots.append((split, root))
    return roots


def read_task_description(root: Path, ep_idx: int) -> str:
    meta = get_episode_meta(root, ep_idx)
    if "lang" in meta:
        return str(meta["lang"])
    cache = source_cache(root)
    task_map = cache["task_map"]
    ep_df = load_episode_dataframe(cache, ep_idx)
    task_idx = int(ep_df["task_index"].iloc[0])
    return str(task_map[task_idx])


def collect_matching_human_episodes(
    task_name: str,
    layout_id: int | None,
    style_id: int | None,
) -> list[EpisodeRef]:
    refs: list[EpisodeRef] = []
    for split, root in find_human_roots(task_name):
        episode_dirs = sorted((root / "extras").glob("episode_*"))
        for ep_dir in episode_dirs:
            ep_idx = int(ep_dir.name.split("_")[-1])
            ep_meta = json.loads((ep_dir / "ep_meta.json").read_text())
            ep_layout = ep_meta.get("layout_id")
            ep_style = ep_meta.get("style_id")
            if layout_id is not None and int(ep_layout) != layout_id:
                continue
            if style_id is not None and int(ep_style) != style_id:
                continue
            refs.append(
                EpisodeRef(
                    source_root=root,
                    episode_index=ep_idx,
                    task_description=read_task_description(root, ep_idx),
                    split=split,
                    source="human",
                    layout_id=int(ep_layout) if ep_layout is not None else None,
                    style_id=int(ep_style) if ep_style is not None else None,
                )
            )
    return refs


def canonical_schema() -> pa.Schema:
    metadata = {b"huggingface": json.dumps(CANONICAL_HF_METADATA).encode()}
    return pa.schema(
        [
            pa.field("observation.state", pa.list_(pa.float32(), 16)),
            pa.field("action", pa.list_(pa.float32(), 12)),
            pa.field("annotation.human.task_name", pa.string()),
            pa.field("annotation.human.task_description", pa.string()),
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
        ],
        metadata=metadata,
    )


def write_canonical_data_parquet(df: pd.DataFrame, out_path: Path) -> None:
    schema = canonical_schema()
    arrays = [
        pa.array(df["observation.state"].tolist(), type=pa.list_(pa.float32(), 16)),
        pa.array(df["action"].tolist(), type=pa.list_(pa.float32(), 12)),
        pa.array(df["annotation.human.task_name"].tolist(), type=pa.string()),
        pa.array(df["annotation.human.task_description"].tolist(), type=pa.string()),
        pa.array(df["timestamp"].astype(np.float32).tolist(), type=pa.float32()),
        pa.array(df["frame_index"].astype(np.int64).tolist(), type=pa.int64()),
        pa.array(df["episode_index"].astype(np.int64).tolist(), type=pa.int64()),
        pa.array(df["index"].astype(np.int64).tolist(), type=pa.int64()),
        pa.array(df["task_index"].astype(np.int64).tolist(), type=pa.int64()),
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema), out_path)


def copy_metadata_assets(reference_root: Path, output_root: Path) -> None:
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for name in ["modality.json", "embodiment.json"]:
        src = reference_root / "meta" / name
        if src.exists():
            shutil.copy2(src, meta_dir / name)
            continue
        asset_src = ROBOCASA_REPO / "robocasa" / "models" / "assets" / "groot_dataset_assets" / f"PandaOmron_{name}"
        if asset_src.exists():
            shutil.copy2(asset_src, meta_dir / name)


def make_output_info(total_episodes: int, total_frames: int) -> dict[str, Any]:
    return {
        "codebase_version": "v3.0",
        "robot_type": "PandaOmron",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 0,
        "chunks_size": 1000,
        "fps": 20,
        "splits": {},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": CANONICAL_FEATURES,
        "data_files_size_in_mb": 1000,
        "video_files_size_in_mb": 1000,
    }


def source_cache(root: Path) -> dict[str, Any]:
    tasks_parquet = root / "meta" / "tasks.parquet"
    if tasks_parquet.exists():
        tasks = pd.read_parquet(tasks_parquet)
        task_map = {int(i): str(task) for i, task in enumerate(tasks.index.tolist())}
        return {
            "format": "aggregated",
            "root": root,
            "info": json.loads((root / "meta" / "info.json").read_text()),
            "tasks": tasks,
            "task_map": task_map,
            "data": pd.read_parquet(root / "data" / "chunk-000" / "file-000.parquet"),
            "episodes": pd.read_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"),
        }

    tasks_jsonl = root / "meta" / "tasks.jsonl"
    if tasks_jsonl.exists():
        info = json.loads((root / "meta" / "info.json").read_text())
        tasks_df = pd.read_json(tasks_jsonl, lines=True)
        task_map = {int(row["task_index"]): str(row["task"]) for _, row in tasks_df.iterrows()}
        episodes_jsonl = pd.read_json(root / "meta" / "episodes.jsonl", lines=True)
        fps = float(info.get("fps", 20))
        chunk_size = int(info.get("chunks_size", 1000))
        running_index = 0
        episode_rows: list[dict[str, Any]] = []
        for _, row in episodes_jsonl.sort_values("episode_index").iterrows():
            ep_idx = int(row["episode_index"])
            ep_len = int(row["length"])
            episode_row: dict[str, Any] = {
                "episode_index": ep_idx,
                "data/chunk_index": ep_idx // chunk_size,
                "data/file_index": ep_idx,
                "dataset_from_index": running_index,
                "dataset_to_index": running_index + ep_len,
                "tasks": np.asarray(row["tasks"], dtype=object),
                "length": ep_len,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
            for key in VIDEO_KEYS:
                episode_row[f"videos/{key}/chunk_index"] = ep_idx // chunk_size
                episode_row[f"videos/{key}/file_index"] = ep_idx
                episode_row[f"videos/{key}/from_timestamp"] = 0.0
                episode_row[f"videos/{key}/to_timestamp"] = ep_len / fps
            episode_rows.append(episode_row)
            running_index += ep_len

        return {
            "format": "per_episode",
            "root": root,
            "info": info,
            "tasks": tasks_df,
            "task_map": task_map,
            "data": None,
            "episodes": pd.DataFrame(episode_rows),
        }

    raise FileNotFoundError(f"Unsupported LeRobot dataset layout under {root}")


def load_episode_dataframe(cache: dict[str, Any], episode_index: int) -> pd.DataFrame:
    if cache["format"] == "aggregated":
        data_df = cache["data"]
        return data_df.loc[data_df["episode_index"] == episode_index].copy()

    root = cache["root"]
    info = cache["info"]
    chunk_size = int(info.get("chunks_size", 1000))
    data_rel = info["data_path"].format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
    )
    return pd.read_parquet(root / data_rel)


def resolve_source_video_path(cache: dict[str, Any], video_key: str, ep_row: pd.Series) -> Path:
    root = cache["root"]
    if cache["format"] == "aggregated":
        src_chunk = int(ep_row[f"videos/{video_key}/chunk_index"])
        src_file = int(ep_row[f"videos/{video_key}/file_index"])
        return root / "videos" / video_key / f"chunk-{src_chunk:03d}" / f"file-{src_file:03d}.mp4"

    info = cache["info"]
    episode_index = int(ep_row["episode_index"])
    chunk_size = int(info.get("chunks_size", 1000))
    video_rel = info["video_path"].format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
        video_key=video_key,
    )
    return root / video_rel


def remap_video_files(
    episode_refs: list[EpisodeRef],
    output_root: Path,
) -> dict[tuple[Path, str, int, int], tuple[int, int]]:
    mapping: dict[tuple[Path, str, int, int], tuple[int, int]] = {}
    next_file_idx = {key: 0 for key in VIDEO_KEYS}
    caches: dict[Path, dict[str, Any]] = {}

    for ref in episode_refs:
        if ref.source_root not in caches:
            caches[ref.source_root] = source_cache(ref.source_root)
        cache = caches[ref.source_root]
        ep_row = cache["episodes"].loc[cache["episodes"]["episode_index"] == ref.episode_index].iloc[0]
        for key in VIDEO_KEYS:
            src_chunk = int(ep_row[f"videos/{key}/chunk_index"])
            src_file = int(ep_row[f"videos/{key}/file_index"])
            map_key = (ref.source_root, key, src_chunk, src_file)
            if map_key in mapping:
                continue
            dst_chunk = 0
            dst_file = next_file_idx[key]
            next_file_idx[key] += 1
            src_path = resolve_source_video_path(cache, key, ep_row)
            dst_path = output_root / "videos" / key / f"chunk-{dst_chunk:03d}" / f"file-{dst_file:03d}.mp4"
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            mapping[map_key] = (dst_chunk, dst_file)
    return mapping


def shift_numeric_list(value: Any, delta: float) -> Any:
    if isinstance(value, np.ndarray):
        return value + delta
    if isinstance(value, list):
        return [v + delta for v in value]
    return value + delta


def validate_dataset_root(root: Path, expected_episodes: int) -> None:
    required = [
        root / "data" / "chunk-000" / "file-000.parquet",
        root / "meta" / "tasks.parquet",
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        root / "meta" / "info.json",
        root / "meta" / "stats.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"Dataset build incomplete, missing required files: {missing}")

    episodes_df = pd.read_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    if len(episodes_df) != expected_episodes:
        raise RuntimeError(f"Expected {expected_episodes} episodes, found {len(episodes_df)} in meta/episodes parquet")

    for key in VIDEO_KEYS:
        for _, row in episodes_df.iterrows():
            video_path = (
                root
                / "videos"
                / key
                / f"chunk-{int(row[f'videos/{key}/chunk_index']):03d}"
                / f"file-{int(row[f'videos/{key}/file_index']):03d}.mp4"
            )
            if not video_path.exists():
                raise RuntimeError(f"Missing referenced video file: {video_path}")


def infer_dataset_meta(task_name: str, layout_id: int | None, style_id: int | None) -> dict[str, Any]:
    env_kwargs = {
        "robots": "PandaOmron",
        "layout_ids": None,
        "style_ids": None,
        "obj_instance_split": "target",
        "clutter_mode": 1,
        "translucent_robot": False,
        "layout_and_style_ids": (
            [[layout_id, style_id]] if layout_id is not None and style_id is not None else None
        ),
        "controller_configs": PANDA_OMRON_CONTROLLER_CONFIG,
        "reward_shaping": False,
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "use_camera_obs": False,
        "camera_depths": False,
        "camera_heights": 84,
        "camera_widths": 84,
        "camera_names": [],
    }
    return {
        "env_args": {
            "env_name": task_name,
            "env_version": "1.5.2",
            "type": 1,
            "env_kwargs": env_kwargs,
        }
    }


def build_lerobot_dataset_from_refs(
    episode_refs: list[EpisodeRef],
    output_root: Path,
    task_name: str,
    layout_id: int | None,
    style_id: int | None,
) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not episode_refs:
        raise ValueError("No matching human demonstrations were found.")

    reference_root = episode_refs[0].source_root
    copy_metadata_assets(reference_root, output_root)

    caches: dict[Path, dict[str, Any]] = {}
    video_mapping = remap_video_files(episode_refs, output_root)
    task_to_idx: dict[str, int] = {}

    all_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    global_index = 0

    for new_ep_idx, ref in enumerate(episode_refs):
        if ref.source_root not in caches:
            caches[ref.source_root] = source_cache(ref.source_root)
        cache = caches[ref.source_root]
        eps_df = cache["episodes"]

        ep_data = load_episode_dataframe(cache, ref.episode_index)
        ep_data = ep_data.sort_values("frame_index").reset_index(drop=True)
        ep_len = len(ep_data)
        if ep_len == 0:
            raise ValueError(f"No rows found for episode {ref.episode_index} in {ref.source_root}")

        if ref.task_description not in task_to_idx:
            task_to_idx[ref.task_description] = len(task_to_idx)
        new_task_idx = task_to_idx[ref.task_description]

        episode_start = global_index
        for frame_idx, (_, row) in enumerate(ep_data.iterrows()):
            all_rows.append(
                {
                    "observation.state": np.asarray(row["observation.state"], dtype=np.float32).tolist(),
                    "action": np.asarray(row["action"], dtype=np.float32).tolist(),
                    "annotation.human.task_name": task_name,
                    "annotation.human.task_description": ref.task_description,
                    "timestamp": np.float32(row["timestamp"]),
                    "frame_index": np.int64(frame_idx),
                    "episode_index": np.int64(new_ep_idx),
                    "index": np.int64(global_index),
                    "task_index": np.int64(new_task_idx),
                }
            )
            global_index += 1

        src_ep_row = eps_df.loc[eps_df["episode_index"] == ref.episode_index].iloc[0]
        episode_row: dict[str, Any] = {
            "episode_index": new_ep_idx,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": episode_start,
            "dataset_to_index": episode_start + ep_len,
            "tasks": [ref.task_description],
            "length": ep_len,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for key in VIDEO_KEYS:
            src_chunk = int(src_ep_row[f"videos/{key}/chunk_index"])
            src_file = int(src_ep_row[f"videos/{key}/file_index"])
            dst_chunk, dst_file = video_mapping[(ref.source_root, key, src_chunk, src_file)]
            episode_row[f"videos/{key}/chunk_index"] = dst_chunk
            episode_row[f"videos/{key}/file_index"] = dst_file
            episode_row[f"videos/{key}/from_timestamp"] = float(src_ep_row.get(f"videos/{key}/from_timestamp", 0.0))
            episode_row[f"videos/{key}/to_timestamp"] = float(
                src_ep_row.get(f"videos/{key}/to_timestamp", ep_len / 20.0)
            )
        episode_rows.append(episode_row)

        src_extra_dir = ref.source_root / "extras" / f"episode_{ref.episode_index:06d}"
        dst_extra_dir = output_root / "extras" / f"episode_{new_ep_idx:06d}"
        shutil.copytree(src_extra_dir, dst_extra_dir)

    output_df = pd.DataFrame(all_rows)
    write_canonical_data_parquet(output_df, output_root / "data" / "chunk-000" / "file-000.parquet")

    tasks_df = pd.DataFrame({"task_index": list(task_to_idx.values())}, index=list(task_to_idx.keys()))
    tasks_df.to_parquet(output_root / "meta" / "tasks.parquet")

    episodes_df = pd.DataFrame(episode_rows)
    (output_root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    episodes_df.to_parquet(output_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)

    info = make_output_info(total_episodes=len(episode_refs), total_frames=len(output_df))
    info["total_tasks"] = len(tasks_df)
    with open(output_root / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    stats = calculate_dataset_statistics([output_root / "data" / "chunk-000" / "file-000.parquet"])
    with open(output_root / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    dataset_meta = infer_dataset_meta(task_name=task_name, layout_id=layout_id, style_id=style_id)
    with open(output_root / "extras" / "dataset_meta.json", "w") as f:
        json.dump(dataset_meta, f, indent=2)


def build_fixed_env_info(task_name: str, layout_id: int, style_id: int) -> dict[str, Any]:
    return {
        "env_name": task_name,
        "robots": "PandaOmron",
        "layout_ids": None,
        "style_ids": None,
        "obj_instance_split": "target",
        "clutter_mode": 1,
        "translucent_robot": False,
        "layout_and_style_ids": [[layout_id, style_id]],
        "controller_configs": PANDA_OMRON_CONTROLLER_CONFIG,
        "reward_shaping": False,
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "use_camera_obs": False,
        "camera_depths": False,
        "camera_heights": 84,
        "camera_widths": 84,
        "camera_names": [],
    }


def rebuild_source_hdf5(
    source_dataset_dir: Path,
    output_hdf5: Path,
    task_name: str,
    layout_id: int,
    style_id: int,
) -> None:
    output_hdf5.parent.mkdir(parents=True, exist_ok=True)
    if output_hdf5.exists():
        output_hdf5.unlink()

    dataset_meta = LeRobotDatasetMetadata(repo_id="local/rebuild", root=source_dataset_dir)
    episode_dirs = sorted((source_dataset_dir / "extras").glob("episode_*"))
    modality_dict = json.loads((source_dataset_dir / "meta" / "modality.json").read_text())

    total_samples = 0
    with h5py.File(output_hdf5, "w") as f:
        data_grp = f.create_group("data")
        data_grp.attrs["env"] = task_name
        data_grp.attrs["env_info"] = json.dumps(build_fixed_env_info(task_name, layout_id, style_id))
        data_grp.attrs["env_args"] = json.dumps(infer_dataset_meta(task_name, layout_id, style_id)["env_args"])
        data_grp.attrs["robocasa_version"] = robocasa.__version__
        data_grp.attrs["robosuite_version"] = robosuite.__version__
        data_grp.attrs["mujoco_version"] = mujoco.__version__

        for ep_idx, _ in enumerate(episode_dirs):
            states = get_episode_states(source_dataset_dir, ep_idx)
            data_file = source_dataset_dir / dataset_meta.get_data_file_path(ep_idx)
            df = pd.read_parquet(data_file)
            ep_df = df[df["episode_index"] == ep_idx]
            lerobot_actions = np.stack(ep_df["action"].to_list())
            actions = reorder_lerobot_action(lerobot_actions, source_dataset_dir)
            ep_meta = get_episode_meta(source_dataset_dir, ep_idx)
            ep_meta["layout_id"] = layout_id
            ep_meta["style_id"] = style_id
            model_xml = get_episode_model_xml(source_dataset_dir, ep_idx)

            demo_grp = data_grp.create_group(f"demo_{ep_idx}")
            demo_grp.attrs["model_file"] = model_xml
            demo_grp.attrs["ep_meta"] = json.dumps(ep_meta)
            demo_grp.create_dataset("states", data=np.asarray(states))
            demo_grp.create_dataset("actions", data=np.asarray(actions))
            total_samples += int(actions.shape[0])

        data_grp.attrs["total"] = total_samples

    convert_to_robomimic_format(str(output_hdf5), verbose=True)


def normalize_generated_lerobot(
    source_root: Path,
    reference_root: Path,
    output_root: Path,
    task_name: str,
) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    shutil.copytree(source_root, output_root)

    task_descs = pd.read_parquet(output_root / "meta" / "tasks.parquet").index.tolist()
    data_df = pd.read_parquet(output_root / "data" / "chunk-000" / "file-000.parquet")
    data_df["annotation.human.task_name"] = task_name
    data_df["annotation.human.task_description"] = [task_descs[int(idx)] for idx in data_df["task_index"].tolist()]
    data_df = data_df.drop(columns=["next.reward", "next.done"], errors="ignore")
    data_df = data_df[
        [
            "observation.state",
            "action",
            "annotation.human.task_name",
            "annotation.human.task_description",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ]
    ]
    data_df["observation.state"] = data_df["observation.state"].apply(lambda x: np.asarray(x, dtype=np.float32).tolist())
    data_df["action"] = data_df["action"].apply(lambda x: np.asarray(x, dtype=np.float32).tolist())
    write_canonical_data_parquet(data_df, output_root / "data" / "chunk-000" / "file-000.parquet")

    info = make_output_info(
        total_episodes=int(pd.read_parquet(output_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").shape[0]),
        total_frames=len(data_df),
    )
    info["total_tasks"] = len(task_descs)
    with open(output_root / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    stats = calculate_dataset_statistics([output_root / "data" / "chunk-000" / "file-000.parquet"])
    with open(output_root / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    copy_metadata_assets(reference_root, output_root)


def generate_missing_with_mimicgen(
    human_seed_root: Path,
    reference_root: Path,
    task_name: str,
    layout_id: int,
    style_id: int,
    missing_episodes: int,
    work_root: Path,
    seed: int,
) -> Path:
    import mimicgen.configs  # noqa: F401
    from mimicgen.configs import config_factory
    from mimicgen.scripts.prepare_src_dataset import prepare_src_dataset
    from mimicgen.scripts.generate_dataset import generate_dataset

    if task_name not in TASK_TO_MIMICGEN_CONFIG:
        supported = ", ".join(sorted(TASK_TO_MIMICGEN_CONFIG))
        raise ValueError(f"MimicGen backfill is not configured for {task_name}. Supported tasks: {supported}")

    source_dir = work_root / "source"
    generated_dir = work_root / "generated"
    source_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    source_hdf5 = source_dir / "demo.hdf5"
    rebuild_source_hdf5(
        source_dataset_dir=human_seed_root,
        output_hdf5=source_hdf5,
        task_name=task_name,
        layout_id=layout_id,
        style_id=style_id,
    )
    prepare_src_dataset(
        dataset_path=str(source_hdf5),
        env_interface_name=f"MG_{TASK_TO_MIMICGEN_CONFIG[task_name]}",
        env_interface_type="robosuite",
        filter_key=None,
        n=None,
        output_path=None,
    )

    mg_name = default_repo_id(task_name, missing_episodes, layout_id, style_id).replace("robocasa_", "")
    mg_config = config_factory(TASK_TO_MIMICGEN_CONFIG[task_name], "robosuite")
    mg_config.experiment.name = f"{mg_name}_mg_seed{seed}"
    mg_config.experiment.source.dataset_path = str(source_hdf5)
    mg_config.experiment.generation.path = str(generated_dir)
    mg_config.experiment.generation.guarantee = True
    mg_config.experiment.generation.keep_failed = True
    mg_config.experiment.generation.num_trials = missing_episodes
    mg_config.experiment.generation.select_src_per_subtask = False
    mg_config.experiment.generation.transform_first_robot_pose = True
    mg_config.experiment.generation.interpolate_from_last_target_pose = True
    mg_config.experiment.task.name = task_name
    mg_config.experiment.render_video = False
    mg_config.experiment.num_demo_to_render = 0
    mg_config.experiment.num_fail_demo_to_render = 0
    mg_config.experiment.seed = seed
    mg_config.obs.collect_obs = True
    mg_config.obs.camera_names = []
    mg_config.obs.camera_height = 84
    mg_config.obs.camera_width = 84

    generate_dataset(mg_config=mg_config, auto_remove_exp=True, render=False, video_path=None)

    mg_run_dir = generated_dir / mg_config.experiment.name
    raw_hdf5 = mg_run_dir / "demo.hdf5"
    convert_script = ROBOCASA_REPO / "robocasa" / "scripts" / "dataset_scripts" / "convert_hdf5_lerobot.py"
    cmd = (
        f"{sys.executable} {convert_script}"
        f" --raw_dataset_path {raw_hdf5}"
        f" --camera_names robot0_eye_in_hand robot0_agentview_left robot0_agentview_right"
        f" --camera_height 256 --camera_width 256"
    )
    status = os.system(cmd)
    if status != 0 and not (mg_run_dir / "lerobot").exists():
        raise RuntimeError(f"RoboCasa conversion failed with status {status} for {raw_hdf5}")

    normalized_root = mg_run_dir / "lerobot_norm"
    normalize_generated_lerobot(
        source_root=mg_run_dir / "lerobot",
        reference_root=reference_root,
        output_root=normalized_root,
        task_name=task_name,
    )
    return normalized_root


def subset_episode_refs(episode_refs: list[EpisodeRef], limit: int) -> list[EpisodeRef]:
    return episode_refs[:limit]


def generated_episode_refs(root: Path, task_name: str) -> list[EpisodeRef]:
    refs: list[EpisodeRef] = []
    episode_dirs = sorted((root / "extras").glob("episode_*"))
    for ep_dir in episode_dirs:
        ep_idx = int(ep_dir.name.split("_")[-1])
        ep_meta = json.loads((ep_dir / "ep_meta.json").read_text())
        refs.append(
            EpisodeRef(
                source_root=root,
                episode_index=ep_idx,
                task_description=read_task_description(root, ep_idx),
                split="generated",
                source="mimicgen",
                layout_id=int(ep_meta.get("layout_id")) if ep_meta.get("layout_id") is not None else None,
                style_id=int(ep_meta.get("style_id")) if ep_meta.get("style_id") is not None else None,
            )
        )
    return refs


def push_dataset(output_root: Path, repo_id: str) -> None:
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(output_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fixed-task RoboCasa LeRobot dataset, optionally backfilling with MimicGen."
    )
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--layout-id", type=int)
    parser.add_argument("--style-id", type=int)
    parser.add_argument("--desired-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--repo-id", type=str)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-mimicgen", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repo_id and not args.push_to_hub:
        args.push_to_hub = True

    human_refs = collect_matching_human_episodes(
        task_name=args.task_name,
        layout_id=args.layout_id,
        style_id=args.style_id,
    )
    if not human_refs:
        raise SystemExit("No matching human demonstrations found after downloading available human splits.")

    final_refs = subset_episode_refs(human_refs, min(len(human_refs), args.desired_episodes))
    output_root = resolve_output_root(args, total_eps=args.desired_episodes)
    work_root = REPO_ROOT / "outputs" / "robocasa_dataset_builder" / default_repo_id(
        args.task_name,
        args.desired_episodes,
        args.layout_id,
        args.style_id,
    )
    work_root.mkdir(parents=True, exist_ok=True)

    human_root = work_root / "human_subset"
    build_lerobot_dataset_from_refs(
        episode_refs=final_refs,
        output_root=human_root,
        task_name=args.task_name,
        layout_id=args.layout_id,
        style_id=args.style_id,
    )

    if len(final_refs) < args.desired_episodes:
        missing = args.desired_episodes - len(final_refs)
        if args.skip_mimicgen:
            raise SystemExit(
                f"Only found {len(final_refs)} human demonstrations, but {args.desired_episodes} were requested."
            )
        if args.layout_id is None or args.style_id is None:
            raise SystemExit("MimicGen backfill requires both --layout-id and --style-id.")

        generated_root = generate_missing_with_mimicgen(
            human_seed_root=human_root,
            reference_root=human_root,
            task_name=args.task_name,
            layout_id=args.layout_id,
            style_id=args.style_id,
            missing_episodes=missing,
            work_root=work_root,
            seed=args.seed,
        )
        final_refs.extend(subset_episode_refs(generated_episode_refs(generated_root, args.task_name), missing))

    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"{output_root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)

    build_lerobot_dataset_from_refs(
        episode_refs=final_refs,
        output_root=output_root,
        task_name=args.task_name,
        layout_id=args.layout_id,
        style_id=args.style_id,
    )
    validate_dataset_root(output_root, expected_episodes=len(final_refs))

    if args.push_to_hub:
        repo_id = args.repo_id or f"steb6/{default_repo_id(args.task_name, len(final_refs), args.layout_id, args.style_id)}"
        push_dataset(output_root=output_root, repo_id=repo_id)
        print(f"Pushed dataset to https://huggingface.co/datasets/{repo_id}")

    print(f"Built dataset with {len(final_refs)} episodes at {output_root}")


if __name__ == "__main__":
    main()
