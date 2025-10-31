#!/usr/bin/env python3
"""
Split a long teleoperation dataset into per-action episodes by streaming frames to
an external action recognition module over YARP and collecting its labels.

Behavior (assumptions):
- Input dataset is a HuggingFace dataset (local path or hub repo id) and contains
  a sequence of frames in a column whose name is one of: 'image','rgb','frame','frames','images','rgb_image'.
  If an Image feature exists the script will use it.
- The script streams each frame (as a JPEG, base64 encoded) over a YARP output
  port (default: '/splitter/video:o') inside a Bottle with fields:
      [ int(frame_index) , string(jpeg_base64) ]
    Each episode will contain only frames assigned to the label and will have the
    episode task equal to the label string.
- The external action recognition module must listen to the video port and reply
  labels by publishing to a label port (remote) which will be connected to the
  local label input port (default: '/splitter/labels:i'). The external module
  must send a Bottle for each frame containing at least: [ int(frame_index), string(label) ]
- The script collects one label per frame (waiting up to --label-timeout seconds)
  and assigns that label to the frame. After processing all frames it groups
  frames by label and creates one episode per label (i.e., the resulting dataset
  will contain N episodes, one for each detected label). The per-label datasets
  are saved locally (and optionally pushed to the Hub).

Notes:
- This script is intentionally conservative about YARP message formats. If your
  action recognition module expects a different message format, adjust the
  `send_frame` and `read_label` helper functions accordingly.

Usage:
    python -m lerobot.scripts.split_dataset_by_action --dataset repo_id_or_path \
        --label-port-remote /action_detector/labels:o --output-prefix my-splits \
        --push-to-hub --hf-token <TOKEN>

"""

from __future__ import annotations
import argparse
import base64
import io
import os
import sys
import time
from collections import defaultdict
from typing import Iterable

import numpy as np
from datasets import load_dataset, Dataset
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import yarp
from PIL import Image
from pathlib import Path
import torch
import pandas as pd
from huggingface_hub import HfApi
import tempfile
import shutil

from lerobot.datasets.video_utils import decode_video_frames, encode_video_frames

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import DEFAULT_DATA_PATH, write_info
from lerobot.datasets.dataset_tools import _write_parquet



class YarpStreamer:
    def __init__(self, video_out_port_local: str, video_out_port_remote: str, label_in_port_local: str, label_in_port_remote: str, label_timeout: float = 0.5):
        yarp.Network.init()
        self.video_out = yarp.BufferedPortImageRgb()

        if not self.video_out.open(video_out_port_local):
            raise ConnectionError(f"Failed to open video out port {video_out_port_local}")
        connected = False
        for _ in range(10):
            if yarp.Network.connect(video_out_port_local, video_out_port_remote):
                connected = True
                break
            time.sleep(0.1)
        if not connected:
            raise ConnectionError(f"Failed to connect {video_out_port_local} -> {video_out_port_remote}")
        # Connect label input
        self.label_in = yarp.BufferedPortBottle()
        if not self.label_in.open(label_in_port_local):
            raise ConnectionError(f"Failed to open local label input port {label_in_port_local}")
        connected = False
        for _ in range(10):
            if yarp.Network.connect(label_in_port_remote, label_in_port_local):
                connected = True
                break
            time.sleep(0.1)
        if not connected:
            raise ConnectionError(f"Failed to connect {label_in_port_remote} -> {label_in_port_local}")
        self.label_timeout = label_timeout

    def send_frame(self, index: int, image: Image.Image) -> None:
        """Publish a frame on the image port so image readers can use
        ImageRgb.setExternal on their own numpy buffer.

        We convert the PIL image to a contiguous HxWx3 uint8 numpy array (RGB),
        set it as the external buffer of the prepared ImageRgb and write the port.
        This matches the pattern used by many consumers which call:
            img = yarp.ImageRgb(); img.resize(w,h); img.setExternal(buf.data, w, h)
        """
        # Ensure RGB uint8 numpy array, contiguous
        arr = np.array(image.convert("RGB"), dtype=np.uint8)
        arr = np.ascontiguousarray(arr)
        h, w = arr.shape[0], arr.shape[1]

        # Prepare the image port's internal image and set external buffer
        out = self.video_out.prepare()
        try:
            # For BufferedPortImageRgb the prepare() returns a yarp.ImageRgb
            out.resize(w, h)
            # setExternal expects a pointer to the buffer; numpy's .data is fine in these bindings
            out.setExternal(arr.data, w, h)
        except Exception:
            # If the port is actually a Bottle fallback, send a base64 JPEG as before
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=90)
            jpeg = buf.getvalue()
            b64 = base64.b64encode(jpeg).decode("ascii")
            out.clear()
            out.addInt32(int(index))
            out.addString(b64)

        # write the prepared image/bottle
        self.video_out.write()

    def read_label(self) -> tuple[int, str] | None:
        """Read a label Bottle from the label input port. Expected format: [index, label]
        Returns (index,label) or None on timeout.
        """
        b = self.label_in.read(False)
        if b is None:
            return None
        return b.get(0).toString()

    def close(self):
        try:
            self.video_out.close()
        except Exception:
            pass
        try:
            self.label_in.close()
        except Exception:
            pass


def pil_from_dataset_item(item, img_field: str) -> Image.Image:
    val = item[img_field]
    if torch is not None and isinstance(val, torch.Tensor):
        val = val.cpu().numpy()
    if np.issubdtype(val.dtype, np.floating):
        if np.nanmax(val) <= 1.0:
            val = (val * 255.0)
    val = np.clip(val, 0, 255).astype("uint8")
    if val.shape[0] == 3:
        val = val.swapaxes(0, 1).swapaxes(1, 2)
    im = Image.fromarray(val)
    return im.convert("RGB")


def split_dataset_by_label(ds: Dataset, labels: list[str], output_prefix: str, push_to_hub: bool = False) -> dict[str, str]:
    """Given dataset and per-frame labels (aligned by index), create one dataset per unique label
    and save locally under output_prefix + label. Optionally push to hub.
    Returns mapping label -> saved_path or hub repo id.
    """
    # labels is a list aligned to examples in ds, one label per example
    indices_by_label: dict[str, list[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        if lbl is None:
            continue
        indices_by_label[lbl].append(i)

    results = {}
    for lbl, idxs in indices_by_label.items():
        sub = ds.select(idxs)
        out_dir = f"{output_prefix}-{lbl}"
        sub.save_to_disk(out_dir)
        results[lbl] = out_dir
    return results


def split_one_episode_by_labels(src_ds: LeRobotDataset, labels: list[str], output_prefix: str, push_to_hub: bool = False, hf_token: str | None = None) -> dict[str, str]:
    """Split a single-episode LeRobotDataset into one dataset that contains K episodes (one per label).

    The resulting dataset is written under `output_prefix` (a single folder). Each label becomes one
    episode (episode_index = 0..K-1). Episode videos (if present) are encoded per-episode as H.264
    files and referenced from the episode metadata.

    Returns a dict with a single entry {'dataset': out_dir} for compatibility with the original
    return type.
    """
    if src_ds.meta.total_episodes != 1:
        raise ValueError("Source dataset must contain exactly one episode")

    hf = src_ds.hf_dataset
    if hf is None:
        raise RuntimeError("Source LeRobotDataset has no hf_dataset loaded")
    # Since AR module has window of length 16, we do the following
    # tapullo per i None
    # test = []
    # for i in range(len(labels)-1):
    #     if labels[i] is None and labels[i+1] is not None:
    #         test.append(labels[i+1])
    #     else:
    #         test.append('')
    # test.append(labels[-1])
    # for i in range(len(test)-1):
    #     if test[i] == '' and test[i+1] != '':
    #         test[i] = test[i+1]
    # # move predictions
    # test = test[16:] + [None] * 16
    # # Clean results with consistency
    # consistency_window = 8
    # test_ = test.copy()
    # for i in range(consistency_window, len(test_)-consistency_window):
    #     window = test_[i-consistency_window:i+consistency_window]
    #     most_frequent = max(set(window), key=window.count)
    #     test_[i] = most_frequent
    # labels = test_

    if len(labels) != len(hf):
        raise ValueError("Labels length must match number of frames in dataset")

    # Build indices by label preserving order of first appearance
    indices_by_label: dict[str, list[int]] = defaultdict(list)
    labels_order: list[str] = []
    for i, lbl in enumerate(labels):
        if lbl is None:
            continue
        if lbl not in indices_by_label:
            labels_order.append(lbl)
        indices_by_label[lbl].append(i)

    if len(labels_order) == 0:
        raise ValueError("No labels found to split into episodes")

    src_meta = src_ds.meta

    # Prepare single output directory
    out_dir = Path(str(output_prefix))
    repo_id = out_dir.name

    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=src_meta.fps,
        features=src_meta.features,
        robot_type=src_meta.robot_type,
        root=out_dir,
        use_videos=len(src_meta.video_keys) > 0,
    )

    # Build a combined dataframe where each label becomes a distinct episode
    frames_list = []
    global_index = 0
    episode_lengths = []
    for ep_idx, lbl in enumerate(labels_order):
        idxs = indices_by_label[lbl]
        sub = hf.select(idxs)
        try:
            df = sub.to_pandas()
        except Exception:
            df = pd.DataFrame(list(sub))
        df = df.reset_index(drop=True)
        df["episode_index"] = ep_idx
        df["frame_index"] = list(range(len(df)))
        df["index"] = list(range(global_index, global_index + len(df)))
        df["timestamp"] = df["frame_index"].astype(float) / float(src_meta.fps)
        # task_index points to the task (we'll store tasks in the same order as labels_order)
        df["task_index"] = ep_idx
        frames_list.append(df)
        global_index += len(df)
        episode_lengths.append(len(df))

    full_df = pd.concat(frames_list, ignore_index=True)

    # Write single parquet file
    data_path = out_dir / DEFAULT_DATA_PATH.format(chunk_index=0, file_index=0)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet(full_df, data_path, new_meta)

    # Save tasks (labels as tasks)
    new_meta.save_episode_tasks(labels_order)

    # Encode per-episode videos and write episode metadata
    total_frames = 0
    for ep_idx, lbl in enumerate(labels_order):
        ep_len = episode_lengths[ep_idx]

        # default episode metadata (data indices point to the single parquet file)
        ep_meta = {
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": int(full_df[full_df["episode_index"] == ep_idx]["index"].min()),
            "dataset_to_index": int(full_df[full_df["episode_index"] == ep_idx]["index"].max() + 1),
        }

        video_fields = {}
        if src_meta.video_keys:
            src_ep = src_ds.meta.episodes[0]
            ep_start = int(src_ep.get("dataset_from_index", 0))
            for video_key in src_meta.video_keys:
                from_ts = float(src_ep.get(f"videos/{video_key}/from_timestamp", 0.0))
                vinfo = src_ds.meta.info.get("features", {}).get(video_key, {}).get("info", {}) or {}
                video_fps = float(vinfo.get("video.fps", src_meta.fps))

                src_video_path = src_ds.root / src_ds.meta.get_video_file_path(0, video_key)
                if not src_video_path.exists():
                    continue

                # timestamps for this episode relative to source video
                idxs = indices_by_label[lbl]
                timestamps = [from_ts + ((int(idx) - ep_start) / float(src_meta.fps)) for idx in idxs]
                print(f"Encoding video for episode {ep_idx} label '{lbl}' with {len(timestamps)} frames from {src_video_path}")

                # decode frames and materialize to PNGs
                try:
                    frames_t = decode_video_frames(src_video_path, timestamps, src_ds.tolerance_s, backend=src_ds.video_backend)
                except Exception:
                    frames_t = []

                imgs_dir = None
                if len(frames_t) > 0:
                    imgs_dir = Path(tempfile.mkdtemp())
                    for i, frame in enumerate(frames_t):
                        arr = (frame.mul(255).clamp(0, 255).permute(1, 2, 0).cpu().numpy()).astype("uint8")
                        img = Image.fromarray(arr)
                        fname = imgs_dir / f"frame-{i:06d}.png"
                        img.save(fname, format="PNG")

                    # destination video path per episode (one file per episode)
                    dst_rel = new_meta.video_path.format(video_key=video_key, chunk_index=0, file_index=ep_idx)
                    dst_video_path = new_meta.root / dst_rel
                    dst_video_path.parent.mkdir(parents=True, exist_ok=True)

                    encode_video_frames(imgs_dir, dst_video_path, fps=int(video_fps), vcodec="h264", pix_fmt="yuv420p", overwrite=True)

                    duration = len(frames_t) / float(video_fps) if video_fps > 0 else 0.0
                    video_fields[f"videos/{video_key}/chunk_index"] = 0
                    video_fields[f"videos/{video_key}/file_index"] = ep_idx
                    video_fields[f"videos/{video_key}/from_timestamp"] = 0.0
                    video_fields[f"videos/{video_key}/to_timestamp"] = duration

                if imgs_dir is not None:
                    try:
                        shutil.rmtree(str(imgs_dir))
                    except Exception:
                        pass

        episode_dict = {"episode_index": ep_idx, "tasks": [labels_order[ep_idx]], "length": ep_len}
        episode_dict.update(ep_meta)
        episode_dict.update(video_fields)
        new_meta._save_episode_metadata(episode_dict)

        total_frames += ep_len

    # finalize episode metadata and info
    new_meta._close_writer()
    new_meta.info.update({"total_episodes": len(labels_order), "total_frames": total_frames, "total_tasks": len(new_meta.tasks) if new_meta.tasks is not None else 0, "splits": {"train": f"0:{len(labels_order)}"}})
    write_info(new_meta.info, new_meta.root)

    # Optionally push whole dataset to HF hub
    results = {}
    if push_to_hub:
        try:
            hub = HfApi()
            repo_id = "steb6/" + repo_id.replace("-", "_").replace(" ", "_")
            hub.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
            hub.upload_folder(repo_id=repo_id, folder_path=str(out_dir), repo_type="dataset")
            results["dataset"] = str(out_dir) + " (pushed)"
        except Exception as e:
            results["dataset"] = str(out_dir) + f" (push failed: {e})"
    else:
        results["dataset"] = str(out_dir)

    return results

def main(argv: Iterable[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Dataset repo id or local path (HuggingFace dataset)")
    p.add_argument("--split", default="train", help="Dataset split or name to load from the repo")
    p.add_argument("--video-out-port-local", default="/splitter/video:o", help="YARP port to publish frames to")
    p.add_argument("--video-out-port-remote", default="/safsar/action_recognition/image:i", help="YARP port to publish frames to")
    p.add_argument("--label-in-port-local", default="/splitter/labels:i", help="Local YARP port to receive labels")
    p.add_argument("--label-in-port-remote", default="/safsar/action_recognition/action:o", help="Remote label publisher port to connect from (e.g. /action_detector/labels:o)")
    p.add_argument("--push-to-hub", action="store_true", help="Push the per-label datasets to the hub (requires --hf-token)")
    p.add_argument("--repo-id", default=None, help="Hugging Face Hub repo id (if pushing to hub)")
    p.add_argument("--img-field", default=None, help="Image field/key to use (e.g. observation.images.egocentric). If not provided, will try to auto-detect")
    args = p.parse_args(argv)

    ds = LeRobotDataset(args.dataset)
    streamer = YarpStreamer(args.video_out_port_local, args.video_out_port_remote, args.label_in_port_local, args.label_in_port_remote)
    if not os.path.exists("labels.json"):
        n = len(ds)
        labels = [None] * n
        try:
            for i in tqdm(range(n), desc="Streaming frames"):
                item = ds[i]
                img = pil_from_dataset_item(item, args.img_field)
                streamer.send_frame(i, img)
                # wait for label
                res = streamer.read_label()
                if res is None:
                    labels[i] = None
                else:
                    lbl = res
                    # accept label (we don't strictly require matching indices from detector)
                    labels[i] = lbl
                time.sleep(0.1)  # TODO FIND REAL FPS
        finally:
            streamer.close()
    else:
        import json
        with open("labels.json", "r") as f:
            labels = json.load(f)

    # # Split dataset by label and save
    results = split_one_episode_by_labels(
        ds,
        labels,
        output_prefix=args.repo_id if args.repo_id is not None else "split_dataset",
        push_to_hub=args.push_to_hub,
        hf_token=os.getenv("HF_TOKEN"),
    )
    # from lerobot.datasets.dataset_tools import split_dataset
    # splits = split_dataset(
    #     ds,
    #     splits={"train": 0.8, "val": 0.2},
    # )
    # results = split_dataset_by_label(ds, labels, args.repo_id, push_to_hub=args.push_to_hub)

    # print("Created per-label datasets:")
    # for k, v in results.items():
    #     print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
