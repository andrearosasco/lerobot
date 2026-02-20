# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numbers
import os

import numpy as np
import rerun as rr
import torch

from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.rotation import Rotation, rotation_6d_to_matrix

from .constants import ACTION, ACTION_PREFIX, OBS_PREFIX, OBS_STR


_ERGOCUB_URDF_LOGGED = False


def init_rerun(
    session_name: str = "lerobot_control_loop", ip: str | None = None, port: int | None = None
) -> None:
    """
    Initializes the Rerun SDK for visualizing the control loop.

    Args:
        session_name: Name of the Rerun session.
        ip: Optional IP for connecting to a Rerun server.
        port: Optional port for connecting to a Rerun server.
    """
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    if ip and port:
        rr.connect_grpc(url=f"rerun+http://{ip}:{port}/proxy")
    else:
        rr.spawn(memory_limit=memory_limit)


def _is_scalar(x):
    return isinstance(x, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


def log_rerun_data(
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
) -> None:
    """
    Logs observation and action data to Rerun for real-time visualization.

    This function iterates through the provided observation and action dictionaries and sends their contents
    to the Rerun viewer. It handles different data types appropriately:
    - Scalars values (floats, ints) are logged as `rr.Scalars`.
    - 3D NumPy arrays that resemble images (e.g., with 1, 3, or 4 channels first) are transposed
      from CHW to HWC format, (optionally) compressed to JPEG and logged as `rr.Image` or `rr.EncodedImage`.
    - 1D NumPy arrays are logged as a series of individual scalars, with each element indexed.
    - Other multi-dimensional arrays are flattened and logged as individual scalars.

    Keys are automatically namespaced with "observation." or "action." if not already present.

    Args:
        observation: An optional dictionary containing observation data to log.
        action: An optional dictionary containing action data to log.
        compress_images: Whether to compress images before logging to save bandwidth & memory in exchange for cpu and quality.
    """
    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                arr = v
                # Convert CHW -> HWC when needed
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 1:
                    for i, vi in enumerate(arr):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    img_entity = rr.Image(arr).compress() if compress_images else rr.Image(arr)
                    rr.log(key, entity=img_entity, static=True)

    if action:
        for k, v in action.items():
            if v is None:
                continue
            key = k if str(k).startswith(ACTION_PREFIX) else f"{ACTION}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                if v.ndim == 1:
                    for i, vi in enumerate(v):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    # Fall back to flattening higher-dimensional arrays
                    flat = v.flatten()
                    for i, vi in enumerate(flat):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))


def _extract_pose_dict(data: dict[str, float] | None, prefix: str) -> tuple[np.ndarray, np.ndarray] | None:
    if not data:
        return None

    pos_keys = [f"{prefix}.position.x", f"{prefix}.position.y", f"{prefix}.position.z"]
    rot_keys = [
        f"{prefix}.orientation.d1",
        f"{prefix}.orientation.d2",
        f"{prefix}.orientation.d3",
        f"{prefix}.orientation.d4",
        f"{prefix}.orientation.d5",
        f"{prefix}.orientation.d6",
    ]

    if not all(k in data for k in pos_keys + rot_keys):
        return None

    position = np.array([float(data[k]) for k in pos_keys], dtype=np.float32)
    rot6d = np.array([float(data[k]) for k in rot_keys], dtype=np.float32)
    rotation_matrix = rotation_6d_to_matrix(torch.tensor(rot6d, dtype=torch.float32)).numpy()
    quaternion_xyzw = Rotation.from_matrix(rotation_matrix).as_quat().astype(np.float32)
    return position, quaternion_xyzw


def _log_ergocub_pose_frame(entity_path: str, pose: tuple[np.ndarray, np.ndarray]) -> None:
    position, quaternion_xyzw = pose
    rr.log(
        entity_path,
        rr.Transform3D(
            translation=position,
            rotation=rr.Quaternion(xyzw=quaternion_xyzw),
            axis_length=0.08,
        ),
    )


def log_rerun_data_ergocub(
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
) -> None:
    """
    Logs standard observation/action streams and ErgoCub-specific 3D entities to Rerun.

    Adds:
    - static ErgoCub URDF (`ergocub/model`)
    - target frames from action (`ergocub/frames/target/*`)
    - actual frames from observation (`ergocub/frames/actual/*`)
    """
    global _ERGOCUB_URDF_LOGGED

    log_rerun_data(observation=observation, action=action, compress_images=compress_images)

    if not _ERGOCUB_URDF_LOGGED:
        try:
            from lerobot.motors.ergocub.urdf_utils import resolve_ergocub_urdf

            urdf_path = resolve_ergocub_urdf()
        except Exception as exc:
            raise RuntimeError(
                "Failed to resolve ergoCub URDF for Rerun visualization. "
                "Set ROBOT_URDF_PATH to a valid URDF file."
            ) from exc

        if not urdf_path or not os.path.exists(urdf_path):
            raise FileNotFoundError(
                "Failed to resolve ergoCub URDF for Rerun visualization. "
                "Set ROBOT_URDF_PATH to a valid URDF file."
            )

        rr.log("ergocub/model", rr.Asset3D(path=urdf_path), static=True)
        _ERGOCUB_URDF_LOGGED = True

    target_left = _extract_pose_dict(action, "left_hand")
    target_right = _extract_pose_dict(action, "right_hand")
    actual_left = _extract_pose_dict(observation, "left_hand")
    actual_right = _extract_pose_dict(observation, "right_hand")

    if target_left is not None:
        _log_ergocub_pose_frame("ergocub/frames/target/left_hand", target_left)
    if target_right is not None:
        _log_ergocub_pose_frame("ergocub/frames/target/right_hand", target_right)
    if actual_left is not None:
        _log_ergocub_pose_frame("ergocub/frames/actual/left_hand", actual_left)
    if actual_right is not None:
        _log_ergocub_pose_frame("ergocub/frames/actual/right_hand", actual_right)
