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
from pathlib import Path
import logging
import importlib
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
import rerun as rr
import torch

from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.rotation import Rotation, rotation_6d_to_matrix

from .constants import ACTION, ACTION_PREFIX, OBS_PREFIX, OBS_STR


_ERGOCUB_URDF_LOGGED = False
_ERGOCUB_MODEL_WARNED = False
_ERGOCUB_SCENE_EXPORT_WARNED = False
_ERGOCUB_URDF_PATH: str | None = None
_ERGOCUB_URDF_MODEL: Any | None = None
_ERGOCUB_SCENE_FRAME_IDX = 0
logger = logging.getLogger(__name__)


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
    robot: Any | None = None,
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


def _require_pose_dict(data: dict[str, float] | None, prefix: str, source_name: str) -> tuple[np.ndarray, np.ndarray]:
    pose = _extract_pose_dict(data, prefix)
    if pose is not None:
        return pose

    available_keys = sorted(k for k in (data or {}).keys() if str(k).startswith(prefix))
    raise ValueError(
        f"Missing required {source_name} pose for '{prefix}'. "
        f"Expected keys: {prefix}.position.(x|y|z) and {prefix}.orientation.(d1..d6). "
        f"Available matching keys: {available_keys}"
    )


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


def _resolve_visual_asset_path(urdf_path: str) -> str | None:
    urdf_file = Path(urdf_path)
    if urdf_file.suffix.lower() != ".urdf":
        return str(urdf_file)

    # Fast path: same basename next to the URDF.
    for suffix in (".glb", ".gltf", ".obj", ".stl"):
        candidate = urdf_file.with_suffix(suffix)
        if candidate.exists():
            return str(candidate)

    # Fallback: parse URDF and try to resolve mesh file references.
    def _resolve_mesh_reference(mesh_ref: str) -> Path | None:
        ref = mesh_ref.strip()
        if not ref:
            return None

        if ref.startswith("file://"):
            candidate = Path(ref.removeprefix("file://"))
            return candidate if candidate.exists() else None

        if ref.startswith("package://") or ref.startswith("model://"):
            remainder = ref.split("://", 1)[1]
            if "/" not in remainder:
                return None
            pkg_name, rel_path = remainder.split("/", 1)

            # Typical ROS / robotology installation layout:
            # .../share/<pkg_name>/...
            share_root = next((p for p in urdf_file.parents if p.name == "share"), None)
            if share_root is not None:
                candidate = share_root / pkg_name / rel_path
                if candidate.exists():
                    return candidate

            # Fallback: find an ancestor matching pkg_name.
            pkg_root = next((p for p in urdf_file.parents if p.name == pkg_name), None)
            if pkg_root is not None:
                candidate = pkg_root / rel_path
                if candidate.exists():
                    return candidate

            return None

        candidate = (urdf_file.parent / ref).resolve()
        return candidate if candidate.exists() else None

    try:
        root = ET.parse(urdf_file).getroot()
    except Exception:
        return None

    mesh_paths: list[Path] = []
    for mesh_el in root.findall(".//mesh"):
        filename = mesh_el.attrib.get("filename")
        if not filename:
            continue
        resolved = _resolve_mesh_reference(filename)
        if resolved is not None:
            mesh_paths.append(resolved)

    # Prefer directly renderable mesh formats.
    for mesh_path in mesh_paths:
        if mesh_path.suffix.lower() in (".glb", ".gltf", ".obj", ".stl"):
            return str(mesh_path)

    # If URDF points to unsupported formats (e.g. .dae), try sibling conversions.
    for mesh_path in mesh_paths:
        for suffix in (".glb", ".gltf", ".obj", ".stl"):
            candidate = mesh_path.with_suffix(suffix)
            if candidate.exists():
                return str(candidate)

    return None


def _export_urdf_scene_glb(
    urdf_path: str,
    joint_values: dict[str, float] | None = None,
    slot: int | None = None,
) -> str:
    """Export full URDF visual scene to GLB and return the exported path.

    This allows rendering the whole robot (all links/meshes) as a single static asset
    in Rerun, instead of showing only one mesh file.
    """

    global _ERGOCUB_SCENE_EXPORT_WARNED
    global _ERGOCUB_URDF_MODEL
    global _ERGOCUB_URDF_PATH

    try:
        trimesh = importlib.import_module("trimesh")
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency `trimesh` required for full ergoCub URDF rendering."
        ) from exc

    try:
        yourdfpy = importlib.import_module("yourdfpy")
        URDF = yourdfpy.URDF
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency `yourdfpy` required for full ergoCub URDF rendering."
        ) from exc

    try:
        if _ERGOCUB_URDF_MODEL is None or _ERGOCUB_URDF_PATH != urdf_path:
            _ERGOCUB_URDF_MODEL = URDF.load(urdf_path, build_scene_graph=True, load_meshes=True)
            _ERGOCUB_URDF_PATH = urdf_path

        urdf = _ERGOCUB_URDF_MODEL

        if joint_values:
            # YARP state ports are often in degrees; yourdfpy expects radians.
            vals = {k: float(v) for k, v in joint_values.items()}
            if any(abs(v) > 9.5 for v in vals.values()):
                vals = {k: np.deg2rad(v) for k, v in vals.items()}

            if hasattr(urdf, "actuated_joint_names") and getattr(urdf, "actuated_joint_names"):
                cfg = [float(vals.get(name, 0.0)) for name in urdf.actuated_joint_names]
                urdf.update_cfg(cfg)
            else:
                urdf.update_cfg(vals)

        scene = getattr(urdf, "scene", None)
        if scene is None:
            raise RuntimeError("URDF scene graph is empty; unable to export full ergoCub model.")

        cache_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "rerun"
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = cache_dir / (f"ergocub_scene_{slot}.glb" if slot is not None else "ergocub_scene.glb")

        glb_bytes = trimesh.exchange.gltf.export_glb(scene)
        out_path.write_bytes(glb_bytes)
        return str(out_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to export full URDF scene as GLB: {exc}") from exc


def _log_ergocub_skeleton(actual_left, actual_right, target_left, target_right) -> None:
    torso_origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    head_anchor = np.array([0.0, 0.0, 0.3], dtype=np.float32)

    actual_strips = [[torso_origin, head_anchor]]
    target_strips = [[torso_origin, head_anchor]]

    if actual_left is not None:
        actual_strips.append([torso_origin, actual_left[0]])
    if actual_right is not None:
        actual_strips.append([torso_origin, actual_right[0]])
    if target_left is not None:
        target_strips.append([torso_origin, target_left[0]])
    if target_right is not None:
        target_strips.append([torso_origin, target_right[0]])

    rr.log("ergocub/body/actual", rr.LineStrips3D(actual_strips), static=False)
    rr.log("ergocub/body/target", rr.LineStrips3D(target_strips), static=False)


def _log_ergocub_joint_states(robot: Any | None = None, controllers: dict[str, Any] | None = None) -> None:
    """Log torso/arm/head joint scalars if controllers expose latest joint states."""
    if controllers is None and robot is not None:
        bus = getattr(robot, "bus", None)
        controllers = getattr(bus, "controllers", None)

    if not controllers:
        return

    for ctrl_name in ("bimanual", "head"):
        ctrl = controllers.get(ctrl_name)
        if ctrl is None or not hasattr(ctrl, "get_latest_joint_states"):
            continue

        joint_states = ctrl.get_latest_joint_states()
        for joint_name, value in joint_states.items():
            rr.log(f"ergocub/joints/{ctrl_name}/{joint_name}", rr.Scalars(float(value)))


def _collect_ergocub_joint_states(robot: Any | None = None, controllers: dict[str, Any] | None = None) -> dict[str, float]:
    if controllers is None and robot is not None:
        bus = getattr(robot, "bus", None)
        controllers = getattr(bus, "controllers", None)

    if not controllers:
        raise RuntimeError("ErgoCub joint-state visualization requires robot controllers.")

    merged: dict[str, float] = {}
    for ctrl_name in ("bimanual", "head"):
        if ctrl_name not in controllers:
            raise RuntimeError(f"Missing required controller '{ctrl_name}' for ErgoCub joint-state visualization.")

        ctrl = controllers[ctrl_name]
        if not hasattr(ctrl, "get_latest_joint_states"):
            raise RuntimeError(
                f"Controller '{ctrl_name}' must implement get_latest_joint_states() for ErgoCub model animation."
            )

        ctrl_joints = ctrl.get_latest_joint_states()
        if not ctrl_joints:
            raise RuntimeError(
                f"Controller '{ctrl_name}' returned no joint states. Ensure YARP state ports are connected and readable."
            )
        merged.update(ctrl_joints)

    return merged


def log_rerun_data_ergocub(
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
    robot: Any | None = None,
    controllers: dict[str, Any] | None = None,
) -> None:
    """
    Logs standard observation/action streams and ErgoCub-specific 3D entities to Rerun.

    Adds:
    - static ErgoCub full URDF scene (`ergocub/model`)
    - target frames from action (`ergocub/frames/target/*`)
    - actual frames from observation (`ergocub/frames/actual/*`)

    Strict behavior:
    - Raises on URDF resolution/export failures
    - Raises if any required target/actual pose keys are missing
    - No fallback rendering path
    """
    global _ERGOCUB_URDF_LOGGED
    global _ERGOCUB_MODEL_WARNED
    global _ERGOCUB_SCENE_FRAME_IDX
    global _ERGOCUB_URDF_PATH

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

        # Do not log as static here; model is updated dynamically each frame below.
        _ERGOCUB_URDF_PATH = urdf_path
        _ERGOCUB_URDF_LOGGED = True

    urdf_path = _ERGOCUB_URDF_PATH
    if not urdf_path:
        raise RuntimeError("ErgoCub URDF path not initialized for visualization.")

    joint_states = _collect_ergocub_joint_states(robot=robot, controllers=controllers)
    _log_ergocub_joint_states(robot=robot, controllers=controllers)

    visual_asset_path = _export_urdf_scene_glb(
        urdf_path,
        joint_values=joint_states,
        slot=_ERGOCUB_SCENE_FRAME_IDX,
    )
    _ERGOCUB_SCENE_FRAME_IDX += 1
    rr.log("ergocub/model", rr.Asset3D(path=visual_asset_path), static=False)

    target_left = _require_pose_dict(action, "left_hand", "target(action)")
    target_right = _require_pose_dict(action, "right_hand", "target(action)")
    actual_left = _require_pose_dict(observation, "left_hand", "actual(observation)")
    actual_right = _require_pose_dict(observation, "right_hand", "actual(observation)")

    _log_ergocub_pose_frame("ergocub/frames/target/left_hand", target_left)
    _log_ergocub_pose_frame("ergocub/frames/target/right_hand", target_right)
    _log_ergocub_pose_frame("ergocub/frames/actual/left_hand", actual_left)
    _log_ergocub_pose_frame("ergocub/frames/actual/right_hand", actual_right)

    _log_ergocub_joint_states(robot=robot, controllers=controllers)
    _log_ergocub_skeleton(actual_left, actual_right, target_left, target_right)
