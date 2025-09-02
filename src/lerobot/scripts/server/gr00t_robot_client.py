"""GR00T-integrated Robot Client.

This file started as a copy of `robot_client.py` (gRPC LeRobot policy server client) and was
refactored to talk to a GR00T inference service (ZMQ/HTTP) using the GR00T *interface client*.

Key design goals kept from original LeRobot client:
    - Asynchronous control loop vs. remote inference (decoupling action execution from network latency)
    - Action queue with aggregation functions
    - Latency & FPS diagnostics
    - Multi-camera handling (first camera mapped to ego view; can be extended)

Key changes:
    - Removed gRPC protobuf plumbing; replaced with a lightweight `RobotInterfaceClient` call pattern.
    - Added an internal inference thread consuming a queue of observations and producing TimedAction chunks.
    - On policy response, action tensors are converted into per‑timestep TimedAction objects and queued.
    - Added translation layer raw robot observation -> GR00T observation dict (minimal, assertive, concise).

Assumptions (can be revisited):
    - We use `RobotInterfaceClient` (exposes all endpoints); if unavailable fallback to `RobotInferenceClient`.
    - A single camera is mapped to `video.ego_view` (first enumerated camera); extend easily for more.
    - Joint positions flatten into one state array; we pick the first `state.*` modality matching dimension.
    - Task instruction (if provided) mapped to `annotation.human.action.task_description`.
    - Returned action keys like `action.left_arm` (or multiple) are concatenated (axis=1) to match robot action feature ordering when needed.

If richer grouping is needed later (e.g. arms, hands segmentation), a modality mapping json can be plugged in
inside `_build_modality_binding()`.
"""




# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
Example command:
```shell
python src/lerobot/scripts/server/robot_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pprint import pformat
from queue import Queue, Empty
from typing import Any, Dict

import draccus
import torch
import numpy as np

from lerobot.cameras.yarp.configuration_yarp import YarpCameraConfig

# CHANGE: Remove gRPC specific imports; we now rely on GR00T client library.
try:  # Minimal fallback logic, no defensive branching beyond import resolution.
    from gr00t.eval.robot_interface import RobotInterfaceClient as _Gr00tClient  # type: ignore
except ImportError:  # pragma: no cover - fallback if only RobotInferenceClient exists.
    from gr00t.eval.robot import RobotInferenceClient as _Gr00tClient  # type: ignore

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs.policies import PreTrainedConfig
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.scripts.server.configs import RobotClientConfig
from lerobot.scripts.server.constants import SUPPORTED_ROBOTS, DEFAULT_FPS
from lerobot.scripts.server.helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    validate_robot_cameras_for_policy,
    visualize_action_queue_size,
)
# CHANGE: Remove protobuf / transport utilities (unused after GR00T integration).

from dataclasses import dataclass, field
from lerobot.robots.ergocub.configuration_ergocub import ErgoCubConfig


@dataclass
class ErgoCubRobotClientConfig(RobotClientConfig):
    # Explicitly set all RobotClientConfig fields (using same defaults where they exist)
    policy_type: str = "_"
    pretrained_name_or_path: str = "_"
    actions_per_chunk: int = 50
    task: str = ""
    server_address: str = "localhost:5555"
    policy_device: str = "_"
    chunk_size_threshold: float = 0.5
    fps: int = DEFAULT_FPS
    aggregate_fn_name: str = "weighted_average"
    debug_visualize_queue_size: bool = False
    verify_robot_cameras: bool = True

    # Explicitly set every field of ErgoCubConfig, including inherited ones.
    robot: ErgoCubConfig = field(
        default_factory=lambda: ErgoCubConfig(
            # Inherited RobotConfig fields
            id=None,
            calibration_dir=None,
            # ErgoCubConfig fields (with their default values)
            name="ergocub",
            remote_prefix="/ergocub",
            local_prefix="/gr00t_client",
            cameras={"agentview": YarpCameraConfig(yarp_name="depthCamera/rgbImage:o", width=1280, height=720, fps=30)},
            use_left_arm=True,
            use_right_arm=True,
            use_neck=True,
            use_bimanual_controller=True,
            encoders_control_boards=["head", "left_arm", "right_arm", "torso"],
        )
    )


class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig):  # noqa: D401 (doc retained above)
        # CHANGE: Initialize without gRPC; prepare GR00T client & async inference pipeline.
        self.config = config
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        # LeRobot feature registry (still needed for action feature ordering / validation)
        self.lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        # Parse host:port for GR00T service
        host, port = config.server_address.split(":")
        self.host = host
        self.port = int(port)
        self.logger.info(f"Connecting GR00T interface client at {self.host}:{self.port}")
        self.policy_client = _Gr00tClient(host=self.host, port=self.port)

        # Client state & threading primitives
        self.shutdown_event = threading.Event()
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = config.actions_per_chunk  # CHANGE: seed with expected chunk size to avoid div by -1
        self._chunk_size_threshold = config.chunk_size_threshold

        # Queues
        self.action_queue: Queue[TimedAction] = Queue()
        self.action_queue_lock = threading.Lock()
        self.action_queue_size: list[int] = []
        self.observation_queue: Queue[TimedObservation] = Queue(maxsize=2)  # small buffer
        self.start_barrier = threading.Barrier(2)  # control loop + inference loop

        # Diagnostics
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)
        self.must_go = threading.Event()
        self.must_go.set()
        self.logger.info("Robot connected and GR00T client ready")

        # Modality binding prepared on `start()` (after fetching modality config)
        self._modality_binding: dict[str, Any] = {}
        self._state_key: str | None = None
        self._video_key: str | None = None

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def start(self) -> bool:
        # CHANGE: Fetch modality config from GR00T server & build local binding once.
        self.logger.info("Connecting to GR00T server: fetching modality config…")
        modality_config = self.policy_client.get_modality_config()
        self.logger.info("GR00T modality config received")
        self._build_modality_binding(modality_config)
        self.shutdown_event.clear()
        return True

    def stop(self):  # noqa: D401
        # CHANGE: Simplified shutdown (no channel/stream teardown needed).
        self.shutdown_event.set()
        self.robot.disconnect()
        self.logger.debug("Robot disconnected & client stopped")

    def queue_observation(self, obs: TimedObservation) -> None:
        # CHANGE: Replace gRPC streaming send with enqueue to local inference thread.
        self.observation_queue.put(obs)
        self.logger.debug(f"Enqueued observation #{obs.get_timestep()} (must_go={obs.must_go})")

    # ------ Modality & translation helpers ------
    def _build_modality_binding(self, modality_config: Dict[str, Any]):
        # Load modality.json for per-key specs
        import json
        import os
        modality_path = os.path.join(os.path.dirname(__file__), "modality.json")
        with open(modality_path, "r") as f:
            modality_specs = json.load(f)

        # Build state and video key mappings
        self._state_keys = list(modality_specs["state"].keys())
        self._state_slices = {k: (v["start"], v["end"]) for k, v in modality_specs["state"].items()}
        self._video_keys = list(modality_specs["video"].keys())
        self._video_map = {k: v["original_key"] for k, v in modality_specs["video"].items()}
        self._modality_binding = {"state": self._state_keys, "video": self._video_keys}
        self.logger.info(f"Modality binding: state keys={self._state_keys}, video keys={self._video_keys}")

    def _raw_obs_to_gr00t(self, raw_observation: RawObservation, task: str) -> dict:
        # Use modality.json specs to build full GR00T obs dict from flat raw keys
        assert hasattr(self, "_state_keys") and hasattr(self, "_state_slices"), "Modality binding not initialized."
        obs: dict[str, Any] = {}

        def _get_list(prefix: str, fields: list[str]) -> list[float]:
            vals: list[float] = []
            for f in fields:
                key = f"{prefix}.{f}"
                vals.append(float(raw_observation.get(key, 0.0)))
            return vals

        # Build all state keys using naming conventions
        for k in self._state_keys:
            if k.endswith("_position"):
                base_name = k[: -len("_position")]  # preserve underscores (e.g., left_arm)
                base = f"{base_name}.position"
                vec = _get_list(base, ["x", "y", "z"])  # 3
            elif k.endswith("_orientation"):
                base_name = k[: -len("_orientation")]  # preserve underscores
                base = f"{base_name}.orientation"
                vec = _get_list(base, ["qx", "qy", "qz", "qw"])  # 4
            elif k == "left_hand":
                vec = _get_list(
                    "left_fingers",
                    ["thumb_add", "thumb_oc", "index_add", "index_oc", "middle_oc", "ring_pinky_oc"],
                )  # 6
            elif k == "right_hand":
                vec = _get_list(
                    "right_fingers",
                    ["thumb_add", "thumb_oc", "index_add", "index_oc", "middle_oc", "ring_pinky_oc"],
                )  # 6
            else:
                # Fallback: use configured slice size to zero-fill
                start, end = self._state_slices[k]
                vec = [0.0] * (end - start)
            obs[f"state.{k}"] = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).numpy()

        # Build all video keys: prefer mapped original_key; also accept its basename and any configured camera keys
        for k in self._video_keys:
            val = None
            orig = self._video_map.get(k)
            candidates: list[str] = []
            if orig:
                candidates.append(orig)
                candidates.append(orig.split(".")[-1])  # basename like 'agentview'
            candidates.extend(list(self.robot.cameras.keys()))
            for c in candidates:
                if c in raw_observation:
                    val = raw_observation[c]
                    break
            if val is not None:
                # Convert to numpy and add leading time/batch dimension if missing
                if isinstance(val, torch.Tensor):
                    arr = val.detach().cpu().numpy()
                else:
                    try:
                        arr = np.asarray(val)
                    except Exception:
                        arr = val
                if isinstance(arr, np.ndarray):
                    if arr.ndim == 2:  # H, W -> 1, H, W, 1
                        arr = arr[..., None]
                        arr = arr[None, ...]
                    elif arr.ndim == 3:  # H, W, C -> 1, H, W, C
                        arr = arr[None, ...]
                    # if 4D, assume already (T, H, W, C)
                    obs[f"video.{k}"] = arr
                else:
                    obs[f"video.{k}"] = val

        # Task annotation
        if task:
            obs["annotation.human.action.task_description"] = [task]
        return obs

    def _gr00t_actions_to_timed(self, action_dict: dict, base_timestep: int) -> list[TimedAction]:
        # CHANGE: Flatten GR00T multi-modality action dict into per-timestep TimedAction list.
        # Collect all action.* arrays and concatenate along last dim preserving order.
        arrays = []
        for k, v in action_dict.items():
            if k.startswith("action."):
                arrays.append(torch.tensor(v))  # (K, Dk)
        assert arrays, "No action.* keys returned by server."
        actions_cat = torch.cat(arrays, dim=1) if len(arrays) > 1 else arrays[0]
        timed = []
        now = time.time()
        for i in range(actions_cat.shape[0]):
            timed.append(
                TimedAction(
                    timestamp=now,  # single timestamp (network latency already elapsed)
                    timestep=base_timestep + i + 1,
                    action=actions_cat[i],
                )
            )
        self.action_chunk_size = max(self.action_chunk_size, actions_cat.shape[0])
        return timed

    # ------ Inference thread ------
    def inference_loop(self):
        # CHANGE: Separate thread pulling observations & performing remote inference to retain asynchronicity.
        self.start_barrier.wait()
        self.logger.info("Inference loop thread starting")
        while self.running:
            try:
                obs = self.observation_queue.get(timeout=0.1)
            except Empty:
                continue
            t_req_start = time.perf_counter()
            gr00t_obs = self._raw_obs_to_gr00t(obs.get_observation(), task=obs.get_observation().get("task", ""))
            action_dict = self.policy_client.get_action(gr00t_obs)
            rpc_latency_ms = (time.perf_counter() - t_req_start) * 1000
            with self.latest_action_lock:
                base_ts = self.latest_action
            timed_actions = self._gr00t_actions_to_timed(action_dict, base_ts)
            # Aggregate into queue
            self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
            self.must_go.set()  # allow next must-go observation once queue empties
            self.logger.info(
                f"Received {len(timed_actions)} actions (rpc_latency={rpc_latency_ms:.1f}ms) -> queue size {self.action_queue.qsize()}"  # noqa: E501
            )
            self.observation_queue.task_done()

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            # New action is older than the latest action in the queue, skip it
            if new_action.get_timestep() <= latest_action:
                continue

            # If the new action's timestep is not in the current action queue, add it directly
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            # If the new action's timestep is in the current action queue, aggregate it
            # TODO: There is probably a way to do this with broadcasting of the two action tensors
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    # CHANGE: `receive_actions` removed (handled by `inference_loop`).

    def actions_available(self):
        """Check if there are actions available in the queue"""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        """Reading and performing actions in local queue"""

        # Lock only for queue operations
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            # Get action from queue
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        _performed_action = self.robot.send_action(
            self._action_tensor_to_action_dict(timed_action.get_action())
        )
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return _performed_action

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        # CHANGE: Guard first-iteration case (action_chunk_size may still be default); always allow first send.
        with self.action_queue_lock:
            if self.action_chunk_size <= 0:
                return True
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            raw_observation: RawObservation = self.robot.get_observation()
            raw_observation["task"] = task

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing!
            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            # CHANGE: enqueue observation for async inference instead of gRPC send.
            self.queue_observation(observation)

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                # must-go event will be set again after receiving actions
                self.must_go.clear()

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations"""
        # CHANGE: Barrier kept (sync with inference thread start).
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()
            # (1) Perform actions if available
            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)
            # (2) Send observation when queue below threshold
            if self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(task, verbose)
            self.logger.info(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


@draccus.wrap()
def async_client(cfg: ErgoCubRobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    if cfg.robot.type not in SUPPORTED_ROBOTS:
        raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClient(cfg)

    client.logger.info("Initializing client: calling start()")
    if client.start():
        client.logger.info("Client start() returned successfully")
        client.logger.info("Starting inference thread...")
        # CHANGE: Start inference loop thread instead of gRPC action receiver.
        inference_thread = threading.Thread(target=client.inference_loop, daemon=True)
        inference_thread.start()

        try:
            client.control_loop(task=cfg.task)
        finally:
            client.stop()
            inference_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


if __name__ == "__main__":
    async_client()  # run the client
