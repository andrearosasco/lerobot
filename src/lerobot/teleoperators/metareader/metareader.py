import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceNotConnectedError

from .config_metareader import MetaReaderConfig

repo_root = Path(__file__).resolve().parents[4]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import metareader


TIPS = ("thumb", "index", "middle", "ring", "little")


def _tip_features(prefix: str = "") -> dict[str, type[float]]:
    return {f"{prefix}fingertip.{tip}.{axis}": float for tip in TIPS for axis in "xyz"}


class MetaReaderTeleoperator(Teleoperator):
    config_class = MetaReaderConfig
    name = "metareader"

    def __init__(self, config: MetaReaderConfig):
        self.config = config
        self._reader = None
        self._is_connected = False
        self._last_action = self._neutral_action(0.0)
        super().__init__(config)

    @property
    def action_features(self) -> dict:
        return {
            "position.x": float,
            "position.y": float,
            "position.z": float,
            "orientation.x": float,
            "orientation.y": float,
            "orientation.z": float,
            "gripper": float,
            "is_engaged": float,
            "exit_episode": float,
            "discard_episode": float,
            **_tip_features(),
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if self._is_connected:
            return
        self._reader = metareader.MetaReader(
            port=self.config.port,
            tcp_port=self.config.tcp_port,
            no_advertise=self.config.no_advertise,
            auto_adb_reverse=self.config.auto_adb_reverse,
        )
        if hasattr(self._reader, "__enter__"):
            self._reader.__enter__()
        deadline = time.monotonic() + self.config.connection_timeout_s
        while time.monotonic() < deadline:
            frame = self._reader.read_latest(timeout=self.config.read_timeout_s)
            if frame is not None:
                self._last_action = self._frame_to_action(frame)
                self._is_connected = True
                return
        self.disconnect()
        raise TimeoutError("MetaReader did not produce any frame before timeout.")

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, Any]:
        if not self._is_connected or self._reader is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        frame = self._reader.read_latest(timeout=self.config.read_timeout_s)
        if frame is None:
            return {**self._last_action, "is_engaged": 0.0}
        self._last_action = self._frame_to_action(frame)
        return dict(self._last_action)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback

    def disconnect(self) -> None:
        if self._reader is not None and hasattr(self._reader, "__exit__"):
            self._reader.__exit__(None, None, None)
        self._reader = None
        self._is_connected = False

    def _neutral_action(self, engaged: float) -> dict[str, float]:
        return {
            "position.x": 0.0,
            "position.y": 0.0,
            "position.z": 0.0,
            "orientation.x": 0.0,
            "orientation.y": 0.0,
            "orientation.z": 0.0,
            "gripper": 1.0,
            "is_engaged": engaged,
            "exit_episode": 0.0,
            "discard_episode": 0.0,
            **{key: 0.0 for key in _tip_features()},
        }

    def _frame_to_action(self, frame: Any) -> dict[str, float]:
        hand = frame.right_hand
        palm = getattr(hand.palm, "pose", None)
        engaged = float(hand.tracked or not self.config.require_tracked_right_hand)
        if engaged == 0.0 or palm is None or not palm.valid:
            return self._neutral_action(engaged)

        rotvec = R.from_quat(palm.orientation).as_rotvec()
        palm_position = np.asarray(palm.position, dtype=float)
        palm_inverse = R.from_quat(palm.orientation).inv()
        action = self._neutral_action(engaged)
        action.update(
            {
                "position.x": float(palm_position[0]),
                "position.y": float(palm_position[1]),
                "position.z": float(palm_position[2]),
                "orientation.x": float(rotvec[0]),
                "orientation.y": float(rotvec[1]),
                "orientation.z": float(rotvec[2]),
            }
        )

        thumb = hand.fingertips.get("thumb_tip")
        index = hand.fingertips.get("index_tip")
        if all(getattr(tip, "pose", None) and tip.pose.valid for tip in (thumb, index)):
            pinch = math.dist(thumb.pose.position, index.pose.position)
            span = self.config.pinch_open_distance_m - self.config.pinch_close_distance_m
            action["gripper"] = float(np.clip((pinch - self.config.pinch_close_distance_m) / span, 0.0, 1.0))
        else:
            action["gripper"] = self._last_action["gripper"]

        for tip in TIPS:
            fingertip = hand.fingertips.get(f"{tip}_tip")
            pose = getattr(fingertip, "pose", None)
            if pose is None or not pose.valid:
                continue
            relative = palm_inverse.apply(np.asarray(pose.position, dtype=float) - palm_position)
            action[f"fingertip.{tip}.x"] = float(relative[0])
            action[f"fingertip.{tip}.y"] = float(relative[1])
            action[f"fingertip.{tip}.z"] = float(relative[2])
        return action
        return action
