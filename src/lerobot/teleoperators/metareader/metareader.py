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

try:
    import metareader as metareader_sdk
except ImportError as exc:
    repo_root = Path(__file__).resolve().parents[4]
    workspace_metareader = repo_root / "metareader"
    if workspace_metareader.is_dir() and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
        try:
            import metareader as metareader_sdk
        except ImportError:
            metareader_sdk = None
            _metareader_import_error = exc
        else:
            _metareader_import_error = None
    else:
        metareader_sdk = None
        _metareader_import_error = exc


FINGERTIP_NAMES = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip")
FINGERTIP_PREFIX = {
    "thumb_tip": "thumb",
    "index_tip": "index",
    "middle_tip": "middle",
    "ring_tip": "ring",
    "little_tip": "little",
}


def _vector_sub(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _vector_norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _vector_add(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _vector_scale(vector: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return (vector[0] * scale, vector[1] * scale, vector[2] * scale)


def _vector_cross(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float] | None:
    norm = _vector_norm(vector)
    if norm <= 1e-8:
        return None
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _rotation_matrix_to_quaternion(
    rotation: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
) -> tuple[float, float, float, float]:
    r00, r01, r02 = rotation[0]
    r10, r11, r12 = rotation[1]
    r20, r21, r22 = rotation[2]
    trace = r00 + r11 + r22

    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return (
            (r21 - r12) / scale,
            (r02 - r20) / scale,
            (r10 - r01) / scale,
            0.25 * scale,
        )
    if r00 > r11 and r00 > r22:
        scale = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        return (
            0.25 * scale,
            (r01 + r10) / scale,
            (r02 + r20) / scale,
            (r21 - r12) / scale,
        )
    if r11 > r22:
        scale = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        return (
            (r01 + r10) / scale,
            0.25 * scale,
            (r12 + r21) / scale,
            (r02 - r20) / scale,
        )

    scale = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
    return (
        (r02 + r20) / scale,
        (r12 + r21) / scale,
        0.25 * scale,
        (r10 - r01) / scale,
    )


def _tracked_fingertip_positions(hand: Any) -> list[tuple[float, float, float]]:
    positions: list[tuple[float, float, float]] = []
    for fingertip in hand.fingertips.values():
        if fingertip.tracked and fingertip.pose is not None and fingertip.pose.valid:
            positions.append(fingertip.pose.position)
    return positions


def _synthesize_hand_pose_from_fingertips(hand: Any) -> Any | None:
    points = _tracked_fingertip_positions(hand)
    if len(points) < 2 or metareader_sdk is None:
        return None

    origin = (0.0, 0.0, 0.0)
    for point in points:
        origin = _vector_add(origin, point)
    origin = _vector_scale(origin, 1.0 / len(points))

    index_tip = hand.fingertips.get("index_tip")
    little_tip = hand.fingertips.get("little_tip")
    middle_tip = hand.fingertips.get("middle_tip")

    across = None
    if index_tip and index_tip.pose and index_tip.pose.valid and little_tip and little_tip.pose and little_tip.pose.valid:
        across = _normalize(_vector_sub(index_tip.pose.position, little_tip.pose.position))
    elif len(points) >= 2:
        across = _normalize(_vector_sub(points[0], points[-1]))

    forward = None
    if middle_tip and middle_tip.pose and middle_tip.pose.valid:
        forward = _normalize(_vector_sub(middle_tip.pose.position, origin))
    elif hand.wrist.pose is not None and hand.wrist.pose.valid:
        forward = _normalize(_vector_sub(origin, hand.wrist.pose.position))

    if across is None or forward is None:
        return metareader_sdk.Pose(position=origin, orientation=(0.0, 0.0, 0.0, 1.0), valid=True)

    normal = _normalize(_vector_cross(across, forward))
    if normal is None:
        return metareader_sdk.Pose(position=origin, orientation=(0.0, 0.0, 0.0, 1.0), valid=True)

    forward = _normalize(_vector_cross(normal, across))
    if forward is None:
        return metareader_sdk.Pose(position=origin, orientation=(0.0, 0.0, 0.0, 1.0), valid=True)

    return metareader_sdk.Pose(
        position=origin,
        orientation=_rotation_matrix_to_quaternion((across, forward, normal)),
        valid=True,
    )


def _hand_reference_pose(hand: Any) -> Any | None:
    for joint in (hand.palm, hand.wrist):
        if joint.tracked and joint.pose is not None and joint.pose.valid:
            return joint.pose
    return _synthesize_hand_pose_from_fingertips(hand)


def _tip_feature_keys(prefix: str = "") -> dict[str, type[float]]:
    features: dict[str, type[float]] = {}
    for tip_name in FINGERTIP_NAMES:
        key_root = f"{prefix}fingertip.{FINGERTIP_PREFIX[tip_name]}"
        for axis in ("x", "y", "z"):
            features[f"{key_root}.{axis}"] = float
    return features


class MetaReaderTeleoperator(Teleoperator):
    config_class = MetaReaderConfig
    name = "metareader"

    def __init__(self, config: MetaReaderConfig):
        self.config = config
        self._reader = None
        self._is_connected = False
        self._last_action = self._neutral_action(is_engaged=0.0)
        super().__init__(config)

    @property
    def action_features(self) -> dict:
        features = {
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
        }
        features.update(_tip_feature_keys())
        return features

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if self._is_connected:
            return
        if metareader_sdk is None:
            raise ImportError(
                "MetaReader teleoperator requires the 'metareader' package in the active lerobot environment."
            ) from _metareader_import_error

        self._reader = metareader_sdk.MetaReader(
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

        if hasattr(self._reader, "__exit__"):
            self._reader.__exit__(None, None, None)
        self._reader = None
        raise TimeoutError("MetaReader did not produce any frame before the connection timeout elapsed.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected or self._reader is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        frame = self._reader.read_latest(timeout=self.config.read_timeout_s)
        if frame is None:
            stale_action = dict(self._last_action)
            stale_action["is_engaged"] = 0.0
            return stale_action

        self._last_action = self._frame_to_action(frame)
        return dict(self._last_action)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback

    def disconnect(self) -> None:
        if self._reader is not None and hasattr(self._reader, "__exit__"):
            self._reader.__exit__(None, None, None)
        self._reader = None
        self._is_connected = False

    def _neutral_action(self, is_engaged: float) -> dict[str, float]:
        action = {
            "position.x": 0.0,
            "position.y": 0.0,
            "position.z": 0.0,
            "orientation.x": 0.0,
            "orientation.y": 0.0,
            "orientation.z": 0.0,
            "gripper": 1.0,
            "is_engaged": is_engaged,
            "exit_episode": 0.0,
            "discard_episode": 0.0,
        }
        for key in _tip_feature_keys():
            action[key] = 0.0
        return action

    def _frame_to_action(self, frame: Any) -> dict[str, float]:
        right_hand = frame.right_hand
        hand_pose = _hand_reference_pose(right_hand)
        palm_pose = right_hand.palm.pose if right_hand.palm.tracked and right_hand.palm.pose is not None and right_hand.palm.pose.valid else hand_pose
        if hand_pose is None or palm_pose is None or not hand_pose.valid or not palm_pose.valid:
            return self._neutral_action(is_engaged=0.0)

        axis_angle = R.from_quat(hand_pose.orientation).as_rotvec()
        action = {
            "position.x": float(hand_pose.position[0]),
            "position.y": float(hand_pose.position[1]),
            "position.z": float(hand_pose.position[2]),
            "orientation.x": float(axis_angle[0]),
            "orientation.y": float(axis_angle[1]),
            "orientation.z": float(axis_angle[2]),
            "is_engaged": float((not self.config.require_tracked_right_hand) or right_hand.tracked),
            "exit_episode": 0.0,
            "discard_episode": 0.0,
        }

        thumb_tip = right_hand.fingertips.get("thumb_tip")
        index_tip = right_hand.fingertips.get("index_tip")
        pinch_distance = None
        if (
            thumb_tip is not None
            and thumb_tip.pose is not None
            and thumb_tip.pose.valid
            and index_tip is not None
            and index_tip.pose is not None
            and index_tip.pose.valid
        ):
            pinch_distance = _vector_norm(_vector_sub(index_tip.pose.position, thumb_tip.pose.position))

        if pinch_distance is None:
            action["gripper"] = self._last_action.get("gripper", 1.0)
        else:
            span = max(self.config.pinch_open_distance_m - self.config.pinch_close_distance_m, 1e-6)
            openness = (pinch_distance - self.config.pinch_close_distance_m) / span
            action["gripper"] = float(np.clip(openness, 0.0, 1.0))

        for tip_name in FINGERTIP_NAMES:
            tip_prefix = FINGERTIP_PREFIX[tip_name]
            for axis in ("x", "y", "z"):
                action[f"fingertip.{tip_prefix}.{axis}"] = 0.0

            fingertip = right_hand.fingertips.get(tip_name)
            if fingertip is None or fingertip.pose is None or not fingertip.pose.valid:
                continue

            relative_world = np.asarray(_vector_sub(fingertip.pose.position, palm_pose.position), dtype=float)
            relative_palm = R.from_quat(palm_pose.orientation).inv().apply(relative_world)
            action[f"fingertip.{tip_prefix}.x"] = float(relative_palm[0])
            action[f"fingertip.{tip_prefix}.y"] = float(relative_palm[1])
            action[f"fingertip.{tip_prefix}.z"] = float(relative_palm[2])

        return action
