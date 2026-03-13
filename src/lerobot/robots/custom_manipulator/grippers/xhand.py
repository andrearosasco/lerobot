import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..configs import GripperConfig

try:
    from xhand_controller import xhand_control
except ImportError as exc:
    xhand_control = None
    _xhand_import_error = exc

try:
    from klampt import WorldModel
    from klampt.math import so3, vectorops
    from klampt.model import ik
except ImportError as exc:
    WorldModel = None
    so3 = None
    vectorops = None
    ik = None
    _klampt_import_error = exc


LOGGER = logging.getLogger(__name__)
TIP_NAMES = ("thumb", "index", "middle", "ring", "little")


def _tip_feature_keys(prefix: str = "") -> dict[str, type[float]]:
    features: dict[str, type[float]] = {}
    for tip_name in TIP_NAMES:
        for axis in ("x", "y", "z"):
            features[f"{prefix}fingertip.{tip_name}.{axis}"] = float
    return features


@GripperConfig.register_subclass("xhand")
@dataclass
class XHandConfig(GripperConfig):
    protocol: str = "RS485"
    serial_port: str = "/dev/ttyUSB0"
    baud_rate: int = 3000000
    hand_id: int = 0
    control_mode: int = 3
    kp: float = 100.0
    ki: float = 0.0
    kd: float = 0.0
    tor_max: float = 300.0
    poll_force_update: bool = True
    enable_tip_ik: bool = True
    require_ik: bool = True
    urdf_path: str | None = None
    palm_link_name: str = "palm"
    tip_link_names: dict[str, str] = field(
        default_factory=lambda: {
            "thumb": "thumb_tip",
            "index": "index_tip",
            "middle": "middle_tip",
            "ring": "ring_tip",
            "little": "little_tip",
        }
    )
    niter: int = 20000
    open_positions: list[float] = field(default_factory=lambda: [0.0] * 12)
    closed_positions: list[float] = field(default_factory=lambda: [0.8] * 12)
    startup_delay_s: float = 1.0
    connect_reset: bool = False
    command_retries: int = 3
    retry_delay_s: float = 0.5
    auto_detect_hand_id: bool = True

    @property
    def type(self) -> str:
        return "xhand"


class XHandKinematics:
    def __init__(self, config: XHandConfig):
        if WorldModel is None or ik is None or so3 is None or vectorops is None:
            raise ImportError(
                "xHand fingertip IK requires the 'klampt' package in the active lerobot environment."
            ) from _klampt_import_error
        if not config.urdf_path:
            raise ValueError("XHandConfig.urdf_path must point to the xHand URDF when enable_tip_ik is True.")

        urdf_path = Path(config.urdf_path)
        if not urdf_path.is_file():
            raise FileNotFoundError(f"xHand URDF was not found at {urdf_path}")

        self.config = config
        self.world = WorldModel()
        load_result = self.world.loadRobot(str(urdf_path))
        if load_result < 0:
            raise RuntimeError(f"Failed to load xHand URDF from {urdf_path}")
        self.robot = self.world.robot(0)
        self.driver_names = [self.robot.driver(i).getName() for i in range(self.robot.numDrivers())]

    def set_driver_positions(self, joint_positions: list[float]) -> None:
        self.robot.setConfig(self.robot.configFromDrivers(joint_positions))

    def get_driver_positions(self) -> list[float]:
        return [self.robot.driver(i).getValue() for i in range(self.robot.numDrivers())]

    def fingertip_positions_relative_to_palm(self, joint_positions: list[float]) -> dict[str, tuple[float, float, float]]:
        self.set_driver_positions(joint_positions)
        palm_rotation, palm_translation = self.robot.link(self.config.palm_link_name).getTransform()
        inverse_rotation = so3.inv(palm_rotation)
        relative_positions: dict[str, tuple[float, float, float]] = {}

        for tip_name, link_name in self.config.tip_link_names.items():
            _, tip_translation = self.robot.link(link_name).getTransform()
            relative_positions[tip_name] = tuple(
                so3.apply(inverse_rotation, vectorops.sub(tip_translation, palm_translation))
            )

        return relative_positions

    def inverse_kinematic(
        self,
        fingertip_targets: dict[str, tuple[float, float, float]],
        seed: list[float] | None = None,
    ) -> list[float]:
        if seed is not None:
            self.set_driver_positions(seed)

        palm_rotation, palm_translation = self.robot.link(self.config.palm_link_name).getTransform()
        objectives = []
        for tip_name, target in fingertip_targets.items():
            link_name = self.config.tip_link_names[tip_name]
            world_target = vectorops.add(palm_translation, so3.apply(palm_rotation, target))
            objectives.append(ik.objective(self.robot.link(link_name), local=[0, 0, 0], world=world_target))

        ik.solve(objectives, iters=self.config.niter, tol=1e-3, activeDofs=self.driver_names)
        return self.get_driver_positions()


class XHand:
    def __init__(self, config: XHandConfig | None = None):
        self.config = config if config is not None else XHandConfig()
        self._device = None
        self._kinematics = None
        self._last_joint_positions = list(self.config.open_positions)
        self._is_connected = False

        if self.config.enable_tip_ik:
            try:
                self._kinematics = XHandKinematics(self.config)
            except Exception:
                if self.config.require_ik:
                    raise
                LOGGER.warning("xHand fingertip IK is disabled because the kinematics model could not be initialized.", exc_info=True)

    def connect(self):
        if self._is_connected:
            return True
        if xhand_control is None:
            raise ImportError(
                "The xhand_controller package is required to use the xHand gripper."
            ) from _xhand_import_error
        if self.config.enable_tip_ik and self.config.require_ik and self._kinematics is None:
            raise RuntimeError("xHand fingertip IK is required but could not be initialized. Check klampt and urdf_path.")

        self._device = xhand_control.XHandControl()
        if self.config.protocol.upper() == "RS485":
            response = self._device.open_serial(self.config.serial_port, int(self.config.baud_rate))
        elif self.config.protocol.upper() == "ETHERCAT":
            ports = self._device.enumerate_devices("EtherCAT")
            if not ports:
                raise RuntimeError("No EtherCAT xHand devices were found.")
            response = self._device.open_ethercat(ports[0])
        else:
            raise ValueError(f"Unsupported xHand protocol: {self.config.protocol}")

        if response.error_code != 0:
            raise RuntimeError(f"Failed to open xHand device: {response.error_message}")

        if self.config.auto_detect_hand_id and hasattr(self._device, "list_hands_id"):
            try:
                hand_ids = list(self._device.list_hands_id())
            except Exception:
                hand_ids = []
            if hand_ids:
                self.config.hand_id = int(hand_ids[0])

        if self.config.startup_delay_s > 0:
            time.sleep(self.config.startup_delay_s)

        self._is_connected = True
        if self.config.connect_reset:
            self.reset()
        return True

    def disconnect(self):
        self.close()

    def close(self):
        if self._device is not None:
            self._device.close_device()
        self._device = None
        self._is_connected = False

    def reset(self):
        self._send_joint_positions(self.config.open_positions)
        time.sleep(0.5)

    def apply_commands(self, action: dict[str, Any] | None = None, gripper_state: float | None = None, **kwargs):
        del kwargs
        fingertip_targets = self._extract_fingertip_targets(action)
        if fingertip_targets and self._kinematics is not None:
            joint_seed = self._read_joint_positions(force_update=self.config.poll_force_update)
            target_joint_positions = self._kinematics.inverse_kinematic(fingertip_targets, seed=joint_seed)
        else:
            if gripper_state is None and action is not None:
                gripper_state = float(action.get("gripper", 1.0))
            if gripper_state is None:
                gripper_state = 1.0
            target_joint_positions = self._interpolate_grasp(float(gripper_state))

        self._send_joint_positions(target_joint_positions)

    @property
    def action_features(self) -> dict:
        features = {"action.gripper": float}
        if self.config.enable_tip_ik:
            features.update({f"action.{key}": value for key, value in _tip_feature_keys().items()})
        return features

    @property
    def features(self) -> dict:
        features = {"gripper": float}
        for joint_index in range(len(self.config.open_positions)):
            features[f"xhand.joint_{joint_index}"] = float
        if self.config.enable_tip_ik:
            features.update(_tip_feature_keys())
        return features

    def get_sensors(self):
        joint_positions = self._read_joint_positions(force_update=self.config.poll_force_update)
        observation = {"gripper": self._compute_gripper_scalar(joint_positions)}
        for joint_index, value in enumerate(joint_positions):
            observation[f"xhand.joint_{joint_index}"] = float(value)

        if self.config.enable_tip_ik:
            fingertip_positions = self._relative_tip_positions(joint_positions)
            for tip_name in TIP_NAMES:
                tip_position = fingertip_positions.get(tip_name, (math.nan, math.nan, math.nan))
                observation[f"fingertip.{tip_name}.x"] = float(tip_position[0])
                observation[f"fingertip.{tip_name}.y"] = float(tip_position[1])
                observation[f"fingertip.{tip_name}.z"] = float(tip_position[2])

        return observation

    def _extract_fingertip_targets(self, action: dict[str, Any] | None) -> dict[str, tuple[float, float, float]]:
        if action is None or not self.config.enable_tip_ik:
            return {}

        fingertip_targets: dict[str, tuple[float, float, float]] = {}
        for tip_name in TIP_NAMES:
            keys = [f"fingertip.{tip_name}.{axis}" for axis in ("x", "y", "z")]
            if not all(key in action for key in keys):
                continue
            fingertip_targets[tip_name] = tuple(float(action[key]) for key in keys)

        return fingertip_targets

    def _interpolate_grasp(self, gripper_state: float) -> list[float]:
        state = float(np.clip(gripper_state, 0.0, 1.0))
        open_positions = np.asarray(self.config.open_positions, dtype=float)
        closed_positions = np.asarray(self.config.closed_positions, dtype=float)
        return (closed_positions + (open_positions - closed_positions) * state).tolist()

    def _send_joint_positions(self, joint_positions: list[float]) -> None:
        if self._device is None:
            raise RuntimeError("xHand is not connected.")
        response = None
        for attempt in range(self.config.command_retries):
            command = xhand_control.HandCommand_t()
            for finger_index, target_position in enumerate(joint_positions[:12]):
                command.finger_command[finger_index].id = finger_index
                command.finger_command[finger_index].kp = int(self.config.kp)
                command.finger_command[finger_index].ki = int(self.config.ki)
                command.finger_command[finger_index].kd = int(self.config.kd)
                command.finger_command[finger_index].position = float(target_position)
                command.finger_command[finger_index].tor_max = int(self.config.tor_max)
                command.finger_command[finger_index].mode = int(self.config.control_mode)

            response = self._device.send_command(self.config.hand_id, command)
            if response.error_code == 0:
                self._last_joint_positions = list(joint_positions)
                return
            LOGGER.warning(
                "xHand send_command failed on attempt %s/%s: %s",
                attempt + 1,
                self.config.command_retries,
                response.error_message,
            )
            if attempt + 1 < self.config.command_retries:
                time.sleep(self.config.retry_delay_s)

        raise RuntimeError(f"Failed to send xHand command: {response.error_message}")

    def _read_joint_positions(self, force_update: bool) -> list[float]:
        if self._device is None:
            return list(self._last_joint_positions)
        response, state = self._device.read_state(self.config.hand_id, force_update)
        if response.error_code != 0:
            LOGGER.warning("Failed to read xHand state: %s", response.error_message)
            return list(self._last_joint_positions)

        joint_positions = [float(finger_state.position) for finger_state in state.finger_state]
        self._last_joint_positions = joint_positions
        return joint_positions

    def _relative_tip_positions(self, joint_positions: list[float]) -> dict[str, tuple[float, float, float]]:
        if self._kinematics is None:
            return {}
        try:
            return self._kinematics.fingertip_positions_relative_to_palm(joint_positions)
        except Exception:
            LOGGER.warning("Failed to compute xHand fingertip forward kinematics.", exc_info=True)
            return {}

    def _compute_gripper_scalar(self, joint_positions: list[float]) -> float:
        open_positions = np.asarray(self.config.open_positions, dtype=float)
        closed_positions = np.asarray(self.config.closed_positions, dtype=float)
        joints = np.asarray(joint_positions[: len(open_positions)], dtype=float)
        closure = (joints - closed_positions) / (open_positions - closed_positions + 1e-6)
        closure = np.clip(closure, 0.0, 1.0)
        return float(np.mean(closure))
