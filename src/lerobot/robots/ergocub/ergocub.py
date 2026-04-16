#!/usr/bin/env python

# Copyright 2024 Istituto Italiano di Tecnologia. All rights reserved.
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

import logging
import os
import uuid
from functools import cached_property
from pathlib import Path
from typing import Any

import yarp
from lerobot.cameras import make_cameras_from_configs
from lerobot.motors.ergocub import ErgoCubMotorsBus
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ...motors.ergocub.urdf_utils import resolve_robot_urdf
from .configuration_ergocub import CubRobotConfig, ErgoCubConfig, R1Config
from .profiles import get_cub_robot_profile
from .safety_utils import HandSafetyChecker

logger = logging.getLogger(__name__)


class CubRobot(Robot):
    config_class = CubRobotConfig
    name = "cub_robot"

    def __init__(self, config: CubRobotConfig):
        super().__init__(config)
        self.config = config
        profile_name = config.name if config.name in {"ergocub", "r1"} else config.robot_model
        self.profile = get_cub_robot_profile(profile_name)
        self.name = profile_name
        os.environ["YARP_ROBOT_NAME"] = config.yarp_robot_name or self.profile.yarp_robot_name
        self.urdf_path = self._resolve_urdf_path()
        self.session_id = uuid.uuid4()
        self._is_connected = False
        self.absolute = bool(getattr(config, "absolute", True))
        self.acc_state = None
        self.safety_checker = HandSafetyChecker(position_tolerance=config.position_tolerance)

        yarp.Network.init()

        prepared_camera_configs = {}
        for cam_name, cam_config in config.cameras.items():
            cam_config.local_prefix = f"{config.local_prefix}/{self.session_id}"
            prepared_camera_configs[cam_name] = cam_config

        self.cameras = make_cameras_from_configs(prepared_camera_configs)
        self.bus = ErgoCubMotorsBus(
            remote_prefix=config.remote_prefix,
            local_prefix=f"{config.local_prefix}/{self.session_id}",
            urdf_path=self.urdf_path,
            profile=self.profile,
            control_boards=config.control_boards,
            state_boards=config.state_boards,
            left_hand=config.left_hand,
            right_hand=config.right_hand,
            finger_scale=config.finger_scale,
        )

    def _resolve_urdf_path(self) -> str:
        if self.config.urdf_path:
            return str(Path(self.config.urdf_path).expanduser().resolve())
        return resolve_robot_urdf(
            env_vars=(self.profile.urdf_env_var, "ROBOT_URDF_PATH"),
            fallback_filename=self.profile.urdf_fallback_filename,
        )

    def connect(self, calibrate: bool = True):
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        for cam in self.cameras.values():
            cam.connect()

        self.bus.connect()
        self._is_connected = True
        self.acc_state = self.bus.read_state()

        if not self.is_calibrated and calibrate:
            logger.info("%s doesn't require calibration - skipping.", self.name)

        self.configure()
        logger.info("%s connected.", self)

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        for cam in self.cameras.values():
            cam.disconnect()

        self.bus.disconnect()
        self._is_connected = False
        logger.info("%s disconnected.", self)

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs = {}
        for cam_name, cam in self.cameras.items():
            cam_data = cam.read()
            if "image" in cam_data:
                obs[cam_name] = cam_data["image"]
            if "depth" in cam_data:
                obs[f"{cam_name}_depth"] = cam_data["depth"]

        obs.update(self.bus.read_state())
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not self.absolute:
            action = self.to_absolute(action)

        hands_to_check = [side for side in ["left", "right"] if f"{side}_hand" in self.config.control_boards]
        current_state = self.bus.read_state()

        if not self.safety_checker.is_valid_action(action, hands_to_check):
            return current_state
        if not self.safety_checker.check_hand_position_safety(action, current_state, hands_to_check):
            return current_state

        self.bus.send_commands(action)
        return action

    def reset(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.reset()
        self.acc_state = self.bus.read_state()
        logger.info("%s has been reset.", self)

    def to_absolute(self, action: dict[str, Any]) -> dict[str, Any]:
        abs_action: dict[str, Any] = dict(action)
        for key, value in action.items():
            base = self.acc_state[key]
            abs_action[key] = base + value
            self.acc_state[key] = abs_action[key]
        return abs_action

    @property
    def is_connected(self) -> bool:
        cameras_connected = all(cam.is_connected for cam in self.cameras.values()) if self.cameras else True
        motors_connected = self.bus.is_connected
        return self._is_connected and cameras_connected and motors_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    @property
    def _motors_ft(self) -> dict[str, type]:
        return self.bus.motor_features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        cam_features = {}
        for cam_name, cam_config in self.config.cameras.items():
            cam_features[cam_name] = (cam_config.height, cam_config.width, 3)
            if hasattr(cam_config, "use_depth") and cam_config.use_depth:
                cam_features[f"{cam_name}_depth"] = (cam_config.height, cam_config.width)
        return cam_features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft


class ErgoCub(CubRobot):
    config_class = ErgoCubConfig
    name = "ergocub"


class R1(CubRobot):
    config_class = R1Config
    name = "r1"
