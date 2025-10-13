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

import os
import logging
import uuid
from functools import cached_property
from typing import Any

import numpy as np
import yarp
from lerobot.cameras import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors.ergocub import ErgoCubMotorsBus
from lerobot.robots.robot import Robot

from .configuration_ergocub import ErgoCubConfig
from .safety_utils import HandSafetyChecker

logger = logging.getLogger(__name__)


class ErgoCub(Robot):
    config_class = ErgoCubConfig
    name = "ergocub"
    
    def __init__(self, config: ErgoCubConfig):
        super().__init__(config)
        # Set YARP robot name for resource finding
        os.environ["YARP_ROBOT_NAME"] = "ergoCubSN002"

        self.config = config
        self.session_id = uuid.uuid4()
        self._is_connected = False
        # Absolute vs relative action mode
        self.absolute = bool(getattr(config, "absolute", True))
        # Accumulator cache for relative mode; stores the last absolute command we sent (per-key)
        # Initialized lazily from encoders when first used
        self.acc_state = None

        # Initialize safety checker
        self.safety_checker = HandSafetyChecker(position_tolerance=0.1)
        

        yarp.Network.init()


        prepared_camera_configs = {}
        for cam_name, cam_config in config.cameras.items():
            cam_config.local_prefix = f"{config.local_prefix}/{self.session_id}"
            prepared_camera_configs[cam_name] = cam_config

        self.cameras = make_cameras_from_configs(prepared_camera_configs)

        # Initialize the new ErgoCub motors bus
        self.bus = ErgoCubMotorsBus(
            remote_prefix=config.remote_prefix,
            local_prefix=f"{config.local_prefix}/{self.session_id}",
            control_boards=config.control_boards
            
        )

    def connect(self, calibrate: bool = True):
        """
        Establish communication with the robot.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
            
        # Connect cameras using standard LeRobot interface
        for cam in self.cameras.values():
            cam.connect()
        
        # Connect motor bus (hand and head controllers)
        self.bus.connect()
        self._is_connected = True

        self.acc_state = self.bus.read_state()
        
        if not self.is_calibrated and calibrate:
            logger.info("ErgoCub doesn't require calibration - skipping.")
            
        self.configure()
        logger.info("%s connected.", self)

    def disconnect(self):
        """Disconnect from the robot and perform any necessary cleanup."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
            
        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()
        
        # Disconnect motor bus
        self.bus.disconnect()
        self._is_connected = False
        
        logger.info("%s disconnected.", self)

    def get_observation(self) -> dict[str, Any]:
        """
        Retrieve the current observation from the robot.

        Returns:
            dict[str, Any]: A flat dictionary representing the robot's current sensory state.
            
        Raises:
            DeviceNotConnectedError: if robot is not connected.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
            
        obs = {}
        
        # Read camera data using standard LeRobot interface
        for cam_name, cam in self.cameras.items():
            cam_data = cam.read()
            if "image" in cam_data:
                obs[cam_name] = cam_data["image"]
            if "depth" in cam_data:
                obs[f"{cam_name}_depth"] = cam_data["depth"]
        
        # Read motor data (poses and finger positions) using new motor bus
        motor_data = self.bus.read_state()
        obs.update(motor_data)
        
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send an action command to the robot."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # If using relative mode, convert incoming deltas to absolute targets
        if not self.absolute:
            action = self.to_absolute(action)

        # Basic safety checks (action must be in robot format by now)
        # Determine which hands are active based on configured control boards
        hands_to_check = [side for side in ["left", "right"] if f"{side}_hand" in self.config.control_boards]

        current_state = self.bus.read_state()

        if not self.safety_checker.is_valid_action(action, hands_to_check):
            return current_state

        if not self.safety_checker.check_hand_position_safety(action, current_state, hands_to_check):
            return current_state

        # Send commands via motor bus
        self.bus.send_commands(action)
        return action
    
    def reset(self) -> None:
        """Reset the robot to a default state."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        # Reset motor bus (hands and head)
        self.bus.reset()
        self.acc_state = self.bus.read_state()
        logger.info("%s has been reset.", self)

    # ---------------------------------------------------------------------
    # Relative-to-Absolute conversion
    # ---------------------------------------------------------------------
    def to_absolute(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Convert an input action dictionary interpreted as relative deltas into
        absolute targets using an internal accumulator initialized from encoders.
        """
        # TODO: handle quaternions. For now assuming their value is zero
        abs_action: dict[str, Any] = dict(action)

        for k, v in action.items():
            base = self.acc_state[k]
            abs_action[k] = base + v
            self.acc_state[k] = abs_action[k]

        return abs_action


    @property
    def is_connected(self) -> bool:
        """Whether the robot is currently connected."""
        # Check if cameras are connected
        cameras_connected = all(cam.is_connected for cam in self.cameras.values()) if self.cameras else True
        
        # Check if motor bus is connected
        motors_connected = self.bus.is_connected
        
        return self._is_connected and cameras_connected and motors_connected

    @property
    def is_calibrated(self) -> bool:
        """ErgoCub doesn't require calibration."""
        return True

    def calibrate(self) -> None:
        """ErgoCub doesn't require calibration - no-op."""

    def configure(self) -> None:
        """Apply any one-time configuration - no-op for ErgoCub."""

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Get motor features from the bus."""
        return self.bus.motor_features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Helper property to get camera features in SO101 format."""
        cam_features = {}
        for cam_name, cam_config in self.config.cameras.items():
            cam_features[cam_name] = (cam_config.height, cam_config.width, 3)
            if hasattr(cam_config, 'use_depth') and cam_config.use_depth:
                cam_features[f"{cam_name}_depth"] = (cam_config.height, cam_config.width)
        return cam_features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """
        A dictionary describing the structure and types of the observations produced by the robot.
        Values are either float for single values or tuples for array shapes.
        """
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """
        A dictionary describing the structure and types of the actions expected by the robot.
        ErgoCub actions are pose commands (position + orientation) for hands and head, plus finger positions.
        
        Returns action features in SO100-like format with dot notation.
        """
        return self._motors_ft  # Actions and observations have the same structure
