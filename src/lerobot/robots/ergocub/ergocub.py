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
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors.ergocub import ErgoCubMotorsBus
from lerobot.robots.robot import Robot

from .configuration_ergocub import ErgoCubConfig
from .safety_utils import HandSafetyChecker
from .metaquest_transforms import transform_metaquest_to_ergocub
from .manipulator import Manipulator

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

        # Initialize safety checker
        self.safety_checker = HandSafetyChecker(position_tolerance=0.1)
        
        # Initialize finger kinematics using Manipulator class
        self.finger_kinematics = {}
        
        if config.use_left_arm:
            left_hand_urdf = "src/lerobot/robots/ergocub/ergocub_hand_left/model.urdf"
            self.finger_kinematics["left"] = Manipulator(left_hand_urdf)
        
        if config.use_right_arm:
            right_hand_urdf = "src/lerobot/robots/ergocub/ergocub_hand_right/model.urdf"
            self.finger_kinematics["right"] = Manipulator(right_hand_urdf)

        yarp.Network.init()

        # Use custom camera creation for YARP cameras with prefixes
        self.cameras = {}
        for cam_name, cam_config in config.cameras.items():
            if cam_config.type == "yarp":
                from lerobot.cameras.yarp import YarpCamera
                self.cameras[cam_name] = YarpCamera(
                    cam_config, 
                    f"{config.local_prefix}/{self.session_id}"
                )
            else:
                # Use standard LeRobot camera factory for non-YARP cameras
                other_cameras = make_cameras_from_configs({cam_name: cam_config})
                self.cameras.update(other_cameras)

        # Initialize the new ErgoCub motors bus
        self.bus = ErgoCubMotorsBus(
            remote_prefix=config.remote_prefix,
            local_prefix=f"{config.local_prefix}/{self.session_id}",
            use_left_arm=config.use_left_arm,
            use_right_arm=config.use_right_arm,
            use_neck=config.use_neck,
            use_bimanual_controller=getattr(config, 'use_bimanual_controller', False),
            use_fingers=getattr(config, 'use_fingers', True),
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
        
        # Connect motor bus (arm and neck controllers)
        self.bus.connect()
        self._is_connected = True
        
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
        
        
        # Check for any invalid values that might cause safety checker to fail
        invalid_keys = []
        for key, value in action.items():
            if value is None or (hasattr(value, '__iter__') and not isinstance(value, (str, bytes))):
                invalid_keys.append((key, value))
            try:
                float(value)  # Try to convert to float
            except (ValueError, TypeError):
                invalid_keys.append((key, value))
        
        if invalid_keys:
            print(f"🔧 send_action: Found {len(invalid_keys)} invalid action values:")
            for key, value in invalid_keys[:5]:  # Show first 5
                print(f"🔧   {key}: {value} (type: {type(value)})")
        
        # Apply MetaQuest coordinate transforms if enabled
        action = transform_metaquest_to_ergocub(action)
        # Compute finger joint angles from finger tip positions
        action = self.compute_finger_joints(action)
        
        # Basic safety checks (action must be in robot format by now)
        arms_to_check = [side for side in ["left", "right"] 
                        if getattr(self.config, f"use_{side}_arm")]
        
        current_state = self.bus.read_state()
        
        if not self.safety_checker.is_valid_action(action, arms_to_check):
            return current_state
        
        if not self.safety_checker.check_hand_position_safety(action, current_state, arms_to_check):
            return current_state
        
        # Send commands via motor bus
        self.bus.send_commands(action)
        return action

    def compute_finger_joints(self, action: dict[str, Any]) -> dict[str, Any]:
        """Compute finger joint angles from MetaQuest finger tip positions using Manipulator.
        
        If the action already contains direct joint commands (from policy), skip IK computation.
        If the action contains finger tip positions (from teleoperation), compute joint angles via IK.
        """
        for side in ["left", "right"]:
            if side not in self.finger_kinematics:
                continue
            
            # Check if action already contains direct finger joint commands
            joint_names = ["thumb_add", "thumb_oc", "index_add", "index_oc", "middle_oc", "ring_pinky_oc"]
            has_direct_joints = any(f"{side}_fingers.{joint}" in action for joint in joint_names)
            
            if has_direct_joints:
                # Action contains direct joint commands (from policy) - skip IK computation
                continue
                
            # Extract finger tip positions as list of [x,y,z] positions (from teleoperation)
            finger_positions = []
            finger_tip_keys = []  # Keep track of keys to remove later
            
            for finger in ["thumb", "index", "middle", "ring", "pinky"]:
                keys = [f"{side}_fingers.{finger}.{coord}" for coord in ["x", "y", "z"]]
                if all(key in action for key in keys):
                    finger_positions.append([action[key] for key in keys])
                    finger_tip_keys.extend(keys)  # Store keys for removal
            
            if len(finger_positions) == 0:
                # No finger data in action - skip
                continue
                
            assert len(finger_positions) == 5  # All 5 fingers for teleoperation
            # Use Manipulator to solve IK for finger tip positions
            self.finger_kinematics[side].inverse_kinematic(finger_positions)
            # Get the computed joint angles
            joint_angles = self.finger_kinematics[side].get_driver_value()
            for i, joint in enumerate(joint_names):
                if i < len(joint_angles):
                    action[f"{side}_fingers.{joint}"] = joint_angles[i] * 180 / np.pi
            
            # Remove the finger tip position keys from action
            for key in finger_tip_keys:
                action.pop(key, None)
        
        return action

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
        ErgoCub actions are pose commands (position + orientation) for arms and neck, plus finger positions.
        
        Returns action features in SO100-like format with dot notation.
        """
        return self._motors_ft  # Actions and observations have the same structure
