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
import uuid
from functools import cached_property
from typing import Any

import numpy as np
import yarp
from lerobot.cameras import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors.yarp import YarpEncodersBus
from lerobot.robots.robot import Robot

from .configuration_ergocub import ErgoCubConfig

logger = logging.getLogger(__name__)


class ErgoCub(Robot):
    config_class = ErgoCubConfig
    name = "ergocub"
    
    def __init__(self, config: ErgoCubConfig):
        super().__init__(config)
        self.config = config
        self.session_id = uuid.uuid4()
        self._is_connected = False

        yarp.Network.init()

        # Use custom camera creation for YARP cameras with prefixes
        self.cameras = {}
        for cam_name, cam_config in config.cameras.items():
            if cam_config.type == "yarp":
                from lerobot.cameras.yarp import YarpCamera
                self.cameras[cam_name] = YarpCamera(
                    cam_config, 
                    config.remote_prefix, 
                    f"{config.local_prefix}/{self.session_id}"
                )
            else:
                # Use standard LeRobot camera factory for non-YARP cameras
                other_cameras = make_cameras_from_configs({cam_name: cam_config})
                self.cameras.update(other_cameras)

        # Encoders bus (non-camera data)
        self.encoders_bus = YarpEncodersBus(
            remote_prefix=config.remote_prefix,
            local_prefix=f"{config.local_prefix}/{self.session_id}",
            control_boards=config.encoders_control_boards,
            stream_name="encoders",
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
        
        # Connect encoders bus
        self.encoders_bus.connect()
        self._is_connected = True
        
        if not self.is_calibrated and calibrate:
            logger.info("ErgoCub doesn't require calibration - skipping.")
            
        self.configure()
        logger.info(f"{self} connected.")

    def disconnect(self):
        """Disconnect from the robot and perform any necessary cleanup."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
            
        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()
        
        # Disconnect encoders
        self.encoders_bus.disconnect()
        yarp.Network.fini()
        self._is_connected = False
        
        logger.info(f"{self} disconnected.")

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
        
        # Read encoder data and format like SO101 motor data
        encoder_data = self.encoders_bus.sync_read("Present_Position")
        
        # Define joint names mapping (same as in _encoders_ft)
        board_joints_map = {
            "head": ["neck_pitch", "neck_roll", "neck_yaw", "eyes_tilt"],
            "left_arm": [
                "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow",
                "l_wrist_prosup", "l_wrist_pitch", "l_wrist_yaw", "l_thumb_oppose",
                "l_thumb_proximal", "l_thumb_distal", "l_index_proximal", "l_index_distal",
                "l_middle_proximal"
            ],
            "right_arm": [
                "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow",
                "r_wrist_prosup", "r_wrist_pitch", "r_wrist_yaw", "r_thumb_oppose",
                "r_thumb_proximal", "r_thumb_distal", "r_index_proximal", "r_index_distal",
                "r_middle_proximal"
            ],
            "torso": ["torso_pitch", "torso_roll", "torso_yaw"],
            "left_leg": ["l_hip_pitch", "l_hip_roll", "l_hip_yaw", "l_knee", "l_ankle_pitch", "l_ankle_roll"],
            "right_leg": ["r_hip_pitch", "r_hip_roll", "r_hip_yaw", "r_knee", "r_ankle_pitch", "r_ankle_roll"],
        }
        
        for board_name, joint_positions in encoder_data.items():
            if board_name in board_joints_map:
                joint_names = board_joints_map[board_name]
                for i, joint_value in enumerate(joint_positions):
                    if i < len(joint_names):  # Safety check
                        obs[f"{joint_names[i]}.pos"] = joint_value
        
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Send an action command to the robot.

        Args:
            action (dict[str, Any]): Dictionary representing the desired pose action.
                Expected format matches action_features:
                - neck.{x,y,z}: neck position
                - neck.{qx,qy,qz,qw}: neck orientation (quaternion)
                - left_arm.{x,y,z}: left arm position
                - left_arm.{qx,qy,qz,qw}: left arm orientation (quaternion)
                - right_arm.{x,y,z}: right arm position  
                - right_arm.{qx,qy,qz,qw}: right arm orientation (quaternion)
                - fingers.{finger_name}.{x,y,z}: finger positions

        Returns:
            dict[str, Any]: The action actually sent to the teleoperator.
            
        Raises:
            DeviceNotConnectedError: if robot is not connected.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
            
        # The ErgoCub robot receives pose commands from the teleoperator
        # This method validates and forwards the action to the teleoperator system
        # The actual joint control is handled by the teleoperator
        
        # Validate action format matches expected action_features
        expected_keys = set(self.action_features.keys())
        received_keys = set(action.keys())
        
        # Log any missing or extra keys for debugging
        missing_keys = expected_keys - received_keys
        extra_keys = received_keys - expected_keys
        
        if missing_keys:
            logger.warning(f"Missing action keys: {missing_keys}")
        if extra_keys:
            logger.warning(f"Extra action keys: {extra_keys}")
        
        # Convert to dict if it's an array for compatibility
        if isinstance(action, np.ndarray):
            # Map array elements to expected action structure
            action_dict = {}
            idx = 0
            
            # Map array to structured action format
            for key in sorted(self.action_features.keys()):
                if idx < len(action):
                    action_dict[key] = float(action[idx])
                    idx += 1
                else:
                    action_dict[key] = 0.0  # Default value for missing elements
                    
            return action_dict
            
        return action

    @property
    def is_connected(self) -> bool:
        """Whether the robot is currently connected."""
        # Check if cameras are connected
        cameras_connected = all(cam.is_connected for cam in self.cameras.values()) if self.cameras else True
        
        # Check if encoders bus is connected
        encoders_connected = self.encoders_bus.is_connected
        
        return self._is_connected and cameras_connected and encoders_connected

    @property
    def is_calibrated(self) -> bool:
        """ErgoCub doesn't require calibration."""
        return True

    def calibrate(self) -> None:
        """ErgoCub doesn't require calibration - no-op."""
        pass

    def configure(self) -> None:
        """Apply any one-time configuration - no-op for ErgoCub."""
        pass

    @property
    def _encoders_ft(self) -> dict[str, type]:
        """Helper property to get encoder features in SO101 format."""
        # Define joints per control board (similar to motors in SO101)
        board_joints_map = {
            "head": ["neck_pitch", "neck_roll", "neck_yaw", "eyes_tilt"],
            "left_arm": [
                "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow",
                "l_wrist_prosup", "l_wrist_pitch", "l_wrist_yaw", "l_thumb_oppose",
                "l_thumb_proximal", "l_thumb_distal", "l_index_proximal", "l_index_distal",
                "l_middle_proximal"
            ],
            "right_arm": [
                "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow",
                "r_wrist_prosup", "r_wrist_pitch", "r_wrist_yaw", "r_thumb_oppose",
                "r_thumb_proximal", "r_thumb_distal", "r_index_proximal", "r_index_distal",
                "r_middle_proximal"
            ],
            "torso": ["torso_pitch", "torso_roll", "torso_yaw"],
            "left_leg": ["l_hip_pitch", "l_hip_roll", "l_hip_yaw", "l_knee", "l_ankle_pitch", "l_ankle_roll"],
            "right_leg": ["r_hip_pitch", "r_hip_roll", "r_hip_yaw", "r_knee", "r_ankle_pitch", "r_ankle_roll"],
        }
        
        encoders_ft = {}
        for board_name in self.config.encoders_control_boards:
            if board_name in board_joints_map:
                for joint_name in board_joints_map[board_name]:
                    encoders_ft[f"{joint_name}.pos"] = float
        return encoders_ft

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
        return {**self._encoders_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """
        A dictionary describing the structure and types of the actions expected by the robot.
        ErgoCub actions are pose commands (position + orientation) from the teleoperator.
        
        Uses Pattern 1 format: {str: type} for compatibility with LeRobot ecosystem.
        Action format with descriptive names:
        - neck: 4 values (quaternion)
        - left_arm/right_arm: 7 values each (3 position + 4 quaternion) 
        - fingers: 10 fingers × 3 values each (x, y, z positions)
        """
        # Define nested structure that mirrors the action hierarchy
        nested_actions = {
            "neck": {
                "qx": float, "qy": float, "qz": float, "qw": float,
            },
            "left_arm": {
                "x": float, "y": float, "z": float,
                "qx": float, "qy": float, "qz": float, "qw": float,
            },
            "right_arm": {
                "x": float, "y": float, "z": float,
                "qx": float, "qy": float, "qz": float, "qw": float,
            },
            "fingers": {
                "left": {
                    "thumb": {"x": float, "y": float, "z": float},
                    "index": {"x": float, "y": float, "z": float},
                    "middle": {"x": float, "y": float, "z": float},
                    "ring": {"x": float, "y": float, "z": float},
                    "little": {"x": float, "y": float, "z": float},
                },
                "right": {
                    "thumb": {"x": float, "y": float, "z": float},
                    "index": {"x": float, "y": float, "z": float},
                    "middle": {"x": float, "y": float, "z": float},
                    "ring": {"x": float, "y": float, "z": float},
                    "little": {"x": float, "y": float, "z": float},
                },
            },
        }
        
        return self._flatten_nested_dict(nested_actions)
    
    def _flatten_nested_dict(self, nested_dict: dict, prefix: str = "") -> dict[str, type]:
        """Flatten a nested dictionary into dot-separated keys."""
        flattened = {}
        for key, value in nested_dict.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flattened.update(self._flatten_nested_dict(value, new_key))
            else:
                flattened[new_key] = value
        return flattened
