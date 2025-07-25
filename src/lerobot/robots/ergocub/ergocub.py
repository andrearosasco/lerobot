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

import uuid
from typing import Any

import numpy as np
import yarp
from lerobot.robots.robot import Robot
from lerobot.cameras import make_cameras_from_configs

from .configuration_ergocub import ErgoCubConfig
from .yarp_encoders_bus import YarpEncodersBus


class ErgoCub(Robot):
    # Required class attributes for LeRobot compatibility
    config_class = ErgoCubConfig
    name = "ergocub"
    
    def __init__(self, cfg: ErgoCubConfig):
        self.cfg = cfg
        self.session_id = uuid.uuid4()
        self._is_connected = False
        
        # Call parent constructor after setting cfg
        super().__init__(cfg)

        yarp.Network.init()

        # Use custom camera creation for YARP cameras with prefixes
        self.cameras = {}
        for cam_name, cam_config in cfg.cameras.items():
            if cam_config.type == "yarp":
                from lerobot.cameras.yarp import YarpCamera
                self.cameras[cam_name] = YarpCamera(
                    cam_config, 
                    cfg.remote_prefix, 
                    f"{cfg.local_prefix}/{self.session_id}"
                )
            else:
                # Use standard LeRobot camera factory for non-YARP cameras
                from lerobot.cameras import make_cameras_from_configs
                other_cameras = make_cameras_from_configs({cam_name: cam_config})
                self.cameras.update(other_cameras)

        # Encoders bus (non-camera data)
        self.encoders_bus = YarpEncodersBus(
            remote_prefix=cfg.remote_prefix,
            local_prefix=f"{cfg.local_prefix}/{self.session_id}",
            control_boards=cfg.encoders_control_boards,
            stream_name="encoders",
        )

    def connect(self, calibrate: bool = True):
        # Connect cameras using standard LeRobot interface
        for cam in self.cameras.values():
            cam.connect()
        
        # Connect encoders bus
        self.encoders_bus.connect()
        self._is_connected = True

    def disconnect(self):
        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()
        
        # Disconnect encoders
        self.encoders_bus.disconnect()
        yarp.Network.fini()
        self._is_connected = False

    def get_observation(self) -> dict[str, Any]:
        obs = {}
        
        # Read camera data using standard LeRobot interface
        for cam_name, cam in self.cameras.items():
            cam_data = cam.read()
            if "image" in cam_data:
                obs[cam_name] = cam_data["image"]
            if "depth" in cam_data:
                obs[f"{cam_name}_depth"] = cam_data["depth"]
        
        # Read encoder data
        encoder_data = self.encoders_bus.sync_read("Present_Position")
        for board_name, joint_positions in encoder_data.items():
            obs[f"state/joint_positions/{board_name}"] = joint_positions
        
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # This is handled by the teleoperator, so this is a no-op for the robot.
        # Convert to dict if it's an array for compatibility
        if isinstance(action, np.ndarray):
            # For compatibility with existing code, just return as dict
            return {"action": action}
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
    def action_features(self) -> dict:
        """ErgoCub actions come from the teleoperator."""
        # Since this robot doesn't directly handle actions (teleoperator does),
        # we'll define a basic action space that matches what the teleoperator provides
        # Based on the teleoperator's action_features
        features = {
            "action/neck": {"shape": (9,), "dtype": "float32", "space": "joint_positions"},
            "action/left_arm": {"shape": (7,), "dtype": "float32", "space": "joint_positions"},
            "action/right_arm": {"shape": (7,), "dtype": "float32", "space": "joint_positions"},
            "action/fingers": {"shape": (10, 3), "dtype": "float32", "space": "joint_positions"},
        }
        return features

    @property
    def observation_features(self):
        features = {}
        
        # Camera features using standard LeRobot format
        for cam_name, cam_config in self.cfg.cameras.items():
            features[cam_name] = {
                "shape": (cam_config.height, cam_config.width, 3),
                "dtype": "uint8",
                "space": "rgb",
            }
            if hasattr(cam_config, 'use_depth') and cam_config.use_depth:
                features[f"{cam_name}_depth"] = {
                    "shape": (cam_config.height, cam_config.width),
                    "dtype": "uint16", 
                    "space": "depth",
                }

        # Encoder features
        board_joints_lengths = {
            "head": 4,
            "left_arm": 13,
            "right_arm": 13,
            "torso": 3,
            "left_leg": 6,
            "right_leg": 6,
        }
        for board_name in self.cfg.encoders_control_boards:
            features[f"state/joint_positions/{board_name}"] = {
                "shape": (board_joints_lengths[board_name],),
                "dtype": "float32",
                "space": "joint_positions",
            }
        return features
