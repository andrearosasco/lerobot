#!/usr/bin/env python

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

import time
import logging
from typing import Any
import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation as R

# ROS2 imports
import rclpy

from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.cameras.utils import make_cameras_from_configs
from ..robot import Robot
from .config_custom_manipulator import CustomManipulatorConfig
from .utils import make_arm_from_config, make_gripper_from_config
from lerobot.configs.types import FeatureType, PolicyFeature

logger = logging.getLogger(__name__)

class CustomManipulator(Robot):
    config_class = CustomManipulatorConfig
    name = "custom_manipulator"

    def __init__(self, config: CustomManipulatorConfig):
        Robot.__init__(self, config)
        if not rclpy.ok():
            rclpy.init()
        
        self.config = config
        self._is_connected = False

        self.arm_interface = make_arm_from_config(config.arm)
        self.gripper_interface = make_gripper_from_config(config.gripper)
            
        self.cameras = make_cameras_from_configs(config.cameras)
        
        self.last_step = time.perf_counter()

    @property
    def observation_features(self) -> dict:
        features = {}
        features.update(self.arm_interface.features)
        features.update(self.gripper_interface.features)
        
        # Cameras
        for name, cfg in self.config.cameras.items():
            features[f"{name}_rgb"] = (cfg.height, cfg.width, 3)
        
        return features

    @property
    def action_features(self) -> dict:
        return {
            "action.position.x": float,
            "action.position.y": float,
            "action.position.z": float,
            "action.orientation.x": float,
            "action.orientation.y": float,
            "action.orientation.z": float,
            "action.gripper": float,
        }

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            return
            
        self.gripper_interface.connect()
        self.arm_interface.connect()
        
        for cam in self.cameras.values():
            cam.connect()
        
        self._is_connected = True
        logger.info(f"{self} connected.")

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            return
            
        self.arm_interface.close()
        self.gripper_interface.close()
        for cam in self.cameras.values():
            cam.disconnect()
        
        self._is_connected = False
        logger.info(f"{self} disconnected.")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}
        arm_sensors = self.arm_interface.get_sensors()
        gripper_sensors = self.gripper_interface.get_sensors()
        for name, cam in self.cameras.items():
            obs_dict[f"{name}_rgb"] = cam.read()
        
        obs_dict = {**arm_sensors, **gripper_sensors, **obs_dict}
        
        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Apply commands
        self.arm_interface.apply_commands(action=action)
        
        grip = action["gripper"]
        self.gripper_interface.apply_commands(gripper_state=grip)
        
        # Wait for step time (simple rate limiting)
        # In original code: while (time.perf_counter() - self.last_step) < (1/20): pass
        # We can implement similar logic if needed, or rely on the control loop calling this at fixed rate.
        # For now, let's just update last_step
        self.last_step = time.perf_counter()
        
        return action

    def reset(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        self.arm_interface.reset()
        self.gripper_interface.reset()

