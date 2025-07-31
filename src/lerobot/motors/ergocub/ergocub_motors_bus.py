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

import numpy as np
import yarp
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .arm_controller import ErgoCubArmController
from .neck_controller import ErgoCubNeckController

logger = logging.getLogger(__name__)


class ErgoCubMotorsBus:
    """
    YARP-based motors bus for ErgoCub that manages arm and neck controllers.
    """
    
    def __init__(
        self,
        remote_prefix: str,
        local_prefix: str,
        use_left_arm: bool = True,
        use_right_arm: bool = True,
        use_neck: bool = True,
    ):
        """
        Initialize ErgoCub YARP motors bus.
        
        Args:
            remote_prefix: Remote YARP prefix (e.g., "/ergocubSim")
            local_prefix: Local YARP prefix (e.g., "/lerobot/session_id")
            use_left_arm: Whether to enable left arm
            use_right_arm: Whether to enable right arm
            use_neck: Whether to enable neck
        """
        self.remote_prefix = remote_prefix
        self.local_prefix = local_prefix
        
        # Initialize controllers
        self.controllers = {}
        
        if use_left_arm:
            self.controllers["left_arm"] = ErgoCubArmController("left", remote_prefix, local_prefix)
            
        if use_right_arm:
            self.controllers["right_arm"] = ErgoCubArmController("right", remote_prefix, local_prefix)
            
        if use_neck:
            self.controllers["neck"] = ErgoCubNeckController(remote_prefix, local_prefix)
        
        # Initialize YARP network
        yarp.Network.init()
    
    @property
    def is_connected(self) -> bool:
        """Check if all controllers are connected."""
        return all(controller.is_connected for controller in self.controllers.values())
    
    def connect(self) -> None:
        """Connect all controllers."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError("ErgoCubMotorsBus already connected")
        
        for name, controller in self.controllers.items():
            logger.info(f"Connecting {name} controller...")
            controller.connect()
        
        logger.info("ErgoCubMotorsBus connected")
    
    def disconnect(self) -> None:
        """Disconnect all controllers."""
        if not self.is_connected:
            raise DeviceNotConnectedError("ErgoCubMotorsBus not connected")
        
        for name, controller in self.controllers.items():
            logger.info(f"Disconnecting {name} controller...")
            controller.disconnect()
        
        yarp.Network.fini()
        logger.info("ErgoCubMotorsBus disconnected")
    
    def read_state(self) -> dict[str, float]:
        """Read current state from all controllers."""
        if not self.is_connected:
            raise DeviceNotConnectedError("ErgoCubMotorsBus not connected")
        
        state = {}
        for controller in self.controllers.values():
            controller_state = controller.read_current_state()
            state.update(controller_state)
        
        return state
    
    def send_commands(self, commands: dict[str, float]) -> None:
        """Send commands to controllers."""
        if not self.is_connected:
            raise DeviceNotConnectedError("ErgoCubMotorsBus not connected")
        
        # Group commands by controller
        controller_commands = {
            "left_arm": {"pose": np.zeros(7), "fingers": np.zeros(6)},
            "right_arm": {"pose": np.zeros(7), "fingers": np.zeros(6)},
            "neck": {"orientation": np.zeros(4)},
        }
        
        # Parse commands
        for key, value in commands.items():
            if key.startswith("left_arm."):
                coord = key.split(".")[1]
                if coord in ["x", "y", "z", "qx", "qy", "qz", "qw"]:
                    idx = ["x", "y", "z", "qx", "qy", "qz", "qw"].index(coord)
                    controller_commands["left_arm"]["pose"][idx] = value
            elif key.startswith("right_arm."):
                coord = key.split(".")[1]
                if coord in ["x", "y", "z", "qx", "qy", "qz", "qw"]:
                    idx = ["x", "y", "z", "qx", "qy", "qz", "qw"].index(coord)
                    controller_commands["right_arm"]["pose"][idx] = value
            elif key.startswith("left_fingers."):
                finger_joint = key.split(".")[1]
                if finger_joint in ["thumb_add", "thumb_oc", "index_add", "index_oc", "middle_oc", "ring_pinky_oc"]:
                    idx = ["thumb_add", "thumb_oc", "index_add", "index_oc", "middle_oc", "ring_pinky_oc"].index(finger_joint)
                    controller_commands["left_arm"]["fingers"][idx] = value
            elif key.startswith("right_fingers."):
                finger_joint = key.split(".")[1]
                if finger_joint in ["thumb_add", "thumb_oc", "index_add", "index_oc", "middle_oc", "ring_pinky_oc"]:
                    idx = ["thumb_add", "thumb_oc", "index_add", "index_oc", "middle_oc", "ring_pinky_oc"].index(finger_joint)
                    controller_commands["right_arm"]["fingers"][idx] = value
            elif key.startswith("neck."):
                coord = key.split(".")[1]
                if coord in ["qx", "qy", "qz", "qw"]:
                    idx = ["qx", "qy", "qz", "qw"].index(coord)
                    controller_commands["neck"]["orientation"][idx] = value
        
        # Send commands to controllers
        if "left_arm" in self.controllers:
            self.controllers["left_arm"].send_command(
                controller_commands["left_arm"]["pose"],
                controller_commands["left_arm"]["fingers"]
            )
        
        if "right_arm" in self.controllers:
            self.controllers["right_arm"].send_command(
                controller_commands["right_arm"]["pose"],
                controller_commands["right_arm"]["fingers"]
            )
        
        if "neck" in self.controllers:
            self.controllers["neck"].send_command(
                controller_commands["neck"]["orientation"]
            )
