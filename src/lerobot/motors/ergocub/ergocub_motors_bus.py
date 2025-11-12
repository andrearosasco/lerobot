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
import time

import numpy as np
import yarp
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .head_controller import ErgoCubHeadController
from .finger_controller import ErgoCubFingerController
from .bimanual_controller import ErgoCubBimanualController

logger = logging.getLogger(__name__)


class ErgoCubMotorsBus:
    """
    YARP-based motors bus for ErgoCub that manages bimanual and head controllers.
    """
    
    def __init__(
        self,
        remote_prefix: str,
        local_prefix: str,
        control_boards: list[str],
    ):
        """
        Initialize ErgoCub YARP motors bus.
        
        Args:
            remote_prefix: Remote YARP prefix (e.g., "/ergocubSim")
            local_prefix: Local YARP prefix (e.g., "/lerobot/session_id")
            control_boards: List of control boards to manage (e.g., ["left_hand", "right_hand", "head", "fingers"])
        """
        self.remote_prefix = remote_prefix
        self.local_prefix = local_prefix
        
        # Initialize controllers
        self.controllers = {}
        
        self.controllers["bimanual"] = ErgoCubBimanualController(
            remote_prefix, local_prefix, 'left_hand' in control_boards, 'right_hand' in control_boards
        )
            
        if 'head' in control_boards:
            self.controllers["head"] = ErgoCubHeadController(remote_prefix, local_prefix)
        
        # Optionally add finger controller
        if 'fingers' in control_boards:
            self.controllers["fingers"] = ErgoCubFingerController(local_prefix)

        reset_local = f"{self.local_prefix}/reset:o"
        reset_remote = "/spawn_trigger"
        self._reset_port = yarp.BufferedPortBottle()
        self._reset_port.open(reset_local)

        while not yarp.Network.connect(reset_local, reset_remote):
            logger.warning(f"Failed to connect {reset_local} -> {reset_remote}, retrying...")
            time.sleep(1)
        logger.info(f"Connected {reset_local} -> {reset_remote}")

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
        
        for controller in self.controllers.values():
            controller.send_commands(commands)

    @property
    def motor_features(self) -> dict[str, type]:
        """Get motor features by aggregating from all controllers."""
        features = {}
        for controller in self.controllers.values():
            features.update(controller.motor_features)
        return features

    # ---------------------------------------------------------------------
    # Reset handling
    # ---------------------------------------------------------------------
    def reset(self) -> None:
        bottle = self._reset_port.prepare()
        bottle.clear()
        bottle.addInt32(1)
        self._reset_port.write()
        self.controllers['bimanual'].reset()
        self.controllers['head'].reset()
        self.controllers['fingers'].reset()
        time.sleep(5)  # Allow some time for reset to take effect
