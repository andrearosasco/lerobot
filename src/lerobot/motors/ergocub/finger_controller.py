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
from typing import Dict
import math

import yarp
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

logger = logging.getLogger(__name__)


class ErgoCubFingerController:
    """
    Finger controller for ergoCub that controls both hands through a single port.
    Sends 12 floats (6 for left hand, 6 for right hand) to /ergocub_finger_controller/finger_commands:i
    """
    
    def __init__(self, local_prefix: str):
        """
        Initialize finger controller.
        
        Args:
            local_prefix: Local YARP prefix (e.g., "/lerobot/session_id")
        """
        self.local_prefix = local_prefix
        
        # YARP port for finger commands
        self.finger_cmd_port = yarp.Port()
        
        self._is_connected = False
        
        # Joint names for each hand (same order for left and right)
        self.joint_names = ["thumb_add", "thumb_oc", "index_add", "index_oc", "middle_oc", "ring_pinky_oc"]
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    def connect(self, active_boards: list[str] = None) -> None:
        """Connect to YARP finger control port."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError("ErgoCubFingerController already connected")
        
        # Store active boards for later use
        if active_boards is None:
            active_boards = ["left_fingers", "right_fingers"]
        self.active_boards = active_boards
        
        # Open port for finger commands
        finger_cmd_local = f"{self.local_prefix}/finger_commands:o"
        if not self.finger_cmd_port.open(finger_cmd_local):
            raise ConnectionError(f"Failed to open finger commands port {finger_cmd_local}")
        
        # Always connect to the same finger controller port
        finger_cmd_remote = "/ergocub_finger_controller/finger_commands:i"
        while not yarp.Network.connect(finger_cmd_local, finger_cmd_remote):
            logger.warning(f"Failed to connect {finger_cmd_local} -> {finger_cmd_remote}, retrying...")
            time.sleep(1)
        
        self._is_connected = True
        logger.info(f"ErgoCubFingerController connected for boards: {active_boards}")
    
    def disconnect(self) -> None:
        """Disconnect from YARP ports."""
        if not self.is_connected:
            raise DeviceNotConnectedError("ErgoCubFingerController not connected")
        
        self.finger_cmd_port.close()
        
        self._is_connected = False
        logger.info("ErgoCubFingerController disconnected")
    
    def read_current_state(self, active_boards: list[str] = None) -> Dict[str, float]:
        """Read current state for finger joints."""
        if not self.is_connected:
            raise DeviceNotConnectedError("ErgoCubFingerController not connected")
        
        # Default to both hands if not specified
        if active_boards is None:
            active_boards = ["left_fingers", "right_fingers"]
        
        # For now, return empty state since we don't have encoder feedback for fingers
        # In the future, this could be implemented to read actual finger positions
        state = {}
        
        # Return zero values for active finger joints (placeholder implementation)
        if "left_fingers" in active_boards:
            for joint in self.joint_names:
                state[f"left_fingers.{joint}"] = 0.0
        
        if "right_fingers" in active_boards:
            for joint in self.joint_names:
                state[f"right_fingers.{joint}"] = 0.0
        
        return state
    
    def send_commands(self, commands: dict[str, float], active_boards: list[str] = None) -> None:
        """Send finger commands to controller."""
        if not self.is_connected:
            raise DeviceNotConnectedError("ErgoCubFingerController not connected")
        
        # Default to both hands if not specified
        if active_boards is None:
            active_boards = ["left_fingers", "right_fingers"]
        
        # Extract finger commands for both hands
        left_finger_cmds = {k.split(".", 1)[1]: v for k, v in commands.items() if k.startswith("left_fingers.")}
        right_finger_cmds = {k.split(".", 1)[1]: v for k, v in commands.items() if k.startswith("right_fingers.")}
        
        # Use stored active boards if not provided
        if active_boards is None and hasattr(self, 'active_boards'):
            active_boards = self.active_boards
        elif active_boards is None:
            active_boards = ["left_fingers", "right_fingers"]
        
        # Only send if we have finger commands for active boards
        should_send = False
        if "left_fingers" in active_boards and left_finger_cmds:
            should_send = True
        if "right_fingers" in active_boards and right_finger_cmds:
            should_send = True
        
        if should_send:
            finger_bottle = yarp.Bottle()
            finger_bottle.clear()
            
            # Determine how many values to send based on active boards
            both_hands_active = "left_fingers" in active_boards and "right_fingers" in active_boards
            left_only = "left_fingers" in active_boards and "right_fingers" not in active_boards
            right_only = "right_fingers" in active_boards and "left_fingers" not in active_boards
            
            if both_hands_active:
                # Both hands active - send 12 values (6 left + 6 right)
                for joint in self.joint_names:
                    value = left_finger_cmds.get(joint, 0.0)
                    finger_bottle.addFloat64(value)
                for joint in self.joint_names:
                    value = right_finger_cmds.get(joint, 0.0)
                    finger_bottle.addFloat64(value)
            elif left_only:
                # Only left hand active - send 6 values for left hand
                for joint in self.joint_names:
                    value = left_finger_cmds.get(joint, 0.0)
                    finger_bottle.addFloat64(value)
            elif right_only:
                # Only right hand active - send 6 values for right hand  
                for joint in self.joint_names:
                    value = right_finger_cmds.get(joint, 0.0)
                    finger_bottle.addFloat64(value)
            
            # Send the bottle
            self.finger_cmd_port.write(finger_bottle)
            logger.debug(f"Sent {finger_bottle.size()} finger commands for boards {active_boards}: {finger_bottle.toString()}")
    
    def reset(self, active_boards: list[str] = None) -> None:
        """Reset finger joints to default position for active boards."""
        if not self.is_connected:
            raise DeviceNotConnectedError("ErgoCubFingerController not connected")
        
        # Default to both hands if not specified
        if active_boards is None:
            active_boards = ["left_fingers", "right_fingers"]
        
        # Default reset values for finger joints: [thumb_add, thumb_oc, index_add, index_oc, middle_oc, ring_pinky_oc]
        reset_values = [40.0, 40.0, 15.0, 17.0, 17.0, 20.0]
        
        # Create reset commands for active finger boards only
        reset_commands = {}
        
        # Add reset commands for left hand fingers if active
        if "left_fingers" in active_boards:
            for i, joint in enumerate(self.joint_names):
                reset_commands[f"left_fingers.{joint}"] = reset_values[i]
        
        # Add reset commands for right hand fingers if active
        if "right_fingers" in active_boards:
            for i, joint in enumerate(self.joint_names):
                reset_commands[f"right_fingers.{joint}"] = reset_values[i]
        
        # Send reset commands
        self.send_commands(reset_commands, active_boards)
        logger.info(f"Reset finger joints to default position for boards: {active_boards}")
    
    def get_motor_features(self, active_boards: list[str] = None) -> dict[str, type]:
        """Get motor features for finger controller based on active boards."""
        if active_boards is None:
            active_boards = ["left_fingers", "right_fingers"]
        
        features = {}
        
        # Left hand fingers - only if active
        if "left_fingers" in active_boards:
            for joint in self.joint_names:
                features[f"left_fingers.{joint}"] = float
        
        # Right hand fingers - only if active
        if "right_fingers" in active_boards:
            for joint in self.joint_names:
                features[f"right_fingers.{joint}"] = float
        
        return features
    
    @property
    def motor_features(self) -> dict[str, type]:
        """Get motor features for finger controller (backward compatibility)."""
        return self.get_motor_features()
