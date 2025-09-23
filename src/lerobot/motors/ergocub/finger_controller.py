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
from lerobot.robots.ergocub.manipulator import Manipulator

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
        # Per-hand kinematics solvers for fingertip-to-joint IK
        self.finger_kinematics = {
            "left": Manipulator("src/lerobot/robots/ergocub/ergocub_hand_left/model.urdf"),
            "right": Manipulator("src/lerobot/robots/ergocub/ergocub_hand_right/model.urdf"),
        }
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    def connect(self) -> None:
        """Connect to YARP finger control port."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError("ErgoCubFingerController already connected")
        
        # Open port for finger commands
        finger_cmd_local = f"{self.local_prefix}/finger_commands:o"
        if not self.finger_cmd_port.open(finger_cmd_local):
            raise ConnectionError(f"Failed to open finger commands port {finger_cmd_local}")
        
        # Connect to finger controller
        finger_cmd_remote = "/ergocub_finger_controller/finger_commands:i"
        while not yarp.Network.connect(finger_cmd_local, finger_cmd_remote):
            logger.warning(f"Failed to connect {finger_cmd_local} -> {finger_cmd_remote}, retrying...")
            time.sleep(1)
        
        self._is_connected = True
        logger.info("ErgoCubFingerController connected")
    
    def disconnect(self) -> None:
        """Disconnect from YARP ports."""
        if not self.is_connected:
            raise DeviceNotConnectedError("ErgoCubFingerController not connected")
        
        self.finger_cmd_port.close()
        
        self._is_connected = False
        logger.info("ErgoCubFingerController disconnected")
    
    def read_current_state(self) -> Dict[str, float]:
        """Read current state for finger joints."""
        if not self.is_connected:
            raise DeviceNotConnectedError("ErgoCubFingerController not connected")
        
        return {}
    
    def send_commands(self, commands: dict[str, float]) -> None:
        """Send finger commands to controller."""
        if not self.is_connected:
            raise DeviceNotConnectedError("ErgoCubFingerController not connected")
        
        # Convert fingertip inputs to joint angles if present
        commands = self._compute_finger_joints(dict(commands))

        # Extract finger joint commands for both hands
        left_finger_cmds = {k.split(".", 1)[1]: v for k, v in commands.items() if k.startswith("left_fingers.")}
        right_finger_cmds = {k.split(".", 1)[1]: v for k, v in commands.items() if k.startswith("right_fingers.")}
        
        # Only send if we have finger commands
        if left_finger_cmds or right_finger_cmds:
            # Prepare bottle with 12 floats (6 left + 6 right)
            finger_bottle = yarp.Bottle()
            finger_bottle.clear()
            
            # Add left hand joints (first 6 floats)
            for joint in self.joint_names:
                value = left_finger_cmds.get(joint, 0.0)
                finger_bottle.addFloat64(value)
            
            # Add right hand joints (next 6 floats)
            for joint in self.joint_names:
                value = right_finger_cmds.get(joint, 0.0)
                finger_bottle.addFloat64(value)
            
            # Send the bottle
            self.finger_cmd_port.write(finger_bottle)
            logger.debug(f"Sent finger commands: {finger_bottle.toString()}")
    
    @property
    def motor_features(self) -> dict[str, type]:
        """Get motor features for finger controller."""
        features = {}
        
        # Left hand fingers
        for joint in self.joint_names:
            features[f"left_fingers.{joint}"] = float
        
        # Right hand fingers
        for joint in self.joint_names:
            features[f"right_fingers.{joint}"] = float
        
        return features

    # --------------------------- Helpers ---------------------------------
    def _compute_finger_joints(self, action: dict[str, float]) -> dict[str, float]:
        """Compute finger joint angles from fingertip positions via IK.
        If direct joint commands exist for a hand, skip IK for that hand.
        Output joint values are in degrees.
        """
        for side in ["left", "right"]:
            if side not in self.finger_kinematics:
                continue

            has_direct = any(f"{side}_fingers.{jn}" in action for jn in self.joint_names)
            if has_direct:
                continue

            finger_positions = []
            finger_tip_keys = []
            for finger in ["thumb", "index", "middle", "ring", "pinky"]:
                keys = [f"{side}_fingers.{finger}.{coord}" for coord in ["x", "y", "z"]]
                if all(k in action for k in keys):
                    finger_positions.append([action[k] for k in keys])
                    finger_tip_keys.extend(keys)

            if len(finger_positions) == 0:
                continue

            assert len(finger_positions) == 5, "Expect 5 fingertips when using teleop inputs"
            self.finger_kinematics[side].inverse_kinematic(finger_positions)
            joint_angles = self.finger_kinematics[side].get_driver_value()
            for i, joint in enumerate(self.joint_names):
                if i < len(joint_angles):
                    action[f"{side}_fingers.{joint}"] = joint_angles[i] * 180.0 / math.pi

            for k in finger_tip_keys:
                action.pop(k, None)

        return action
