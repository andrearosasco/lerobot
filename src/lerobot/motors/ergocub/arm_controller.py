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
from typing import Dict, Optional, TYPE_CHECKING

import numpy as np
import yarp
from scipy.spatial.transform import Rotation as R
from .urdf_utils import resolve_ergocub_urdf
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.model.kinematics import RobotKinematics

if TYPE_CHECKING:
    from lerobot.model.kinematics import RobotKinematics

logger = logging.getLogger(__name__)


class ErgoCubArmController:
    """
    Controller for ergoCub arm that handles pose (position + orientation) commands and readings.
    Follows SO100 motor conventions but operates at pose level.
    """
    
    def __init__(self, arm_name: str, remote_prefix: str, local_prefix: str):
        """
        Initialize arm controller.
        
        Args:
            arm_name: "left" or "right"
            remote_prefix: Remote YARP prefix (e.g., "/ergocubSim")
            local_prefix: Local YARP prefix (e.g., "/lerobot/session_id")
        """
        self.arm_name = arm_name
        self.remote_prefix = remote_prefix
        self.local_prefix = local_prefix
        self._is_connected = False
        
        # YARP ports for sending commands
        self.arm_cmd_port = yarp.RpcClient()
        self.fingers_cmd_port = yarp.Port()
        
        # Encoder reading for pose computation  
        self.encoders_port = yarp.BufferedPortVector()
        self.torso_encoders_port = yarp.BufferedPortVector()

        # Initialize kinematics solver with torso + arm joints using shared resolver
        urdf_file = resolve_ergocub_urdf()
        joint_names = [
            "torso_roll",
            "torso_pitch",
            "torso_yaw",
            f"{self.arm_name[0]}_shoulder_pitch",
            f"{self.arm_name[0]}_shoulder_roll",
            f"{self.arm_name[0]}_shoulder_yaw",
            f"{self.arm_name[0]}_elbow",
            f"{self.arm_name[0]}_wrist_yaw",
            f"{self.arm_name[0]}_wrist_roll",
            f"{self.arm_name[0]}_wrist_pitch",
        ]
        self.kinematics_solver = RobotKinematics(urdf_file, f"{self.arm_name[0]}_hand_palm", joint_names)
        
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    def connect(self) -> None:
        """Connect to YARP arm control ports."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"ErgoCubArmController({self.arm_name}) already connected")
        
        # Open RPC port for arm pose commands
        arm_cmd_local = f"{self.local_prefix}/{self.arm_name}_arm/rpc:o"
        if not self.arm_cmd_port.open(arm_cmd_local):
            raise ConnectionError(f"Failed to open arm RPC port {arm_cmd_local}")
        
        # Connect directly to cartesian controller RPC port
        arm_cmd_remote = f"/mc-ergocub-cartesian-controller/{self.arm_name}_arm/rpc:i"
        while not yarp.Network.connect(arm_cmd_local, arm_cmd_remote):
            logger.warning(f"Failed to connect {arm_cmd_local} -> {arm_cmd_remote}, retrying...")
            time.sleep(1)
        
        # Open encoder reading port for current pose estimation
        encoders_local = f"{self.local_prefix}/{self.arm_name}_arm/encoders:i"
        if not self.encoders_port.open(encoders_local):
            raise ConnectionError(f"Failed to open encoders port {encoders_local}")
        
        # Connect to robot encoder stream
        encoders_remote = f"{self.remote_prefix}/{self.arm_name}_arm/state:o"
        while not yarp.Network.connect(encoders_remote, encoders_local):
            logger.warning(f"Failed to connect {encoders_remote} -> {encoders_local}, retrying...")
            time.sleep(1)
        
        # Open torso encoder reading port
        torso_encoders_local = f"{self.local_prefix}/{self.arm_name}_arm/torso_encoders:i"
        if not self.torso_encoders_port.open(torso_encoders_local):
            raise ConnectionError(f"Failed to open torso encoders port {torso_encoders_local}")
        
        # Connect to torso encoder stream
        torso_encoders_remote = f"{self.remote_prefix}/torso/state:o"
        while not yarp.Network.connect(torso_encoders_remote, torso_encoders_local):
            logger.warning(f"Failed to connect {torso_encoders_remote} -> {torso_encoders_local}, retrying...")
            time.sleep(1)
        
        self._is_connected = True
        logger.info(f"ErgoCubArmController({self.arm_name}) connected")
    
    def disconnect(self) -> None:
        """Disconnect from YARP ports."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"ErgoCubArmController({self.arm_name}) not connected")
        
        self.arm_cmd_port.close()
        self.encoders_port.close()
        self.torso_encoders_port.close()
        self._is_connected = False
        logger.info(f"ErgoCubArmController({self.arm_name}) disconnected")
    
    def read_current_state(self) -> Dict[str, float]:
        """
        Read current arm pose and finger positions.
        
        Returns:
            Dictionary with pose and finger data in SO100-like format
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"ErgoCubArmController({self.arm_name}) not connected")
        
        # Read arm encoder data with attempt-based warnings
        read_attempts = 0
        while (bottle := self.encoders_port.read(False)) is None:
            read_attempts += 1
            if read_attempts % 1000 == 0:  # Warning every 1000 attempts
                logger.warning(f"Still waiting for {self.arm_name} arm encoder data (attempt {read_attempts})")
            time.sleep(0.001)  # 1 millisecond sleep
        
        # Read torso encoder data
        read_attempts = 0
        while (torso_bottle := self.torso_encoders_port.read(False)) is None:
            read_attempts += 1
            if read_attempts % 1000 == 0:  # Warning every 1000 attempts
                logger.warning(f"Still waiting for torso encoder data for {self.arm_name} arm (attempt {read_attempts})")
            time.sleep(0.001)  # 1 millisecond sleep
        
        # Extract torso joint values (first 3 joints: pitch, roll, yaw)
        torso_values = np.array([torso_bottle.get(i) for i in range(min(3, torso_bottle.size()))])
        
        # Extract arm joint values and compute pose
        arm_values = np.array([bottle.get(i) for i in range(bottle.size())])
        
        # Combine torso + arm joints for kinematics (10 joints total: 3 torso + 7 arm)
        full_joint_values = np.concatenate([torso_values, arm_values[:7]])
        T = self.kinematics_solver.forward_kinematics(full_joint_values.tolist())
        position = T[:3, 3]
        quaternion = R.from_matrix(T[:3, :3]).as_quat()  # [x, y, z, w]
        pose = np.concatenate([position, quaternion])
        fingers = arm_values[7:13] * 180 / np.pi
        
        # Return state in SO100-like format
        pose_keys = [f"{self.arm_name}_arm.{k}" for k in ["x", "y", "z", "qx", "qy", "qz", "qw"]]
        finger_keys = [f"{self.arm_name}_fingers.{k}" for k in ["thumb_add", "thumb_oc", "index_add", "index_oc", "middle_oc", "ring_pinky_oc"]]
        
        return dict(zip(pose_keys + finger_keys, np.concatenate([pose, fingers])))
    
    def send_command(self, pose: np.ndarray, fingers: np.ndarray) -> None:
        """
        Send pose and finger commands to the arm.
        
        Args:
            pose: Array [x, y, z, qx, qy, qz, qw] for arm pose
            fingers: Array [thumb_add, thumb_oc, index_add, index_oc, middle_oc, ring_pinky_oc]
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"ErgoCubArmController({self.arm_name}) not connected")
        
        # Send arm pose command to cartesian controller (same format as MetaControllServer)
        arm_cmd = yarp.Bottle()
        arm_cmd.addString("go_to_pose")
        for val in pose:
            arm_cmd.addFloat64(float(val))
        arm_cmd.addFloat64(0.0)  # time duration
        
        reply = yarp.Bottle()
        self.arm_cmd_port.write(arm_cmd, reply)

    def send_commands(self, commands: dict[str, float]) -> None:
        """Send commands from dict format."""
        # Extract arm pose if available
        arm_keys = ["x", "y", "z", "qx", "qy", "qz", "qw"]
        if all(key in commands for key in arm_keys):
            pose = np.array([commands[key] for key in arm_keys])
        else:
            pose = None
        
        # Extract finger commands if available  
        finger_keys = ["thumb_add", "thumb_oc", "index_add", "index_oc", "middle_oc", "ring_pinky_oc"]
        finger_cmds = {k.split(".", 1)[1]: v for k, v in commands.items() if k.startswith(f"{self.arm_name}_fingers.")}
        if finger_cmds and all(key in finger_cmds for key in finger_keys):
            fingers = np.array([finger_cmds[key] for key in finger_keys])
        else:
            fingers = None
        
        # Send commands if we have them
        if pose is not None:
            self.send_command(pose, fingers if fingers is not None else np.zeros(6))

    @property
    def motor_features(self) -> dict[str, type]:
        """Get motor features for this arm controller."""
        features = {}
        
        # Arm pose (7 DOF)
        for coord in ["x", "y", "z", "qx", "qy", "qz", "qw"]:
            features[f"{self.arm_name}_arm.{coord}"] = float
        
        # Fingers (6 DOF)
        for joint in ["thumb_add", "thumb_oc", "index_add", "index_oc", "middle_oc", "ring_pinky_oc"]:
            features[f"{self.arm_name}_fingers.{joint}"] = float
        
        return features
