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


class ErgoCubNeckController:
    """
    Controller for ergoCub neck that handles orientation commands and readings.
    Follows SO100 motor conventions but operates at orientation level.
    """
    
    def __init__(self, remote_prefix: str, local_prefix: str):
        """
        Initialize neck controller.
        
        Args:
            remote_prefix: Remote YARP prefix (e.g., "/ergocubSim")
            local_prefix: Local YARP prefix (e.g., "/lerobot/session_id")
        """
        self.remote_prefix = remote_prefix
        self.local_prefix = local_prefix
        self._is_connected = False
        
        # YARP ports
        self.neck_cmd_port = yarp.RpcClient()
        self.encoders_port = yarp.BufferedPortVector()
        self.torso_encoders_port = yarp.BufferedPortVector()

        # Initialize kinematics solver with torso + neck joints using shared resolver
        urdf_file = resolve_ergocub_urdf()
        joint_names = [
            "torso_roll",
            "torso_pitch",
            "torso_yaw",
            "neck_pitch",
            "neck_roll",
            "neck_yaw",
        ]
        self.kinematics_solver = RobotKinematics(urdf_file, "head", joint_names)
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    def connect(self) -> None:
        """Connect to YARP neck control ports."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError("ErgoCubNeckController already connected")
        
        # Open RPC port for neck orientation commands
        neck_cmd_local = f"{self.local_prefix}/neck/rpc:o"
        if not self.neck_cmd_port.open(neck_cmd_local):
            raise ConnectionError(f"Failed to open neck RPC port {neck_cmd_local}")
        
        # Connect directly to head controller RPC port
        neck_cmd_remote = "/mc-ergocub-head-controller/rpc:i"
        while not yarp.Network.connect(neck_cmd_local, neck_cmd_remote):
            logger.warning(f"Failed to connect {neck_cmd_local} -> {neck_cmd_remote}, retrying...")
            time.sleep(1)
        
        # Open encoder reading port for current orientation estimation
        encoders_local = f"{self.local_prefix}/neck/encoders:i"
        if not self.encoders_port.open(encoders_local):
            raise ConnectionError(f"Failed to open neck encoders port {encoders_local}")
        
        # Connect to robot neck encoder stream
        encoders_remote = f"{self.remote_prefix}/head/state:o"
        while not yarp.Network.connect(encoders_remote, encoders_local):
            logger.warning(f"Failed to connect {encoders_remote} -> {encoders_local}, retrying...")
            time.sleep(1)
        
        # Open torso encoder reading port for neck kinematics
        torso_encoders_local = f"{self.local_prefix}/neck/torso_encoders:i"
        if not self.torso_encoders_port.open(torso_encoders_local):
            raise ConnectionError(f"Failed to open torso encoders port {torso_encoders_local}")
        
        # Connect to torso encoder stream
        torso_encoders_remote = f"{self.remote_prefix}/torso/state:o"
        while not yarp.Network.connect(torso_encoders_remote, torso_encoders_local):
            logger.warning(f"Failed to connect {torso_encoders_remote} -> {torso_encoders_local}, retrying...")
            time.sleep(1)
        
        self._is_connected = True
        logger.info("ErgoCubNeckController connected")
    
    def disconnect(self) -> None:
        """Disconnect from YARP ports."""
        if not self.is_connected:
            raise DeviceNotConnectedError("ErgoCubNeckController not connected")
        
        self.neck_cmd_port.close()
        self.encoders_port.close()
        self.torso_encoders_port.close()
        self._is_connected = False
        logger.info("ErgoCubNeckController disconnected")
    
    def read_current_state(self) -> Dict[str, float]:
        """
        Read current neck orientation.
        
        Returns:
            Dictionary with neck orientation in SO100-like format
        """
        if not self.is_connected:
            raise DeviceNotConnectedError("ErgoCubNeckController not connected")
        
        # Read neck encoder data with attempt-based warnings
        read_attempts = 0
        while (bottle := self.encoders_port.read(False)) is None:
            read_attempts += 1
            if read_attempts % 1000 == 0:  # Warning every 1000 attempts
                logger.warning(f"Still waiting for neck encoder data (attempt {read_attempts})")
            time.sleep(0.001)  # 1 millisecond sleep
        
        # Read torso encoder data
        read_attempts = 0
        while (torso_bottle := self.torso_encoders_port.read(False)) is None:
            read_attempts += 1
            if read_attempts % 1000 == 0:  # Warning every 1000 attempts
                logger.warning(f"Still waiting for torso encoder data (attempt {read_attempts})")
            time.sleep(0.001)  # 1 millisecond sleep
        
        # Extract torso joint values (first 3 joints: pitch, roll, yaw)
        torso_values = np.array([torso_bottle.get(i) for i in range(min(3, torso_bottle.size()))])
        
        # Extract neck joint values and compute orientation
        neck_values = np.array([bottle.get(i) for i in range(bottle.size())])
        
        # Combine torso + neck joints for kinematics (6 joints total: 3 torso + 3 neck)
        full_joint_values = np.concatenate([torso_values, neck_values[:3]])
        T = self.kinematics_solver.forward_kinematics(full_joint_values.tolist())
        quaternion = R.from_matrix(T[:3, :3]).as_quat(canonical=True, scalar_first=True)  # [w, x, y, z]

        return dict(zip(["neck.orientation.qw", "neck.orientation.qx", "neck.orientation.qy", "neck.orientation.qz"], quaternion))
    
    def send_command(self, orientation: np.ndarray) -> None:
        """
        Send orientation command to the neck.
        
        Args:
            orientation: Array [qw, qx, qy, qz] for neck orientation
        """
        if not self.is_connected:
            raise DeviceNotConnectedError("ErgoCubNeckController not connected")
        
        # Convert quaternion to rotation matrix (MetaControllServer expects 3x3 matrix)
        rot_matrix = R.from_quat(orientation, scalar_first=True).as_matrix().reshape(-1)
        
        # Send RPC command to head controller
        neck_cmd = yarp.Bottle()
        neck_cmd.addString("setOrientationFlat")
        
        # Add rotation matrix as nested bottle
        for i in range(9):
            neck_cmd.addFloat64(rot_matrix[i])
        
        reply = yarp.Bottle()
        self.neck_cmd_port.write(neck_cmd, reply)

    def send_commands(self, commands: dict[str, float]) -> None:
        """Send commands from dict format."""
        # Extract neck orientation if available
        quat_keys = ["neck.orientation.qw", "neck.orientation.qx", "neck.orientation.qy", "neck.orientation.qz"]
        if all(key in commands for key in quat_keys):
            orientation = np.array([commands[key] for key in quat_keys])
            self.send_command(orientation)

    def reset(self) -> None:
        """Reset the bimanual controller"""
        right_cmd = yarp.Bottle()
        right_cmd.addString("goHome")
        
        reply = yarp.Bottle()
        self.neck_cmd_port.write(right_cmd, reply)

    @property
    def motor_features(self) -> dict[str, type]:
        """Get motor features for neck controller."""
        features = {}
        
        # Neck orientation (4 DOF)
        for coord in ["qw", "qx", "qy", "qz"]:
            features[f"neck.orientation.{coord}"] = float
        
        return features
