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
        
        # Initialize kinematics solver
        urdf_file = yarp.ResourceFinder().findFileByName("model.urdf")
        joint_names = ["neck_pitch", "neck_roll", "neck_yaw"]
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
        
        self._is_connected = True
        logger.info("ErgoCubNeckController connected")
    
    def disconnect(self) -> None:
        """Disconnect from YARP ports."""
        if not self.is_connected:
            raise DeviceNotConnectedError("ErgoCubNeckController not connected")
        
        self.neck_cmd_port.close()
        self.encoders_port.close()
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
        
        # Read encoder data with busy wait and warnings
        bottle = self.encoders_port.read(False)
        last_warning_time = 0
        
        while bottle is None:
            current_time = time.time()
            if current_time - last_warning_time > 1.0:  # Warning every second
                logger.warning("No encoder data received for neck")
                last_warning_time = current_time
            time.sleep(0.01)  # Small sleep to avoid excessive CPU usage
            bottle = self.encoders_port.read(False)
        
        # Extract joint values and compute orientation
        joint_values = np.array([bottle.get(i) for i in range(bottle.size())])
        T = self.kinematics_solver.forward_kinematics(joint_values[:3].tolist())
        quaternion = R.from_matrix(T[:3, :3]).as_quat()  # [x, y, z, w]
        
        return dict(zip(["neck.qx", "neck.qy", "neck.qz", "neck.qw"], quaternion))
    
    def send_command(self, orientation: np.ndarray) -> None:
        """
        Send orientation command to the neck.
        
        Args:
            orientation: Array [qx, qy, qz, qw] for neck orientation
        """
        if not self.is_connected:
            raise DeviceNotConnectedError("ErgoCubNeckController not connected")
        
        # Convert quaternion to rotation matrix (MetaControllServer expects 3x3 matrix)
        q = orientation / np.linalg.norm(orientation)  # Normalize quaternion
        rot_matrix = R.from_quat(q).as_matrix()
        
        # Send RPC command to head controller
        neck_cmd = yarp.Bottle()
        neck_cmd.addString("setOrientation")
        
        # Add rotation matrix as nested bottle
        rot_bottle = neck_cmd.addList()
        for i in range(3):
            row_bottle = rot_bottle.addList()
            for j in range(3):
                row_bottle.addFloat64(rot_matrix[i, j])
        
        reply = yarp.Bottle()
        self.neck_cmd_port.write(neck_cmd, reply)
