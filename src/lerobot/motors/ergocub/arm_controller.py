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
        
        # Encoder reading for pose computation (will use direct kinematics later)
        self.encoders_port = yarp.BufferedPortVector()
        
        # Initialize kinematics solver
        urdf_file = yarp.ResourceFinder().findFileByName("model.urdf")
        joint_names = [f"{self.arm_name[0]}_shoulder_pitch", f"{self.arm_name[0]}_shoulder_roll", f"{self.arm_name[0]}_shoulder_yaw", f"{self.arm_name[0]}_elbow", f"{self.arm_name[0]}_wrist_yaw", f"{self.arm_name[0]}_wrist_roll", f"{self.arm_name[0]}_wrist_pitch"]
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
        
        self._is_connected = True
        logger.info(f"ErgoCubArmController({self.arm_name}) connected")
    
    def disconnect(self) -> None:
        """Disconnect from YARP ports."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"ErgoCubArmController({self.arm_name}) not connected")
        
        self.arm_cmd_port.close()
        self.encoders_port.close()
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
        
        # Read encoder data with busy wait and warnings
        bottle = self.encoders_port.read(False)
        last_warning_time = 0
        
        while bottle is None:
            current_time = time.time()
            if current_time - last_warning_time > 1.0:  # Warning every second
                logger.warning(f"No encoder data received for {self.arm_name} arm")
                last_warning_time = current_time
            time.sleep(0.01)  # Small sleep to avoid excessive CPU usage
            bottle = self.encoders_port.read(False)
        
        # Extract joint values and compute pose
        joint_values = np.array([bottle.get(i) for i in range(bottle.size())])
        T = self.kinematics_solver.forward_kinematics(joint_values[:7].tolist())
        position = T[:3, 3]
        quaternion = R.from_matrix(T[:3, :3]).as_quat()  # [x, y, z, w]
        pose = np.concatenate([position, quaternion])
        fingers = joint_values[7:13]
        
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
