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
from typing import Any, Dict, List
from contextlib import contextmanager

import numpy as np
import yarp
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.motors_bus import MotorsBus

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
        self.arm_cmd_port = yarp.Port()
        self.fingers_cmd_port = yarp.Port()
        
        # Encoder reading for pose computation (will use direct kinematics later)
        self.encoders_port = yarp.BufferedPortVector()
        
        # Current state cache
        self._current_pose = np.zeros(7)  # x, y, z, qx, qy, qz, qw
        self._current_fingers = np.zeros(6)  # thumb_add, thumb_oc, index_add, index_oc, middle_oc, ring_pinky_oc
        
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    def connect(self) -> None:
        """Connect to YARP arm control ports."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"ErgoCubArmController({self.arm_name}) already connected")
        
        # Open command port for arm pose commands
        arm_cmd_local = f"{self.local_prefix}/{self.arm_name}_arm/cmd:o"
        if not self.arm_cmd_port.open(arm_cmd_local):
            raise ConnectionError(f"Failed to open arm command port {arm_cmd_local}")
        
        # Connect to MetaControllServer arm command port
        arm_cmd_remote = f"/metaControllServer/{self.arm_name}_arm/cmd:i"
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
        
        # Read encoder data (non-blocking)
        bottle = self.encoders_port.read(False)
        if bottle is not None:
            # Extract joint values
            joint_values = np.array([bottle.get(i).asFloat64() for i in range(bottle.size())])
            
            # TODO: Implement direct kinematics to compute pose from joint angles
            # For now, use cached values or placeholder
            self._update_pose_from_encoders(joint_values)
        
        # Return state in SO100-like format
        state = {}
        
        # Arm pose (7 DOF: position + quaternion)
        state[f"{self.arm_name}_arm.x"] = float(self._current_pose[0])
        state[f"{self.arm_name}_arm.y"] = float(self._current_pose[1])
        state[f"{self.arm_name}_arm.z"] = float(self._current_pose[2])
        state[f"{self.arm_name}_arm.qx"] = float(self._current_pose[3])
        state[f"{self.arm_name}_arm.qy"] = float(self._current_pose[4])
        state[f"{self.arm_name}_arm.qz"] = float(self._current_pose[5])
        state[f"{self.arm_name}_arm.qw"] = float(self._current_pose[6])
        
        # Finger positions (6 DOF)
        state[f"{self.arm_name}_fingers.thumb_add"] = float(self._current_fingers[0])
        state[f"{self.arm_name}_fingers.thumb_oc"] = float(self._current_fingers[1])
        state[f"{self.arm_name}_fingers.index_add"] = float(self._current_fingers[2])
        state[f"{self.arm_name}_fingers.index_oc"] = float(self._current_fingers[3])
        state[f"{self.arm_name}_fingers.middle_oc"] = float(self._current_fingers[4])
        state[f"{self.arm_name}_fingers.ring_pinky_oc"] = float(self._current_fingers[5])
        
        return state
    
    def send_command(self, pose: np.ndarray, fingers: np.ndarray) -> None:
        """
        Send pose and finger commands to the arm.
        
        Args:
            pose: Array [x, y, z, qx, qy, qz, qw] for arm pose
            fingers: Array [thumb_add, thumb_oc, index_add, index_oc, middle_oc, ring_pinky_oc]
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"ErgoCubArmController({self.arm_name}) not connected")
        
        # Send arm pose command
        arm_cmd = yarp.Bottle()
        for val in pose:
            arm_cmd.addFloat32(float(val))
        
        self.arm_cmd_port.write(arm_cmd)
        
        # Update cached pose
        self._current_pose = pose.copy()
        self._current_fingers = fingers.copy()
    
    def _update_pose_from_encoders(self, joint_values: np.ndarray) -> None:
        """
        Update current pose estimate from encoder readings using direct kinematics.
        
        Args:
            joint_values: Joint encoder values
            
        TODO: Implement actual direct kinematics computation
        """
        # Placeholder: For now, just update finger values from known indices
        # In ergoCub, hand joints are typically at the end of the arm joint list
        if len(joint_values) >= 13:  # 7 arm joints + 6 hand joints
            # Extract finger joint values (adjust indices based on actual robot configuration)
            finger_start_idx = 7  # Assuming first 7 are arm joints
            self._current_fingers = joint_values[finger_start_idx:finger_start_idx+6]
        
        # TODO: Compute end-effector pose from arm joint values
        # self._current_pose = compute_direct_kinematics(joint_values[:7])


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
        self.neck_cmd_port = yarp.Port()
        self.encoders_port = yarp.BufferedPortVector()
        
        # Current state cache
        self._current_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # qx, qy, qz, qw
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    def connect(self) -> None:
        """Connect to YARP neck control ports."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError("ErgoCubNeckController already connected")
        
        # Open command port for neck orientation commands
        neck_cmd_local = f"{self.local_prefix}/neck/cmd:o"
        if not self.neck_cmd_port.open(neck_cmd_local):
            raise ConnectionError(f"Failed to open neck command port {neck_cmd_local}")
        
        # Connect to MetaControllServer neck command port
        neck_cmd_remote = "/metaControllServer/neck/cmd:i"
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
        
        # Read encoder data (non-blocking)
        bottle = self.encoders_port.read(False)
        if bottle is not None:
            # Extract neck joint values and compute orientation
            joint_values = np.array([bottle.get(i).asFloat64() for i in range(min(4, bottle.size()))])
            self._update_orientation_from_encoders(joint_values)
        
        # Return state in SO100-like format
        return {
            "neck.qx": float(self._current_orientation[0]),
            "neck.qy": float(self._current_orientation[1]),
            "neck.qz": float(self._current_orientation[2]),
            "neck.qw": float(self._current_orientation[3]),
        }
    
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
        qx, qy, qz, qw = q
        
        # Convert to rotation matrix
        rot_matrix = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
        ])
        
        # Send as 9-element vector (row-major order)
        neck_cmd = yarp.Bottle()
        for i in range(3):
            for j in range(3):
                neck_cmd.addFloat64(rot_matrix[i, j])
        
        self.neck_cmd_port.write(neck_cmd)
        
        # Update cached orientation
        self._current_orientation = orientation.copy()
    
    def _update_orientation_from_encoders(self, joint_values: np.ndarray) -> None:
        """
        Update current orientation estimate from neck encoder readings.
        
        Args:
            joint_values: Neck joint encoder values
            
        TODO: Implement actual neck kinematics computation
        """
        # Placeholder: For now, assume simple mapping from joint angles to quaternion
        # TODO: Implement proper neck kinematics
        pass


class ErgoCubMotorsBus(MotorsBus):
    """
    ErgoCub motors bus that manages arm and neck controllers.
    Follows SO100 MotorsBus conventions but operates at pose level.
    """
    
    def __init__(
        self,
        remote_prefix: str,
        local_prefix: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
        use_left_arm: bool = True,
        use_right_arm: bool = True,
        use_neck: bool = True,
    ):
        """
        Initialize ErgoCub motors bus.
        
        Args:
            remote_prefix: Remote YARP prefix (e.g., "/ergocubSim")
            local_prefix: Local YARP prefix (e.g., "/lerobot/session_id")
            motors: Motor configuration dictionary (for compatibility)
            calibration: Motor calibration (not used for ErgoCub)
            use_left_arm: Whether to enable left arm
            use_right_arm: Whether to enable right arm
            use_neck: Whether to enable neck
        """
        super().__init__("", motors, calibration)  # Pass empty port for compatibility
        
        self.remote_prefix = remote_prefix
        self.local_prefix = local_prefix
        self.use_left_arm = use_left_arm
        self.use_right_arm = use_right_arm
        self.use_neck = use_neck
        
        # Initialize controllers
        self.controllers = {}
        
        if self.use_left_arm:
            self.controllers["left_arm"] = ErgoCubArmController("left", remote_prefix, local_prefix)
            
        if self.use_right_arm:
            self.controllers["right_arm"] = ErgoCubArmController("right", remote_prefix, local_prefix)
            
        if self.use_neck:
            self.controllers["neck"] = ErgoCubNeckController(remote_prefix, local_prefix)
        
        # Initialize YARP network
        yarp.Network.init()
    
    @property
    def is_connected(self) -> bool:
        """Check if all enabled controllers are connected."""
        return all(controller.is_connected for controller in self.controllers.values())
    
    @property
    def is_calibrated(self) -> bool:
        """ErgoCub doesn't require calibration."""
        return True
    
    def connect(self) -> None:
        """Connect all controllers."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError("ErgoCubMotorsBus already connected")
        
        for name, controller in self.controllers.items():
            logger.info(f"Connecting {name} controller...")
            controller.connect()
        
        logger.info("ErgoCubMotorsBus connected")
    
    def disconnect(self, disable_torque: bool = False) -> None:
        """Disconnect all controllers."""
        if not self.is_connected:
            raise DeviceNotConnectedError("ErgoCubMotorsBus not connected")
        
        for name, controller in self.controllers.items():
            logger.info(f"Disconnecting {name} controller...")
            controller.disconnect()
        
        yarp.Network.fini()
        logger.info("ErgoCubMotorsBus disconnected")
    
    def sync_read(self, data_name: str) -> dict[str, float]:
        """
        Read current state from all controllers.
        
        Args:
            data_name: Data type to read (for compatibility, only "Present_Position" supported)
            
        Returns:
            Dictionary with current pose and finger states
        """
        if not self.is_connected:
            raise DeviceNotConnectedError("ErgoCubMotorsBus not connected")
        
        if data_name != "Present_Position":
            raise ValueError(f"Unsupported data type: {data_name}")
        
        state = {}
        
        # Read from all controllers
        for controller in self.controllers.values():
            controller_state = controller.read_current_state()
            state.update(controller_state)
        
        return state
    
    def sync_write(self, data_name: str, commands: dict[str, float]) -> None:
        """
        Send commands to all controllers.
        
        Args:
            data_name: Data type to write (for compatibility, only "Goal_Position" supported)
            commands: Dictionary with pose and finger commands
        """
        if not self.is_connected:
            raise DeviceNotConnectedError("ErgoCubMotorsBus not connected")
        
        if data_name != "Goal_Position":
            raise ValueError(f"Unsupported data type: {data_name}")
        
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
    
    # Required abstract methods from MotorsBus (not used for ErgoCub)
    def _get_half_turn_homings(self, positions):
        return {}
    
    def disable_torque(self, motors=None, num_retry=0):
        pass  # Not applicable for ErgoCub
    
    def enable_torque(self, motors=None, num_retry=0):
        pass  # Not applicable for ErgoCub
    
    def _encode_sign(self, data_name, ids_values):
        return ids_values
    
    def _decode_sign(self, data_name, ids_values):
        return ids_values
    
    @contextmanager
    def torque_disabled(self, motors=None):
        """Context manager for temporarily disabling torque (no-op for ErgoCub)."""
        yield
