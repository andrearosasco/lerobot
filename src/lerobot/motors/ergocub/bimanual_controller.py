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

import numpy as np
import yarp
from scipy.spatial.transform import Rotation as R
from .urdf_utils import resolve_ergocub_urdf
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.model.kinematics import RobotKinematics
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
import torch

logger = logging.getLogger(__name__)


class ErgoCubBimanualController:
    """
    Bimanual controller for ergoCub that controls both hands through a single port.
    Uses /mc-ergocub-cartesian-bimanual/rpc:i with hand side specified as string parameter.
    """
    
    def __init__(self, remote_prefix: str, local_prefix: str, use_left_hand: bool = True, use_right_hand: bool = True):
        """
        Initialize bimanual controller.
        
        Args:
            remote_prefix: Remote YARP prefix (e.g., "/ergocubSim")
            local_prefix: Local YARP prefix (e.g., "/lerobot/session_id")
            use_left_hand: Whether to control left hand
            use_right_hand: Whether to control right hand
        """
        self.remote_prefix = remote_prefix
        self.local_prefix = local_prefix
        self.use_left_hand = use_left_hand
        self.use_right_hand = use_right_hand
        
        # YARP ports
        self.bimanual_cmd_port = yarp.RpcClient()
        self.left_encoders_port = yarp.BufferedPortBottle()
        self.right_encoders_port = yarp.BufferedPortBottle()
        self.torso_encoders_port = yarp.BufferedPortBottle()
        
        self._is_connected = False
        
        # Initialize kinematics solvers for both hands if needed
        self.kinematics_solvers = {}
        if use_left_hand or use_right_hand:
            urdf_file = resolve_ergocub_urdf()
            
            if use_left_hand:
                left_joint_names = ["torso_roll", "torso_pitch", "torso_yaw", "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow", "l_wrist_yaw", "l_wrist_roll", "l_wrist_pitch"]
                self.kinematics_solvers["left"] = RobotKinematics(urdf_file, "l_hand_palm", left_joint_names)
                
            if use_right_hand:
                right_joint_names = ["torso_roll", "torso_pitch", "torso_yaw", "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow", "r_wrist_yaw", "r_wrist_roll", "r_wrist_pitch"]
                self.kinematics_solvers["right"] = RobotKinematics(urdf_file, "r_hand_palm", right_joint_names)
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    def connect(self) -> None:
        """Connect to YARP bimanual control port."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError("ErgoCubBimanualController already connected")
        
        # Open RPC port for bimanual commands
        bimanual_cmd_local = f"{self.local_prefix}/bimanual/rpc:o"
        if not self.bimanual_cmd_port.open(bimanual_cmd_local):
            raise ConnectionError(f"Failed to open bimanual RPC port {bimanual_cmd_local}")
        
        # Connect to bimanual cartesian controller
        bimanual_cmd_remote = "/mc-ergocub-cartesian-bimanual/rpc:i"
        while not yarp.Network.connect(bimanual_cmd_local, bimanual_cmd_remote):
            logger.warning(f"Failed to connect {bimanual_cmd_local} -> {bimanual_cmd_remote}, retrying...")
            time.sleep(1)
        
        # Connect encoder ports for both hands
        if self.use_left_hand:
            left_encoders_local = f"{self.local_prefix}/bimanual/left_encoders:i"
            if not self.left_encoders_port.open(left_encoders_local):
                raise ConnectionError(f"Failed to open left encoders port {left_encoders_local}")
            
            left_encoders_remote = f"{self.remote_prefix}/left_arm/state:o"
            while not yarp.Network.connect(left_encoders_remote, left_encoders_local):
                logger.warning(f"Failed to connect {left_encoders_remote} -> {left_encoders_local}, retrying...")
                time.sleep(1)
        
        if self.use_right_hand:
            right_encoders_local = f"{self.local_prefix}/bimanual/right_encoders:i"
            if not self.right_encoders_port.open(right_encoders_local):
                raise ConnectionError(f"Failed to open right encoders port {right_encoders_local}")
            
            right_encoders_remote = f"{self.remote_prefix}/right_arm/state:o"
            while not yarp.Network.connect(right_encoders_remote, right_encoders_local):
                logger.warning(f"Failed to connect {right_encoders_remote} -> {right_encoders_local}, retrying...")
                time.sleep(1)
        
        # Connect torso encoders
        torso_encoders_local = f"{self.local_prefix}/bimanual/torso_encoders:i"
        if not self.torso_encoders_port.open(torso_encoders_local):
            raise ConnectionError(f"Failed to open torso encoders port {torso_encoders_local}")
        
        torso_encoders_remote = f"{self.remote_prefix}/torso/state:o"
        while not yarp.Network.connect(torso_encoders_remote, torso_encoders_local):
            logger.warning(f"Failed to connect {torso_encoders_remote} -> {torso_encoders_local}, retrying...")
            time.sleep(1)
        
        self._is_connected = True
        logger.info("ErgoCubBimanualController connected")
    
    def disconnect(self) -> None:
        """Disconnect from YARP ports."""
        if not self.is_connected:
            raise DeviceNotConnectedError("ErgoCubBimanualController not connected")
        
        self.bimanual_cmd_port.close()
        if self.use_left_hand:
            self.left_encoders_port.close()
        if self.use_right_hand:
            self.right_encoders_port.close()
        self.torso_encoders_port.close()
        
        self._is_connected = False
        logger.info("ErgoCubBimanualController disconnected")
    
    def read_current_state(self) -> Dict[str, float]:
        """Read current state for both hands."""
        if not self.is_connected:
            raise DeviceNotConnectedError("ErgoCubBimanualController not connected")
        
        state = {}
        
        # Read torso encoders
        torso_bottle = self.torso_encoders_port.read(False)
        if torso_bottle:
            torso_encoders = [torso_bottle.get(i).asFloat64() for i in range(min(3, torso_bottle.size()))]
        else:
            torso_encoders = [0.0, 0.0, 0.0]
        
        # Read and compute poses for each hand
        for side in ["left", "right"]:
            if (side == "left" and not self.use_left_hand) or (side == "right" and not self.use_right_hand):
                continue
                
            # Read hand encoders with busy wait
            encoders_port = self.left_encoders_port if side == "left" else self.right_encoders_port
            read_attempts = 0
            while (hand_bottle := encoders_port.read(False)) is None:
                read_attempts += 1
                if read_attempts % 1000 == 0:  # Warning every 1000 attempts
                    logger.warning(f"Still waiting for {side} hand encoder data (attempt {read_attempts})")
                time.sleep(0.001)  # 1 millisecond sleep
            
            # Read all available encoders (hand + fingers)
            all_encoders = [hand_bottle.get(i).asFloat64() for i in range(hand_bottle.size())]
            # First 7 are hand joints, last 6 are finger joints
            hand_encoders = all_encoders[:7] if len(all_encoders) >= 7 else [0.0] * 7
            finger_encoders = all_encoders[7:13] if len(all_encoders) >= 13 else [0.0] * 6
            
            # Combine torso + hand encoders
            joint_positions = np.array(torso_encoders + hand_encoders)
            
            # Compute forward kinematics
            if side in self.kinematics_solvers:
                pose_matrix = self.kinematics_solvers[side].forward_kinematics(joint_positions)
                position = pose_matrix[:3, 3]
                rotation_6d = matrix_to_rotation_6d(torch.tensor(pose_matrix[:3, :3], dtype=torch.float32)).numpy().flatten()
                
                # Add to state
                state.update({
                    f"{side}_hand.position.x": position[0].item(),
                    f"{side}_hand.position.y": position[1].item(),
                    f"{side}_hand.position.z": position[2].item(),
                    f"{side}_hand.orientation.d1": rotation_6d[0].item(),
                    f"{side}_hand.orientation.d2": rotation_6d[1].item(),
                    f"{side}_hand.orientation.d3": rotation_6d[2].item(),
                    f"{side}_hand.orientation.d4": rotation_6d[3].item(),
                    f"{side}_hand.orientation.d5": rotation_6d[4].item(),
                    f"{side}_hand.orientation.d6": rotation_6d[5].item(),
                })
                
                # Add actual finger values from encoders
                finger_joint_names = ["thumb_add", "thumb_oc", "index_add", "index_oc", "middle_oc", "ring_pinky_oc"]
                for i, joint in enumerate(finger_joint_names):
                    state[f"{side}_fingers.{joint}"] = finger_encoders[i] if i < len(finger_encoders) else 0.0
        
        return state
    
    def send_command(self, left_pose: np.ndarray = None, right_pose: np.ndarray = None) -> None:
        """
        Send commands to bimanual controller.
        
        Args:
            left_pose: Left hand pose [x, y, z, qw, qx, qy, qz] (optional)
            left_fingers: Left finger commands (optional)
            right_pose: Right hand pose [x, y, z, qw, qx, qy, qz] (optional)
            right_fingers: Right finger commands (optional)
        """
        if not self.is_connected:
            raise DeviceNotConnectedError("ErgoCubBimanualController not connected")

        # Convert 6d rotation to quaternion for both hands
        if left_pose is not None:
            rot_matrix_left = torch.tensor(rotation_6d_to_matrix(torch.tensor(left_pose[3:9], dtype=torch.float32)), dtype=torch.float32)
            quat_left = R.from_matrix(rot_matrix_left.numpy()).as_quat(canonical=True, scalar_first=False)  # [x, y, z, w]
            left_pose = np.concatenate([left_pose[0:3], quat_left])
        if right_pose is not None:
            rot_matrix_right = torch.tensor(rotation_6d_to_matrix(torch.tensor(right_pose[3:9], dtype=torch.float32)), dtype=torch.float32)
            quat_right = R.from_matrix(rot_matrix_right.numpy()).as_quat(canonical=True, scalar_first=False)  # [x, y, z, w]
            right_pose = np.concatenate([right_pose[0:3], quat_right])

        # Send left hand command
        if left_pose is not None and self.use_left_hand:
            left_cmd = yarp.Bottle()
            left_cmd.addString("go_to_pose")
            for val in left_pose:
                left_cmd.addFloat64(float(val))
            left_cmd.addString("left")  # hand side specification
            
            reply = yarp.Bottle()
            self.bimanual_cmd_port.write(left_cmd, reply)
        
        # Send right hand command
        if right_pose is not None and self.use_right_hand:
            right_cmd = yarp.Bottle()
            right_cmd.addString("go_to_pose")
            for val in right_pose:
                right_cmd.addFloat64(float(val))
            right_cmd.addString("right")  # hand side specification
            
            reply = yarp.Bottle()
            self.bimanual_cmd_port.write(right_cmd, reply)
    
    def send_commands(self, commands: dict[str, float]) -> None:
        """Send commands to bimanual controller."""
        if not self.is_connected:
            raise DeviceNotConnectedError("ErgoCubBimanualController not connected")
        
        # Filter commands for each hand
        left_hand_cmds = {k.split(".", 1)[1]: v for k, v in commands.items() if k.startswith("left_hand.")}
        right_hand_cmds = {k.split(".", 1)[1]: v for k, v in commands.items() if k.startswith("right_hand.")}
        
        # Send hand commands
        for hand_name, hand_cmds in [("left", left_hand_cmds), ("right", right_hand_cmds)]:
            if hand_cmds:
                # Extract pose components - position.x, position.y, position.z, orientation qw,qx,qy,qz
                pos_keys = ["position.x", "position.y", "position.z"]
                # Build scalar-first so we can rotate slice with one-liner above
                keys_6d = ["orientation.d1", "orientation.d2", "orientation.d3", "orientation.d4", "orientation.d5", "orientation.d6"]
                
                if all(key in hand_cmds for key in pos_keys + keys_6d):
                    pose = np.array([hand_cmds[key] for key in pos_keys + keys_6d])
                    if hand_name == "left":
                        self.send_command(left_pose=pose)
                    else:
                        self.send_command(right_pose=pose)
        
        # Handle finger commands (bimanual controller doesn't actually control fingers, 
        # but we need to accept the commands to avoid errors during recording)
        for side in ["left", "right"]:
            finger_cmds = {k.split(".", 1)[1]: v for k, v in commands.items() if k.startswith(f"{side}_fingers.")}
            # For now, just log that finger commands were received but not processed
            if finger_cmds:
                logger.debug(f"Received {side} finger commands: {finger_cmds} (not processed by bimanual controller)")

    def reset(self) -> None:
        """Reset the bimanual controller"""
        right_cmd = yarp.Bottle()
        right_cmd.addString("go_home")
        
        reply = yarp.Bottle()
        self.bimanual_cmd_port.write(right_cmd, reply)

    @property
    def motor_features(self) -> dict[str, type]:
        """Get motor features for bimanual controller."""
        features = {}
        
        # Left hand
        if self.use_left_hand:
            for coord in ["x", "y", "z"]:
                features[f"left_hand.position.{coord}"] = float
            for coord in ["d1", "d2", "d3", "d4", "d5", "d6"]:
                features[f"left_hand.orientation.{coord}"] = float
        
        # Right hand
        if self.use_right_hand:
            for coord in ["x", "y", "z"]:
                features[f"right_hand.position.{coord}"] = float
            for coord in ["d1", "d2", "d3", "d4", "d5", "d6"]:
                features[f"right_hand.orientation.{coord}"] = float
        
        return features
