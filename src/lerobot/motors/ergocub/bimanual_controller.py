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
from pysquale import BimanualIK, PoseInput
from lerobot.motors.ergocub.config_real import Config as cfg

logger = logging.getLogger(__name__)


class ErgoCubBimanualController:
    """
    Bimanual controller for ergoCub that controls both hands through a single port.
    Uses /mc-ergocub-cartesian-bimanual/rpc:i with hand side specified as string parameter.
    """
    
    def __init__(self, remote_prefix: str, local_prefix: str, use_left_hand: bool = True, use_right_hand: bool = True, use_torso: bool = True):
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

        if not use_left_hand:
            cfg.left_joints = np.array([])
        if not use_right_hand:
            cfg.right_joints= np.array([])
        # if not use_torso:  # TODO make this work
        #     cfg.torso_joints= np.array([])

        self.ik = BimanualIK(**cfg.to_dict())
        # Initial state (q, dq)
        q0  = np.zeros(len(cfg.right_joints) + len(cfg.left_joints) + len(cfg.torso_joints))
        dq0 = np.zeros(len(cfg.right_joints) + len(cfg.left_joints) + len(cfg.torso_joints))
        self.ik.reset(q0, dq0)
        
        # YARP ports
        self.left_encoders_port = yarp.BufferedPortBottle()
        self.right_encoders_port = yarp.BufferedPortBottle()
        self.torso_encoders_port = yarp.BufferedPortBottle()
        # use a control board remapper to control the joints
        # prepare mapping from joint name -> index in the single remapped controlboard
        self.joint_names = cfg.right_joints + cfg.left_joints + cfg.torso_joints
        # placeholders for YARP driver / interfaces (opened in connect)
        self._driver = None
        self._ipos = None

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
        if self._is_connected:
            raise DeviceAlreadyConnectedError("ErgoCubBimanualController already connected")
        
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

        # Open controlboard remapper PolyDriver for position control of all joints
        props = yarp.Property()
        props.put("robot", "ergocubSim")
        props.put("device", "remotecontrolboardremapper")
        props.put("localPortPrefix", f"{self.local_prefix}/bimanual/controlboard")
        remote_control_boards = yarp.Bottle()
        remote_control_boards_list = remote_control_boards.addList()
        for control_board in ["/ergocubSim/torso", "/ergocubSim/right_arm", "/ergocubSim/left_arm"]:
            remote_control_boards_list.addString(control_board)
        props.put("remoteControlBoards", remote_control_boards.get(0))
        axesNames = yarp.Bottle()
        axesNames_list = axesNames.addList()
        for joint in self.joint_names:
            axesNames_list.addString(joint)
        props.put("axesNames", axesNames.get(0))
        self._driver = yarp.PolyDriver()
        if not self._driver.open(props):
            raise ConnectionError(f"Failed to open PolyDriver for controlboard remapper")
        # Query position control interface
        self._ipos = self._driver.viewIPositionControl()
        if self._ipos is None:
            raise ConnectionError("IPositionControl interface not available on remapper driver")

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
        # Close driver if opened
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception:
                pass
            self._driver = None
            self._ipos = None
    
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
                rotation = R.from_matrix(pose_matrix[:3, :3])
                quaternion = rotation.as_quat(canonical=True, scalar_first=True)  # [x, y, z, w]
                
                # Add to state
                state.update({
                    f"{side}_hand.position.x": position[0].item(),
                    f"{side}_hand.position.y": position[1].item(),
                    f"{side}_hand.position.z": position[2].item(),
                    f"{side}_hand.orientation.qw": quaternion[0].item(),
                    f"{side}_hand.orientation.qx": quaternion[1].item(),
                    f"{side}_hand.orientation.qy": quaternion[2].item(),
                    f"{side}_hand.orientation.qz": quaternion[3].item(),
                })
                
                # Add actual finger values from encoders
                finger_joint_names = ["thumb_add", "thumb_oc", "index_add", "index_oc", "middle_oc", "ring_pinky_oc"]
                for i, joint in enumerate(finger_joint_names):
                    state[f"{side}_fingers.{joint}"] = finger_encoders[i] if i < len(finger_encoders) else 0.0
        
        return state
    
    def send_command(self, q: np.ndarray = None) -> None:
        """Send joint positions to the controlboard remapper.

        q should be a 1D numpy array with length equal to the number of joints in
        `self.joint_names` (left + right + torso order used above).
        """
        if not self.is_connected:
            raise DeviceNotConnectedError("ErgoCubBimanualController not connected")
        if q is None:
            return
        if q.size != len(self.joint_names):
            raise ValueError(f"Expected q of length {len(self.joint_names)}, got {q.size}")

        # If position control interface is available, send all positions
        if self._ipos is not None:
            try:
                pos = yarp.Vector()
                for i in range(len(self.joint_names)):
                    pos.push_back(np.degrees(q[i]))  # convert to degrees
                    # pos.push_back(q[i])
                ok = self._ipos.positionMove(pos.data())
                if not ok:
                    raise ConnectionError("IPositionControl.positionMove reported failure")
            except Exception as e:
                raise ConnectionError("Failed to send positions via IPositionControl: %s", e)
        else:
            # Fallback: log that positions would be sent
            raise ConnectionError("No IPositionControl available; would send positions: %s", q.tolist())

    
    def send_commands(self, commands: dict[str, float]) -> None:
        """Send commands to bimanual controller."""
        if not self.is_connected:
            raise DeviceNotConnectedError("ErgoCubBimanualController not connected")
        
        # Filter commands for each hand
        left_hand_cmds = {k.split(".", 1)[1]: v for k, v in commands.items() if k.startswith("left_hand.")}
        right_hand_cmds = {k.split(".", 1)[1]: v for k, v in commands.items() if k.startswith("right_hand.")}

        # Prepare inputs
        left_pose = PoseInput() if left_hand_cmds else None
        right_pose = PoseInput() if right_hand_cmds else None
        
        # Send hand commands
        for hand_name, hand_cmds in [("left", left_hand_cmds), ("right", right_hand_cmds)]:
            if hand_cmds:
                # Extract pose components - position.x, position.y, position.z, orientation qw,qx,qy,qz
                pos_keys = ["position.x", "position.y", "position.z"]
                # Build scalar-first so we can rotate slice with one-liner above
                quat_keys = ["orientation.qw", "orientation.qx", "orientation.qy", "orientation.qz"]
                
                if all(key in hand_cmds for key in pos_keys + quat_keys):
                    pose = np.array([hand_cmds[key] for key in pos_keys + quat_keys])
                    if hand_name == "left":
                        left_pose.pos = pose[:3]
                        left_pose.quat = np.r_[pose[4:7], pose[3]]
                    else:
                        right_pose.pos = pose[:3]
                        right_pose.quat = np.r_[pose[4:7], pose[3]]

        # Compute inverse kinematics and send commands
        q = self.ik.solve_ik(right_pose=right_pose, left_pose=left_pose)
        self.send_command(q)
        
        # Handle finger commands (bimanual controller doesn't actually control fingers, 
        # but we need to accept the commands to avoid errors during recording)
        for side in ["left", "right"]:
            finger_cmds = {k.split(".", 1)[1]: v for k, v in commands.items() if k.startswith(f"{side}_fingers.")}
            # For now, just log that finger commands were received but not processed
            if finger_cmds:
                logger.debug(f"Received {side} finger commands: {finger_cmds} (not processed by bimanual controller)")

    def reset(self) -> None:
        """Reset the bimanual controller"""
        self.send_command(np.array([0.0]*len(self.joint_names)))

    @property
    def motor_features(self) -> dict[str, type]:
        """Get motor features for bimanual controller."""
        features = {}
        
        # Left hand
        if self.use_left_hand:
            for coord in ["x", "y", "z"]:
                features[f"left_hand.position.{coord}"] = float
            for coord in ["qw", "qx", "qy", "qz"]:
                features[f"left_hand.orientation.{coord}"] = float
        
        # Right hand
        if self.use_right_hand:
            for coord in ["x", "y", "z"]:
                features[f"right_hand.position.{coord}"] = float
            for coord in ["qw", "qx", "qy", "qz"]:
                features[f"right_hand.orientation.{coord}"] = float
        
        return features
