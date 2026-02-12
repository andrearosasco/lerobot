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

import time
import uuid
from typing import Any

import numpy as np
import yarp
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.teleoperators.teleoperator import Teleoperator

from .configuration_metaquest import MetaQuestConfig
from scipy.spatial.transform import Rotation as R
from lerobot.robots.ergocub.manipulator import Manipulator
from lerobot.utils.rotation import matrix_to_rotation_6d
import torch

HEAD_TO_ROOT = np.array([
    [1.0, 0.0, 0.0, 0.005],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.547],
    [0.0, 0.0, 0.0, 1.0]
])
RHS_WO_HEAD_I_TRANSF = np.array([
    [0.0, 0.0, -1.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])
HEAD_ADAPTER = np.array([
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])
RIGHT_HAND_ADAPTER = np.array([
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])
LEFT_HAND_ADAPTER = np.eye(4)
QUEST_TO_ECUB = HEAD_TO_ROOT @ RHS_WO_HEAD_I_TRANSF


class MetaQuest(Teleoperator):
    # Required class attributes for LeRobot compatibility
    config_class = MetaQuestConfig
    name = "metaquest"
    
    def __init__(self, cfg: MetaQuestConfig):
        self.cfg = cfg
        self.session_id = uuid.uuid4()
        self._is_connected = False

        # Optionally set finger rescale from config, default 1.0
        self.finger_scale = getattr(cfg, 'finger_scale', 1.0)

        # Call parent constructor after setting cfg
        super().__init__(cfg)

        # Initialize YARP network
        yarp.Network.init()

        # FrameTransform client will be created in connect() method
        self.tf_driver = None
        self.tf_reader = None
        self.matrix_buffer = yarp.Matrix(4, 4)

        # Finger frame mappings for MetaQuest
        self.finger_index_pairs = [
            ("thumb_tip", 5),
            ("index_tip", 10), 
            ("middle_tip", 15),
            ("ring_tip", 20),
            ("little_tip", 25)
        ]

        # Joint names for each hand (same order for left and right)
        self.joint_names = ["thumb_add", "thumb_oc", "index_add", "index_oc", "middle_oc", "ring_pinky_oc"]
        # Per-hand kinematics solvers for fingertip-to-joint IK
        self.finger_kinematics = {
            "left": Manipulator("src/lerobot/robots/ergocub/ergocub_hand_left/model.urdf"),
            "right": Manipulator("src/lerobot/robots/ergocub/ergocub_hand_right/model.urdf"),
        }

    def _do_inverse_fingers_kinematics(self, side: str, finger_positions: list[np.ndarray]) -> list[float]:
            self.finger_kinematics[side].inverse_kinematic(finger_positions)
            joint_values = self.finger_kinematics[side].get_driver_value()
            return np.degrees(joint_values).tolist()

    def __del__(self):
        """Destructor to ensure proper cleanup and avoid memory leaks."""
        try:
            if self._is_connected:
                self.disconnect()
        except:
            pass  # Ignore errors during cleanup

    def connect(self, calibrate: bool = True):
        """Connect to iFrameTransform to get raw MetaQuest poses."""
        # Check if YARP network is available
        if not yarp.Network.checkNetwork():
            raise DeviceNotConnectedError("YARP network is not available. Please start yarpserver first.")
        
        # Set up frameTransformClient configuration
        tf_client_cfg = yarp.Property()
        tf_client_cfg.put("device", self.cfg.tf_device)
        tf_client_cfg.put("filexml_option", self.cfg.tf_file)
        tf_client_cfg.put("ft_client_prefix", f"/metaquest_{self.session_id}/tf")
        tf_client_cfg.put("ftc_storage_timeout", 300.0)
        
        # Only set remote server if specified and reachable
        if hasattr(self.cfg, 'tf_remote') and self.cfg.tf_remote:
            # Check if the frameTransformServer is reachable
            server_port = f"{self.cfg.tf_remote}/rpc:i"
            if yarp.Network.exists(server_port):
                tf_client_cfg.put("ft_server_prefix", self.cfg.tf_remote)
                print(f"Connecting to frameTransformServer at {self.cfg.tf_remote}")
            else:
                print(f"Warning: frameTransformServer not found at {self.cfg.tf_remote}")
                # Try without remote server (local mode)
                print("Attempting to connect in local mode...")
        
        tf_client_cfg.put("local_rpc", f"/metaquest_{self.session_id}/tf/local_rpc")
        
        print(f"frameTransformClient configuration: {tf_client_cfg.toString()}")
        
        # Open transform client driver
        self.tf_driver = yarp.PolyDriver(tf_client_cfg)
        if not self.tf_driver.isValid():
            # Provide more detailed error information
            error_msg = f"Unable to open frameTransformClient. Possible issues:\n"
            error_msg += f"1. frameTransformServer is not running (try: frameTransformServer --from ftServer.xml)\n"
            error_msg += f"2. YARP ports are not accessible\n"
            error_msg += f"3. Configuration file '{self.cfg.tf_file}' not found\n"
            error_msg += f"Config used: {tf_client_cfg.toString()}"
            raise DeviceNotConnectedError(error_msg)
        
        # Get IFrameTransform interface using the correct YARP method
        self.tf_reader = self.tf_driver.viewIFrameTransform()
        if not self.tf_reader:
            raise DeviceNotConnectedError("Unable to view IFrameTransform interface")
            
        self._is_connected = True
        print("Connected to iFrameTransform successfully")

    def disconnect(self):
        """Close frameTransform connections and disconnect."""
        # Properly cleanup the interface to avoid memory leaks
        if hasattr(self, 'tf_reader'):
            self.tf_reader = None
        
        if self.tf_driver and self.tf_driver.isValid():
            self.tf_driver.close()
            
        self.tf_driver = None
        yarp.Network.fini()
        self._is_connected = False

    def _get_transform(self, target_frame: str, reference_frame: str = "openxr_origin") -> np.ndarray:
        """Get transformation matrix from iFrameTransform."""
        if not self.tf_reader:
            raise DeviceNotConnectedError("FrameTransform reader not available")
        
        while not self.tf_reader.getTransform(target_frame, reference_frame, self.matrix_buffer):
            time.sleep(0.01)  # Wait a bit before retrying
            
        # Convert YARP matrix to numpy array
        transform = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                transform[i, j] = self.matrix_buffer.get(i, j)
        
        return transform
            

    def _get_head_pose(self) -> dict:
        """Get raw head pose from MetaQuest."""
        transform = self._get_transform("openxr_head")
        transform = (QUEST_TO_ECUB @ transform @ HEAD_ADAPTER)

        position = transform[:3, 3]
        # quat = R.from_matrix(transform[:3, :3]).as_quat(canonical=True, scalar_first=True)  # [w, x, y, z]
        pose_6d = matrix_to_rotation_6d(torch.tensor(R.from_matrix(transform[:3, :3]).as_matrix())).numpy()
        return np.r_[position, pose_6d]


    def _get_hand_pose(self, side: str) -> dict:
        """Get raw hand pose from MetaQuest."""
        frame_name = f"openxr_{side}_hand_joint_wrist"
        transform = self._get_transform(frame_name)
        HAND_ADAPTER = LEFT_HAND_ADAPTER if side == "left" else RIGHT_HAND_ADAPTER
        transform = (QUEST_TO_ECUB @ transform @ HAND_ADAPTER)

        position = transform[:3, 3]
        rot_6d = matrix_to_rotation_6d(torch.tensor(transform[:3, :3], dtype=torch.float32)).numpy().flatten()
        return np.r_[position, rot_6d]

    def _get_finger_poses(self, side: str) -> dict:
        """Get raw finger poses from MetaQuest relative to hand frame."""
        hand_frame = f"openxr_{side}_hand_joint_palm"  # Reference frame (hand)
        positions = []

        for finger_name, _ in self.finger_index_pairs:
            finger_frame = f"openxr_{side}_hand_joint_{finger_name}"
            transform = self._get_transform(finger_frame, hand_frame)

            # Extract position from transformation matrix
            position = transform[:3, 3]
            # do here the finger rescaling!!!
            position = position * self.finger_scale
            positions.append(position)

        return positions

    def get_action(self) -> dict[str, Any]:
        """
        Get raw MetaQuest poses without any coordinate system transformations.
        Returns the raw poses from the MetaQuest in the openxr_origin frame.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Get head pose
        head_pose = self._get_head_pose()
        
        # Get hand poses
        left_hand_pose = self._get_hand_pose("left")
        right_hand_pose = self._get_hand_pose("right")
        
        # Get finger poses
        left_finger_poses = self._get_finger_poses("left")
        right_finger_poses = self._get_finger_poses("right")

        # Apply inverse fingers kinematics
        left_fingers_joints = self._do_inverse_fingers_kinematics("left", left_finger_poses)
        right_fingers_joints = self._do_inverse_fingers_kinematics("right", right_finger_poses)
        
        # Build flattened action dictionary that matches action_features
        keys = list(self.action_features.keys())
        vals = np.concatenate([head_pose, left_hand_pose, right_hand_pose, left_fingers_joints, right_fingers_joints])
        action = dict(zip(keys, vals.tolist()))
        return action


    @property
    def action_features(self):
        """
        Return action features for raw MetaQuest poses (no transformations).
        Provides head, hand positions/orientations, and finger positions.
        """
        return {
            # Head pose
            "head.position.x": float,
            "head.position.y": float, 
            "head.position.z": float,
            "head.orientation.d1": float,
            "head.orientation.d2": float,
            "head.orientation.d3": float,
            "head.orientation.d4": float,
            "head.orientation.d5": float,
            "head.orientation.d6": float,
            
            # Left hand pose
            "left_hand.position.x": float,
            "left_hand.position.y": float,
            "left_hand.position.z": float,
            "left_hand.orientation.d1": float,
            "left_hand.orientation.d2": float,
            "left_hand.orientation.d3": float,
            "left_hand.orientation.d4": float,
            "left_hand.orientation.d5": float,
            "left_hand.orientation.d6": float,
            
            # Right hand pose
            "right_hand.position.x": float,
            "right_hand.position.y": float,
            "right_hand.position.z": float,
            "right_hand.orientation.d1": float,
            "right_hand.orientation.d2": float,
            "right_hand.orientation.d3": float,
            "right_hand.orientation.d4": float,
            "right_hand.orientation.d5": float,
            "right_hand.orientation.d6": float,
            
            # Left finger joints
            "left_fingers.thumb_add": float,
            "left_fingers.thumb_oc": float,
            "left_fingers.index_add": float,
            "left_fingers.index_oc": float,
            "left_fingers.middle_oc": float,
            "left_fingers.ring_pinky_oc": float,
            
            # Right finger joints
            "right_fingers.thumb_add": float,
            "right_fingers.thumb_oc": float,
            "right_fingers.index_add": float,
            "right_fingers.index_oc": float,
            "right_fingers.middle_oc": float,
            "right_fingers.ring_pinky_oc": float,
        }

    @property
    def feedback_features(self) -> dict:
        """MetaQuest teleop doesn't provide feedback features."""
        return {}

    @property
    def is_connected(self) -> bool:
        """Whether the teleoperator is currently connected."""
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        """MetaQuest teleop doesn't require calibration."""
        return True

    def calibrate(self) -> None:
        """MetaQuest teleop doesn't require calibration - no-op."""
        pass

    def configure(self) -> None:
        """Apply any one-time configuration - no-op for MetaQuest teleop."""
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """MetaQuest teleop doesn't support feedback - no-op."""
        pass
