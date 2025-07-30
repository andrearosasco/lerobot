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
from lerobot.errors import DeviceNotConnectedError
from lerobot.teleoperators.teleoperator import Teleoperator

from .configuration_metaquest import MetaQuestConfig


class MetaQuest(Teleoperator):
    # Required class attributes for LeRobot compatibility
    config_class = MetaQuestConfig
    name = "metaquest"
    
    def __init__(self, cfg: MetaQuestConfig):
        self.cfg = cfg
        self.session_id = uuid.uuid4()
        self._is_connected = False
        
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
            ("thumb_tip", 1),
            ("index_tip", 2), 
            ("middle_tip", 3),
            ("ring_tip", 4),
            ("pinky_tip", 5)
        ]

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
        
        # Use a try-except block to handle potential YARP errors gracefully
        try:
            success = self.tf_reader.getTransform(target_frame, reference_frame, self.matrix_buffer)
            if not success:
                print(f"Warning: Failed to get transform from {target_frame} to {reference_frame}")
                return np.eye(4)  # Return identity matrix as fallback
                
            # Convert YARP matrix to numpy array
            transform = np.zeros((4, 4))
            for i in range(4):
                for j in range(4):
                    transform[i, j] = self.matrix_buffer.get(i, j)
            
            return transform
            
        except Exception as e:
            print(f"Error getting transform from {target_frame} to {reference_frame}: {e}")
            return np.eye(4)  # Return identity matrix as fallback

    def _get_head_pose(self) -> dict:
        """Get raw head pose from MetaQuest."""
        transform = self._get_transform("openxr_head")
        
        # Extract position and rotation from 4x4 transformation matrix
        position = transform[:3, 3]
        rotation_matrix = transform[:3, :3]
        
        # Convert rotation matrix to quaternion (x, y, z, w)
        # Using Shepperd's method for numerical stability
        trace = np.trace(rotation_matrix)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        else:
            if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
                s = np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2
                qw = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
                qx = 0.25 * s
                qy = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                qz = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
                s = np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2
                qw = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
                qx = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                qy = 0.25 * s
                qz = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            else:
                s = np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2
                qw = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
                qx = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
                qy = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
                qz = 0.25 * s
        
        return {
            "position": {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])},
            "orientation": {"qx": float(qx), "qy": float(qy), "qz": float(qz), "qw": float(qw)}
        }

    def _get_hand_pose(self, side: str) -> dict:
        """Get raw hand pose from MetaQuest."""
        frame_name = f"openxr_{side}_hand"
        transform = self._get_transform(frame_name)
        
        # Extract position and rotation from 4x4 transformation matrix
        position = transform[:3, 3]
        rotation_matrix = transform[:3, :3]
        
        # Convert rotation matrix to quaternion (x, y, z, w)
        trace = np.trace(rotation_matrix)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        else:
            if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
                s = np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2
                qw = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
                qx = 0.25 * s
                qy = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                qz = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
                s = np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2
                qw = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
                qx = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                qy = 0.25 * s
                qz = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            else:
                s = np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2
                qw = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
                qx = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
                qy = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
                qz = 0.25 * s
        
        return {
            "position": {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])},
            "orientation": {"qx": float(qx), "qy": float(qy), "qz": float(qz), "qw": float(qw)}
        }

    def _get_finger_poses(self, side: str) -> dict:
        """Get raw finger poses from MetaQuest relative to hand frame."""
        finger_poses = {}
        
        hand_frame = f"openxr_{side}_hand_finger_0"  # Reference frame (hand)
        
        for finger_name, finger_index in self.finger_index_pairs:
            finger_frame = f"openxr_{side}_hand_finger_{finger_index}"
            transform = self._get_transform(finger_frame, hand_frame)
            
            # Extract position from transformation matrix
            position = transform[:3, 3]
            
            # Clean up finger name (remove _tip suffix for consistency)
            clean_name = finger_name.replace("_tip", "")
            
            finger_poses[clean_name] = {
                "x": float(position[0]),
                "y": float(position[1]), 
                "z": float(position[2])
            }
        
        return finger_poses

    def get_action(self) -> dict[str, Any]:
        """
        Get raw MetaQuest poses without any coordinate system transformations.
        Returns the raw poses from the MetaQuest in the openxr_origin frame.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            # Get head pose
            head_pose = self._get_head_pose()
            
            # Get hand poses
            left_hand_pose = self._get_hand_pose("left")
            right_hand_pose = self._get_hand_pose("right")
            
            # Get finger poses
            left_finger_poses = self._get_finger_poses("left")
            right_finger_poses = self._get_finger_poses("right")
            
            # Build flattened action dictionary that matches action_features
            action = {}
            
            # Head pose
            action["head.position.x"] = head_pose["position"]["x"]
            action["head.position.y"] = head_pose["position"]["y"] 
            action["head.position.z"] = head_pose["position"]["z"]
            action["head.orientation.qx"] = head_pose["orientation"]["qx"]
            action["head.orientation.qy"] = head_pose["orientation"]["qy"]
            action["head.orientation.qz"] = head_pose["orientation"]["qz"]
            action["head.orientation.qw"] = head_pose["orientation"]["qw"]
            
            # Left hand pose
            action["left_hand.position.x"] = left_hand_pose["position"]["x"]
            action["left_hand.position.y"] = left_hand_pose["position"]["y"]
            action["left_hand.position.z"] = left_hand_pose["position"]["z"]
            action["left_hand.orientation.qx"] = left_hand_pose["orientation"]["qx"]
            action["left_hand.orientation.qy"] = left_hand_pose["orientation"]["qy"]
            action["left_hand.orientation.qz"] = left_hand_pose["orientation"]["qz"]
            action["left_hand.orientation.qw"] = left_hand_pose["orientation"]["qw"]
            
            # Right hand pose
            action["right_hand.position.x"] = right_hand_pose["position"]["x"]
            action["right_hand.position.y"] = right_hand_pose["position"]["y"]
            action["right_hand.position.z"] = right_hand_pose["position"]["z"]
            action["right_hand.orientation.qx"] = right_hand_pose["orientation"]["qx"]
            action["right_hand.orientation.qy"] = right_hand_pose["orientation"]["qy"]
            action["right_hand.orientation.qz"] = right_hand_pose["orientation"]["qz"]
            action["right_hand.orientation.qw"] = right_hand_pose["orientation"]["qw"]
            
            # Left finger poses
            for finger_name in ["thumb", "index", "middle", "ring", "pinky"]:
                if finger_name in left_finger_poses:
                    action[f"left_fingers.{finger_name}.x"] = left_finger_poses[finger_name]["x"]
                    action[f"left_fingers.{finger_name}.y"] = left_finger_poses[finger_name]["y"]
                    action[f"left_fingers.{finger_name}.z"] = left_finger_poses[finger_name]["z"]
                else:
                    action[f"left_fingers.{finger_name}.x"] = 0.0
                    action[f"left_fingers.{finger_name}.y"] = 0.0
                    action[f"left_fingers.{finger_name}.z"] = 0.0
            
            # Right finger poses
            for finger_name in ["thumb", "index", "middle", "ring", "pinky"]:
                if finger_name in right_finger_poses:
                    action[f"right_fingers.{finger_name}.x"] = right_finger_poses[finger_name]["x"]
                    action[f"right_fingers.{finger_name}.y"] = right_finger_poses[finger_name]["y"]
                    action[f"right_fingers.{finger_name}.z"] = right_finger_poses[finger_name]["z"]
                else:
                    action[f"right_fingers.{finger_name}.x"] = 0.0
                    action[f"right_fingers.{finger_name}.y"] = 0.0
                    action[f"right_fingers.{finger_name}.z"] = 0.0
                
        except Exception as e:
            print(f"Error getting MetaQuest data: {e}")
            # Return default action if there's an error
            action = self._get_default_flattened_action()
        
        return action

    def _get_default_flattened_action(self) -> dict[str, Any]:
        """Return a default flattened action structure with safe values."""
        action = {}
        
        # Default head pose (identity orientation)
        action["head.position.x"] = 0.0
        action["head.position.y"] = 0.0
        action["head.position.z"] = 0.0
        action["head.orientation.qx"] = 0.0
        action["head.orientation.qy"] = 0.0
        action["head.orientation.qz"] = 0.0
        action["head.orientation.qw"] = 1.0
        
        # Default hand poses (identity orientations)
        for side in ["left_hand", "right_hand"]:
            action[f"{side}.position.x"] = 0.0
            action[f"{side}.position.y"] = 0.0
            action[f"{side}.position.z"] = 0.0
            action[f"{side}.orientation.qx"] = 0.0
            action[f"{side}.orientation.qy"] = 0.0
            action[f"{side}.orientation.qz"] = 0.0
            action[f"{side}.orientation.qw"] = 1.0
        
        # Default finger poses (zero positions)
        for side in ["left_fingers", "right_fingers"]:
            for finger_name in ["thumb", "index", "middle", "ring", "pinky"]:
                action[f"{side}.{finger_name}.x"] = 0.0
                action[f"{side}.{finger_name}.y"] = 0.0
                action[f"{side}.{finger_name}.z"] = 0.0
        
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
            "head.orientation.qx": float,
            "head.orientation.qy": float,
            "head.orientation.qz": float,
            "head.orientation.qw": float,
            
            # Left hand pose
            "left_hand.position.x": float,
            "left_hand.position.y": float,
            "left_hand.position.z": float,
            "left_hand.orientation.qx": float,
            "left_hand.orientation.qy": float,
            "left_hand.orientation.qz": float,
            "left_hand.orientation.qw": float,
            
            # Right hand pose
            "right_hand.position.x": float,
            "right_hand.position.y": float,
            "right_hand.position.z": float,
            "right_hand.orientation.qx": float,
            "right_hand.orientation.qy": float,
            "right_hand.orientation.qz": float,
            "right_hand.orientation.qw": float,
            
            # Left finger positions (relative to left hand)
            "left_fingers.thumb.x": float,
            "left_fingers.thumb.y": float,
            "left_fingers.thumb.z": float,
            "left_fingers.index.x": float,
            "left_fingers.index.y": float,
            "left_fingers.index.z": float,
            "left_fingers.middle.x": float,
            "left_fingers.middle.y": float,
            "left_fingers.middle.z": float,
            "left_fingers.ring.x": float,
            "left_fingers.ring.y": float,
            "left_fingers.ring.z": float,
            "left_fingers.pinky.x": float,
            "left_fingers.pinky.y": float,
            "left_fingers.pinky.z": float,
            
            # Right finger positions (relative to right hand)
            "right_fingers.thumb.x": float,
            "right_fingers.thumb.y": float,
            "right_fingers.thumb.z": float,
            "right_fingers.index.x": float,
            "right_fingers.index.y": float,
            "right_fingers.index.z": float,
            "right_fingers.middle.x": float,
            "right_fingers.middle.y": float,
            "right_fingers.middle.z": float,
            "right_fingers.ring.x": float,
            "right_fingers.ring.y": float,
            "right_fingers.ring.z": float,
            "right_fingers.pinky.x": float,
            "right_fingers.pinky.y": float,
            "right_fingers.pinky.z": float,
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
