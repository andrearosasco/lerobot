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
        
        # YARP port will be created in connect() method
        self.port = None

    def connect(self, calibrate: bool = True):
        """Connect to YARP action streams."""
        # Create and open YARP port
        self.port = yarp.BufferedPortBottle()
        self.port.open(f"{self.cfg.local_prefix}/{self.session_id}/action:i")
        
        # Connect to remote action port
        remote_name = f"{self.cfg.remote_prefix}/action:o"
        local_name = f"{self.cfg.local_prefix}/{self.session_id}/action:i"
        
        while not yarp.Network.connect(remote_name, local_name):
            print(f"Waiting for {remote_name} port to connect...")
            time.sleep(0.1)
            
        self._is_connected = True

    def disconnect(self):
        """Close YARP connections and disconnect."""
        if hasattr(self, 'port') and self.port:
            self.port.close()
        yarp.Network.fini()
        self._is_connected = False

    def _read_yarp_data(self):
        """Read action data from YARP streams."""
        if not self.port:
            return [{"data": {}}]
            
        # Use proper non-blocking read with waiting loop
        read_attempts = 0
        while (bottle := self.port.read(False)) is None:
            read_attempts += 1
            if read_attempts % 1000 == 0:
                print(f"Still waiting for action data (attempt {read_attempts})")
            time.sleep(0.001)  # Small sleep to avoid busy waiting
        
        # Parse bottle data using structured approach
        data = self._parse_bottle(bottle)
        
        # Return data in a format similar to polars DataFrame
        return [{"data": data}]
    
    def _parse_bottle(self, bottle):
        """Parse YARP bottle into structured data using the same approach as ActionInterface."""
        # Define format for parsing - similar to ActionInterface
        format_map = {
            'neck': ['float'] * 9,  # Adjust based on actual neck data size
            'left_arm': ['float'] * 7,
            'right_arm': ['float'] * 7,
            'fingers': ['float'] * 12 # 10 fingers, 3 values each
        }
        
        data = {}
        
        # Parse each control board using bottle.find() like the working interface
        for board_name in self.cfg.control_boards:
            if board_name in format_map:
                board_bottle = bottle.find(board_name)
                if not board_bottle.isNull():
                    parsed_data = self._parse_bottle_recursive(board_bottle, format_map[board_name])
                    if parsed_data is not None:
                        if board_name == "fingers":
                            # Convert list of lists to numpy array for fingers
                            data[board_name] = np.array(parsed_data)
                        else:
                            # Convert list to numpy array for other boards
                            data[board_name] = np.array(parsed_data)
        
        return data
    
    def _parse_bottle_recursive(self, bottle, format_spec):
        """Recursive bottle parsing like ActionInterface.parse_bottle."""
        if isinstance(format_spec, list):
            if bottle.asList().size() == 0:
                return None
            result = []
            bottle_list = bottle.asList()
            for i in range(len(format_spec)):
                if i < bottle_list.size():
                    parsed_value = self._parse_bottle_recursive(bottle_list.get(i), format_spec[i])
                    if parsed_value is None:
                        return None
                    result.append(parsed_value)
                else:
                    return None
            return result
        elif isinstance(format_spec, str):
            if format_spec == 'float':
                return bottle.asFloat64()
            elif format_spec == 'int':
                return bottle.asInt()
            elif format_spec == 'string':
                return bottle.asString()
            else:
                raise ValueError(f"Unsupported format: {format_spec}")
        else:
            raise ValueError(f"Unsupported format: {format_spec}")

    def get_action(self) -> dict[str, Any]:
        """
        Get action from MetaQuest teleoperator in ergoCub-compatible format.
        Returns flattened dictionary matching ergoCub.action_features.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        for data_row in self._read_yarp_data():
            action = {}
            board_data = data_row["data"]
            
            # Process each control board
            for board_name in self.cfg.control_boards:
                if board_name in board_data:
                    data = board_data[board_name]
                    
                    if board_name == "neck":
                        # Map neck data to quaternion format (4 values: qx, qy, qz, qw)
                        if len(data) >= 4:
                            action["neck.qx"] = float(data[0])
                            action["neck.qy"] = float(data[1]) 
                            action["neck.qz"] = float(data[2])
                            action["neck.qw"] = float(data[3])
                        else:
                            # Fill missing values with defaults
                            action["neck.qx"] = float(data[0]) if len(data) > 0 else 0.0
                            action["neck.qy"] = float(data[1]) if len(data) > 1 else 0.0
                            action["neck.qz"] = float(data[2]) if len(data) > 2 else 0.0
                            action["neck.qw"] = 1.0  # Default quaternion w
                    
                    elif board_name in ["left_arm", "right_arm"]:
                        # Map arm data to position + quaternion format (7 values: x,y,z,qx,qy,qz,qw)
                        if len(data) >= 7:
                            action[f"{board_name}.x"] = float(data[0])
                            action[f"{board_name}.y"] = float(data[1])
                            action[f"{board_name}.z"] = float(data[2])
                            action[f"{board_name}.qx"] = float(data[3])
                            action[f"{board_name}.qy"] = float(data[4])
                            action[f"{board_name}.qz"] = float(data[5])
                            action[f"{board_name}.qw"] = float(data[6])
                        else:
                            # Fill missing values with defaults
                            for i, suffix in enumerate(["x", "y", "z", "qx", "qy", "qz", "qw"]):
                                if i < len(data):
                                    action[f"{board_name}.{suffix}"] = float(data[i])
                                elif suffix == "qw":
                                    action[f"{board_name}.{suffix}"] = 1.0  # Default quaternion w
                                else:
                                    action[f"{board_name}.{suffix}"] = 0.0
                    
                    elif board_name == "fingers":
                        # Map finger data to nested finger format (10 fingers × 3 values each)
                        # Assuming data is a 10x3 matrix or flattened 30-element array
                        finger_names = ["thumb", "index", "middle", "ring", "little"]
                        sides = ["left", "right"]
                        
                        if isinstance(data, np.ndarray) and data.shape == (10, 3):
                            # Handle 10x3 matrix format
                            finger_idx = 0
                            for side in sides:
                                for finger_name in finger_names:
                                    if finger_idx < data.shape[0]:
                                        action[f"fingers.{side}.{finger_name}.x"] = float(data[finger_idx, 0])
                                        action[f"fingers.{side}.{finger_name}.y"] = float(data[finger_idx, 1])
                                        action[f"fingers.{side}.{finger_name}.z"] = float(data[finger_idx, 2])
                                        finger_idx += 1
                                    else:
                                        # Fill missing fingers with zeros
                                        action[f"fingers.{side}.{finger_name}.x"] = 0.0
                                        action[f"fingers.{side}.{finger_name}.y"] = 0.0
                                        action[f"fingers.{side}.{finger_name}.z"] = 0.0
                        elif isinstance(data, (list, np.ndarray)) and len(data) >= 30:
                            # Handle flattened 30-element array
                            finger_idx = 0
                            for side in sides:
                                for finger_name in finger_names:
                                    base_idx = finger_idx * 3
                                    if base_idx + 2 < len(data):
                                        action[f"fingers.{side}.{finger_name}.x"] = float(data[base_idx])
                                        action[f"fingers.{side}.{finger_name}.y"] = float(data[base_idx + 1])
                                        action[f"fingers.{side}.{finger_name}.z"] = float(data[base_idx + 2])
                                    else:
                                        action[f"fingers.{side}.{finger_name}.x"] = 0.0
                                        action[f"fingers.{side}.{finger_name}.y"] = 0.0
                                        action[f"fingers.{side}.{finger_name}.z"] = 0.0
                                    finger_idx += 1
                        else:
                            # Fill all fingers with zeros if data format is unexpected
                            for side in sides:
                                for finger_name in finger_names:
                                    action[f"fingers.{side}.{finger_name}.x"] = 0.0
                                    action[f"fingers.{side}.{finger_name}.y"] = 0.0
                                    action[f"fingers.{side}.{finger_name}.z"] = 0.0
            
            return action
        
        # Return empty action dict with default values if no data
        return {key: 1.0 if key.endswith('.qw') else 0.0 for key in self.action_features.keys()}

    @property
    def action_features(self):
        """
        Return action features matching ergoCub format for compatibility.
        Uses flattened dot-notation keys to match ergoCub.action_features.
        """
        # Define nested structure that mirrors ergoCub action hierarchy
        nested_actions = {
            "neck": {
                "qx": float, "qy": float, "qz": float, "qw": float,
            },
            "left_arm": {
                "x": float, "y": float, "z": float,
                "qx": float, "qy": float, "qz": float, "qw": float,
            },
            "right_arm": {
                "x": float, "y": float, "z": float,
                "qx": float, "qy": float, "qz": float, "qw": float,
            },
            "fingers": {
                "left": {
                    "thumb": {"x": float, "y": float, "z": float},
                    "index": {"x": float, "y": float, "z": float},
                    "middle": {"x": float, "y": float, "z": float},
                    "ring": {"x": float, "y": float, "z": float},
                    "little": {"x": float, "y": float, "z": float},
                },
                "right": {
                    "thumb": {"x": float, "y": float, "z": float},
                    "index": {"x": float, "y": float, "z": float},
                    "middle": {"x": float, "y": float, "z": float},
                    "ring": {"x": float, "y": float, "z": float},
                    "little": {"x": float, "y": float, "z": float},
                },
            },
        }
        
        return self._flatten_nested_dict(nested_actions)
    
    def _flatten_nested_dict(self, nested_dict: dict, prefix: str = "") -> dict[str, type]:
        """Flatten a nested dictionary into dot-separated keys."""
        flattened = {}
        for key, value in nested_dict.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flattened.update(self._flatten_nested_dict(value, new_key))
            else:
                flattened[new_key] = value
        return flattened

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
