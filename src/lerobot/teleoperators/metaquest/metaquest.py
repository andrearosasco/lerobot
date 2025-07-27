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

import uuid
from typing import Any

import numpy as np
import yarp
from lerobot.errors import DeviceNotConnectedError
from lerobot.teleoperators.teleoperator import Teleoperator

from .configuration_metaquest import MetaQuestConfig


class ActionInterface:
    """Simple YARP action interface for reading teleoperator commands."""
    
    def __init__(self, remote_prefix: str, local_prefix: str, control_boards: list[str], stream_name: str):
        self.remote_prefix = remote_prefix
        self.local_prefix = local_prefix
        self.control_boards = control_boards
        self.stream_name = stream_name
        
        # Initialize YARP network
        yarp.Network.init()
        
        # Create ports for each control board
        self.ports = {}
        for board in control_boards:
            self.ports[board] = yarp.Port()
        
    def connect(self):
        """Connect to YARP action streams."""
        for board in self.control_boards:
            local_name = f"{self.local_prefix}/{self.stream_name}/{board}:i"
            remote_name = f"{self.remote_prefix}/{board}/cmd:o"
            self.ports[board].open(local_name)
            yarp.Network.connect(remote_name, local_name)
    
    def read(self):
        """Read action data from YARP streams."""
        data = {}
        
        for board in self.control_boards:
            # Read action commands from YARP
            bottle = yarp.Bottle()
            if self.ports[board].read(bottle):
                # Convert YARP bottle to numpy array
                if board == "fingers":
                    # Special handling for fingers (10x3 matrix)
                    values = np.zeros((10, 3))
                    for i in range(min(30, bottle.size())):
                        row = i // 3
                        col = i % 3
                        values[row, col] = bottle.get(i).asFloat64()
                    data[board] = values
                else:
                    # Regular joint array
                    values = []
                    for i in range(bottle.size()):
                        values.append(bottle.get(i).asFloat64())
                    data[board] = np.array(values)
        
        # Return data in a format similar to polars DataFrame
        return [{"data": data}]
    
    def close(self):
        """Close YARP connections."""
        for port in self.ports.values():
            port.close()


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

        yarp.Network.init()

        self.action_interface = ActionInterface(
            remote_prefix=cfg.remote_prefix,
            local_prefix=f"{cfg.local_prefix}/{self.session_id}",
            control_boards=cfg.control_boards,
            stream_name="actions",
        )

    def connect(self, calibrate: bool = True):
        self.action_interface.connect()
        self._is_connected = True

    def disconnect(self):
        self.action_interface.close()
        yarp.Network.fini()
        self._is_connected = False

    def get_action(self) -> dict[str, Any]:
        """
        Get action from MetaQuest teleoperator in ergoCub-compatible format.
        Returns flattened dictionary matching ergoCub.action_features.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        for data_row in self.action_interface.read():
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
