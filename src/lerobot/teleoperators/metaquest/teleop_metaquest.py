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
        for data_row in self.action_interface.read():
            action = {}
            for board_name, board_data in data_row["data"].items():
                action[f"action/{board_name}"] = board_data
            return action
        return {}

    @property
    def action_features(self):
        # From metacub_dashboard/interfaces/interfaces.py:ActionInterface.DEFAULT_BOARD_FORMATS
        # This is a bit tricky because the format is nested.
        # For simplicity, we'll flatten it and assume float32.
        # TODO: This might need to be adjusted based on how the actions are actually used.
        features = {
            "action/neck": {"shape": (9,), "dtype": "float32", "space": "joint_positions"},
            "action/left_arm": {"shape": (7,), "dtype": "float32", "space": "joint_positions"},
            "action/right_arm": {"shape": (7,), "dtype": "float32", "space": "joint_positions"},
            "action/fingers": {"shape": (10, 3), "dtype": "float32", "space": "joint_positions"},
        }
        return {k: v for k, v in features.items() if k.split("/")[1] in self.cfg.control_boards}

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
