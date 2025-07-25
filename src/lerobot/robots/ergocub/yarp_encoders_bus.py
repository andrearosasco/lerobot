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
from typing import Any

import numpy as np
import yarp

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

logger = logging.getLogger(__name__)


class YarpEncodersBus:
    """
    YARP encoders interface for reading joint positions from ergoCub robot.
    
    This class provides a standardized interface for reading encoder data from
    YARP streams, following the LeRobot patterns used by motor buses.
    """

    def __init__(
        self,
        remote_prefix: str,
        local_prefix: str,
        control_boards: list[str],
        stream_name: str = "encoders",
    ):
        """
        Initialize YARP encoders interface.

        Args:
            remote_prefix: Remote YARP prefix (e.g., "/ergocub")
            local_prefix: Local YARP prefix (e.g., "/lerobot/session_id")
            control_boards: List of control board names to read from
            stream_name: Name of the stream (default: "encoders")
        """
        self.remote_prefix = remote_prefix
        self.local_prefix = local_prefix
        self.control_boards = control_boards
        self.stream_name = stream_name
        self._is_connected = False
        
        # Create ports for each control board
        self.ports = {}
        for board in control_boards:
            self.ports[board] = yarp.Port()

    @property
    def is_connected(self) -> bool:
        """Whether the encoders bus is currently connected."""
        return self._is_connected

    def connect(self) -> None:
        """
        Connect to YARP encoder streams.
        
        Raises:
            DeviceAlreadyConnectedError: If already connected
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError("YarpEncodersBus already connected")

        logger.info(f"Connecting to YARP encoders with prefix {self.remote_prefix}")
        
        # Initialize YARP network if not already done
        if not yarp.Network.checkNetwork():
            yarp.Network.init()
        
        success_count = 0
        for board in self.control_boards:
            local_name = f"{self.local_prefix}/{self.stream_name}/{board}:i"
            remote_name = f"{self.remote_prefix}/{board}/state:o"
            
            # Open local port
            if not self.ports[board].open(local_name):
                logger.error(f"Failed to open local port {local_name}")
                continue
                
            # Connect to remote port
            if yarp.Network.connect(remote_name, local_name):
                logger.debug(f"Connected {remote_name} -> {local_name}")
                success_count += 1
            else:
                logger.warning(f"Failed to connect {remote_name} -> {local_name}")
        
        if success_count == len(self.control_boards):
            self._is_connected = True
            logger.info(f"Successfully connected to all {len(self.control_boards)} encoder streams")
        else:
            logger.warning(f"Connected to {success_count}/{len(self.control_boards)} encoder streams")
            self._is_connected = True  # Allow partial connections

    def disconnect(self) -> None:
        """
        Disconnect from YARP encoder streams.
        
        Raises:
            DeviceNotConnectedError: If not connected
        """
        if not self.is_connected:
            raise DeviceNotConnectedError("YarpEncodersBus not connected")

        logger.info("Disconnecting YARP encoders")
        
        for board, port in self.ports.items():
            if port.isOpen():
                port.close()
                logger.debug(f"Closed port for {board}")
        
        self._is_connected = False
        logger.info("YARP encoders disconnected")

    def sync_read(self, data_name: str = "Present_Position") -> dict[str, np.ndarray]:
        """
        Read encoder data from all control boards.
        
        Args:
            data_name: Name of the data to read (for compatibility with motor buses)
            
        Returns:
            Dictionary mapping board names to joint position arrays
            
        Raises:
            DeviceNotConnectedError: If not connected
        """
        if not self.is_connected:
            raise DeviceNotConnectedError("YarpEncodersBus not connected")

        encoder_data = {}
        
        for board in self.control_boards:
            bottle = yarp.Bottle()
            
            # Non-blocking read
            if self.ports[board].read(bottle, shouldWait=False):
                # Convert YARP bottle to numpy array
                values = []
                for i in range(bottle.size()):
                    values.append(bottle.get(i).asFloat64())
                encoder_data[board] = np.array(values, dtype=np.float32)
            else:
                # Return zeros if no data available (for compatibility)
                # You might want to use previous values or handle this differently
                logger.debug(f"No data available for board {board}")
                encoder_data[board] = np.array([], dtype=np.float32)
        
        return encoder_data

    def read(self) -> list[dict[str, Any]]:
        """
        Read encoder data in the original format for backward compatibility.
        
        Returns:
            List containing a single dictionary with the encoder data
        """
        encoder_data = self.sync_read()
        
        # Convert to the original format
        formatted_data = {}
        for board, values in encoder_data.items():
            formatted_data[board] = {"values": values}
        
        return [{"data": formatted_data}]

    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, '_is_connected') and self._is_connected:
            try:
                self.disconnect()
            except:
                pass  # Ignore errors during cleanup
