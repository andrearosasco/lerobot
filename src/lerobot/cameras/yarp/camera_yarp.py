# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from typing import Any

import cv2
import numpy as np
import yarp
from lerobot.cameras.camera import Camera
from lerobot.cameras.utils import get_cv2_rotation

from .configuration_yarp import YarpCameraConfig


class CameraInterface:
    """Simple YARP camera interface for reading RGB and depth data."""
    
    def __init__(self, remote_prefix: str, local_prefix: str, rgb_shape: tuple | None, 
                 depth_shape: tuple | None, stream_name: str):
        self.remote_prefix = remote_prefix
        self.local_prefix = local_prefix
        self.rgb_shape = rgb_shape
        self.depth_shape = depth_shape
        self.stream_name = stream_name
        
        # Initialize YARP network
        yarp.Network.init()
        
        # Create ports
        self.rgb_port = yarp.Port() if rgb_shape else None
        self.depth_port = yarp.Port() if depth_shape else None
        
    def connect(self):
        """Connect to YARP camera streams."""
        if self.rgb_port:
            rgb_local_name = f"{self.local_prefix}/{self.stream_name}/rgb:i"
            rgb_remote_name = f"{self.remote_prefix}/{self.stream_name}/rgb:o"
            self.rgb_port.open(rgb_local_name)
            yarp.Network.connect(rgb_remote_name, rgb_local_name)
            
        if self.depth_port:
            depth_local_name = f"{self.local_prefix}/{self.stream_name}/depth:i"
            depth_remote_name = f"{self.remote_prefix}/{self.stream_name}/depth:o"
            self.depth_port.open(depth_local_name)
            yarp.Network.connect(depth_remote_name, depth_local_name)
    
    def read(self):
        """Read data from YARP camera streams."""
        data = {}
        
        if self.rgb_port:
            # Read RGB image
            img = yarp.ImageRgb()
            if self.rgb_port.read(img):
                # Convert YARP image to numpy array
                h, w = img.height(), img.width()
                rgb_array = np.frombuffer(img.getRawImage(), dtype=np.uint8)
                rgb_array = rgb_array.reshape((h, w, 3))
                data["rgb"] = rgb_array
                
        if self.depth_port:
            # Read depth image
            img = yarp.ImageFloat()
            if self.depth_port.read(img):
                # Convert YARP depth image to numpy array
                h, w = img.height(), img.width()
                depth_array = np.frombuffer(img.getRawImage(), dtype=np.float32)
                depth_array = depth_array.reshape((h, w))
                data["depth"] = depth_array
        
        # Return data in a format similar to polars DataFrame
        return [{"data": data}]
    
    def close(self):
        """Close YARP connections."""
        if self.rgb_port:
            self.rgb_port.close()
        if self.depth_port:
            self.depth_port.close()


class YarpCamera(Camera):
    """YARP-based camera implementation following LeRobot standards.
    
    This camera connects to YARP camera streams and provides RGB and optional depth data
    in the standard LeRobot camera interface format.
    """

    def __init__(self, config: YarpCameraConfig, remote_prefix: str, local_prefix: str):
        super().__init__(config)
        self.config = config
        self.remote_prefix = remote_prefix
        self.local_prefix = local_prefix
        self._is_connected = False
        
        # Create YARP camera interface
        self.interface = CameraInterface(
            remote_prefix=remote_prefix,
            local_prefix=local_prefix,
            rgb_shape=(config.width, config.height) if config.use_depth or True else None,
            depth_shape=(config.width, config.height) if config.use_depth else None,
            stream_name=config.yarp_name,
        )

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, warmup: bool = True) -> None:
        if self._is_connected:
            raise RuntimeError(f"YarpCamera({self.config.yarp_name}) is already connected")

        self.interface.connect()
        self._is_connected = True

        if warmup:
            # Warmup by reading a few frames
            for _ in range(max(1, int(self.config.warmup_s * self.fps))):
                try:
                    self.async_read()
                    time.sleep(1.0 / self.fps)
                except Exception:
                    # Ignore warmup errors
                    pass

    def read(self, temporary_buffer: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Read frame(s) from the camera.
        
        Returns:
            Dictionary with 'image' key and optionally 'depth' key containing numpy arrays
        """
        if not self._is_connected:
            raise RuntimeError(f"YarpCamera({self.config.yarp_name}) is not connected")

        # Read from YARP interface
        df = self.interface.read()
        
        result = {}
        
        for row in df:
            data = row["data"]
            
            if "rgb" in data:
                image = data["rgb"]
                
                # Apply color mode conversion if needed
                if self.config.color_mode.value == "bgr":
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Apply rotation if needed
                rotation = get_cv2_rotation(self.config.rotation)
                if rotation is not None:
                    image = cv2.rotate(image, rotation)
                
                result["image"] = image
            
            if "depth" in data and self.config.use_depth:
                depth = data["depth"]
                
                # Apply rotation to depth if needed  
                rotation = get_cv2_rotation(self.config.rotation)
                if rotation is not None:
                    depth = cv2.rotate(depth, rotation)
                
                result["depth"] = depth
        
        return result

    def async_read(self) -> dict[str, np.ndarray]:
        """Async read (same as sync read for YARP)."""
        return self.read()

    def disconnect(self) -> None:
        if not self._is_connected:
            raise RuntimeError(f"YarpCamera({self.config.yarp_name}) is not connected")

        self.interface.close()
        self._is_connected = False

    @classmethod
    def find_cameras(cls) -> list[int]:
        """Find available YARP cameras.
        
        Note: This is a placeholder implementation as YARP camera discovery
        is typically handled through YARP network configuration.
        """
        # TODO: Implement actual YARP camera discovery if needed
        # For now, return empty list as YARP cameras are configured manually
        return []
