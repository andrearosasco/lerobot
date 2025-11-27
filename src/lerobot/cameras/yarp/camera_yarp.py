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
        
        # Port objects will be created in connect() method
        self.rgb_port = None
        self.depth_port = None
        self.yarp_rgb_image = None
        self.yarp_depth_image = None
        self.rgb_buffer = None
        self.depth_buffer = None
        
    def connect(self):
        """Connect to YARP camera streams."""
        # Create and open RGB port if needed
        if self.rgb_shape:
            self.rgb_port = yarp.BufferedPortImageRgb()
            self.rgb_port.open(f"{self.local_prefix}/{self.stream_name}")
            
            self.yarp_rgb_image = yarp.ImageRgb()
            self.yarp_rgb_image.resize(self.rgb_shape[0], self.rgb_shape[1])
            self.rgb_buffer = bytearray(self.rgb_shape[0] * self.rgb_shape[1] * 3)
            self.yarp_rgb_image.setExternal(self.rgb_buffer, self.rgb_shape[0], self.rgb_shape[1])
            
            rgb_remote_name = f"{self.remote_prefix}/{self.stream_name}"
            rgb_local_name = f"{self.local_prefix}/{self.stream_name}"
            
            while not yarp.Network.connect(rgb_remote_name, rgb_local_name, "mjpeg"):
                print(f"Waiting for RGB port {rgb_remote_name} to connect...")
                time.sleep(1.0)
            
        # Create and open depth port if needed
        if self.depth_shape:
            self.depth_port = yarp.BufferedPortImageFloat()
            self.depth_port.open(f"{self.local_prefix}/{self.stream_name}/depthImage:i")
            
            self.yarp_depth_image = yarp.ImageFloat()
            self.yarp_depth_image.resize(self.depth_shape[0], self.depth_shape[1])
            self.depth_buffer = bytearray(self.depth_shape[0] * self.depth_shape[1] * 4)
            self.yarp_depth_image.setExternal(self.depth_buffer, self.depth_shape[0], self.depth_shape[1])
            
            depth_remote_name = f"{self.remote_prefix}/{self.stream_name}/depthImage:o"
            depth_local_name = f"{self.local_prefix}/{self.stream_name}/depthImage:i"
            
            while not yarp.Network.connect(
                depth_remote_name, depth_local_name,
                "fast_tcp+send.portmonitor+file.bottle_compression_zlib+recv.portmonitor+file.bottle_compression_zlib+type.dll"
            ):
                print(f"Waiting for Depth port {depth_remote_name} to connect...")
                time.sleep(0.1)
    
    def read(self):
        """Read data from YARP camera streams."""
        data = {}
        
        if self.rgb_shape and self.rgb_port:
            # Read RGB image with non-blocking read and wait loop
            read_attempts = 0
            while (image_data := self.rgb_port.read(False)) is None:
                read_attempts += 1
                # Print warning every 1000 attempts (approximately every few seconds)
                if read_attempts % 1000 == 0:
                    print(f"Still waiting for RGB data from {self.stream_name} (attempt {read_attempts})")
                # Small sleep to avoid busy waiting
                time.sleep(0.001)  # 1 millisecond sleep
                    
            # Copy data to pre-allocated image and extract as numpy array
            self.yarp_rgb_image.copy(image_data)
            rgb_array = np.frombuffer(self.rgb_buffer, dtype=np.uint8).reshape(
                self.rgb_shape[1], self.rgb_shape[0], 3
            ).copy()  # Create a copy to avoid buffer reuse issues
            data["rgb"] = rgb_array
                
        if self.depth_shape and self.depth_port:
            # Read depth image with non-blocking read and wait loop
            read_attempts = 0
            while (image_data := self.depth_port.read(False)) is None:
                read_attempts += 1
                # Print warning every 1000 attempts (approximately every few seconds)
                if read_attempts % 1000 == 0:
                    print(f"Still waiting for depth data from {self.stream_name} (attempt {read_attempts})")
                # Small sleep to avoid busy waiting
                time.sleep(0.001)  # 1 millisecond sleep
                    
            # Copy data to pre-allocated image and extract as numpy array
            self.yarp_depth_image.copy(image_data)
            depth_array = (
                np.frombuffer(self.depth_buffer, dtype=np.float32).reshape(
                    self.depth_shape[1], self.depth_shape[0]
                ).copy() * 1000  # Create a copy and convert to mm
            ).astype(np.uint16)
            data["depth"] = depth_array
        
        # Return data in a format similar to polars DataFrame
        return [{"data": data}]
    
    def close(self):
        """Close YARP connections."""
        if hasattr(self, 'rgb_port') and self.rgb_port:
            self.rgb_port.close()
        if hasattr(self, 'depth_port') and self.depth_port:
            self.depth_port.close()


class YarpCamera(Camera):
    """YARP-based camera implementation following LeRobot standards.
    
    This camera connects to YARP camera streams and provides RGB and optional depth data
    in the standard LeRobot camera interface format.
    """

    def __init__(self, config: YarpCameraConfig):
        super().__init__(config)
        self.config = config

        self._is_connected = False
        
        # Create YARP camera interface
        self.interface = CameraInterface(
            remote_prefix=config.remote_prefix,
            local_prefix=config.local_prefix,
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

                if (image==0).all():
                    raise Exception("The image is all black!")
                
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
