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

from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig, ColorMode, Cv2Rotation


@CameraConfig.register_subclass("yarp")
@dataclass
class YarpCameraConfig(CameraConfig):
    """Configuration class for YARP-based cameras.

    This class provides configuration options for cameras accessed through YARP,
    supporting both RGB and depth streams from robotic systems.

    Example configurations:
    ```python
    # Basic RGB camera
    YarpCameraConfig("head_camera", 640, 480, 30)

    # RGB + Depth camera  
    YarpCameraConfig("head_camera", 640, 480, 30, use_depth=True)
    ```

    Attributes:
        yarp_name: The YARP port name for the camera (e.g., "head_camera", "chest_camera")
        fps: Requested frames per second
        width: Frame width in pixels
        height: Frame height in pixels
        use_depth: Whether to enable depth stream. Defaults to False.
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        rotation: Image rotation setting. Defaults to no rotation.
        warmup_s: Time reading frames before returning from connect (in seconds)
        remote_prefix: Remote port prefix for YARP connections. Defaults to empty string.

    Note:
        - The yarp_name corresponds to the camera stream name in YARP
        - RGB stream is always enabled, depth is optional
        - Both streams use the same resolution and FPS
    """

    yarp_name: str
    use_depth: bool = False
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1
    remote_prefix: str = ""

    def __post_init__(self):
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` is expected to be {ColorMode.RGB.value} or {ColorMode.BGR.value}, but {self.color_mode} is provided."
            )

        if self.rotation not in (
            Cv2Rotation.NO_ROTATION,
            Cv2Rotation.ROTATE_90,
            Cv2Rotation.ROTATE_180,
            Cv2Rotation.ROTATE_270,
        ):
            raise ValueError(
                f"`rotation` is expected to be in {(Cv2Rotation.NO_ROTATION, Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_180, Cv2Rotation.ROTATE_270)}, but {self.rotation} is provided."
            )

        # Ensure required camera parameters are set
        if not all([self.fps, self.width, self.height]):
            raise ValueError("fps, width, and height must all be specified for YARP cameras")
