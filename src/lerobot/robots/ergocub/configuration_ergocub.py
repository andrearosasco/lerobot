from dataclasses import dataclass, field
from typing import List

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("ergocub")
@dataclass
class ErgoCubConfig(RobotConfig):
    name: str = "ergocub"
    # YARP remote prefix for observation ports
    remote_prefix: str = "/ergocubSim"
    # YARP local prefix for observation ports. A session ID will be appended.
    local_prefix: str = "/ergocub_dashboard"
    # Configuration for cameras (using standard LeRobot camera interface)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # List of control boards for encoders
    encoders_control_boards: List[str] = field(
        default_factory=lambda: ["head", "left_arm", "right_arm", "torso"]
    )
