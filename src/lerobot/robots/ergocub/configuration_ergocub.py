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

from dataclasses import dataclass, field
from typing import List
import yaml

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
    # Enable/disable specific body parts
    control_boards: List[str] = field(
        default_factory=lambda: ["head", "bimanual", "fingers"]
    )
    state_boards: List[str] = field(
        default_factory=lambda: ["head", "bimanual", "fingers"]
    )
    left_hand: bool = True
    right_hand: bool = True
    # Control mode: if True, actions are absolute targets (default, preserves behavior).
    # If False, actions are interpreted as deltas relative to the current/last target.
    absolute: bool = True
    finger_scale: float = 1.0
    position_tolerance: float = 0.1  # Tolerance for safety checks, in radians or meters depending on the joint
    yarp_robot_name: str = "ergoCubSN002"
    emotion_remote_port_name: str = "/ergoCubEmotions/rpc"

    @classmethod
    def from_yaml(cls, path: str) -> "ErgoCubConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Parsing cameras
        # cameras = {}
        # for cam_name, cam_data in data.get("cameras", {}).items():
        #     cameras[cam_name] = CameraConfig(
        #         resolution=cam_data.get("resolution", [640, 480]),
        #         fps=cam_data.get("fps", 30),
        #         use_depth=cam_data.get("use_depth", False),
        #     )

        return ErgoCubConfig(
            name=data.get("name", "ergocub"),
            remote_prefix=data.get("remote_prefix", "/ergocubSim"),
            local_prefix=data.get("local_prefix", "/ergocub_dashboard"),
            # cameras=data.get("cameras", {}),
            control_boards=data.get(
                "control_boards", ["head", "bimanual", "fingers"]
            ),
            state_boards=data.get(
                "state_boards", ["head", "bimanual", "fingers"]
            ),
            left_hand=data.get("left_hand", True),
            right_hand=data.get("right_hand", True),
            absolute=data.get("absolute", True),
            finger_scale=data.get("finger_scale", 1.0),
            position_tolerance=data.get("position_tolerance", 0.1),
            yarp_robot_name=data.get("yarp_robot_name", "ergoCubSN002"),
        )


# if __name__ == "__main__":
#     # Percorso del file YAML di esempio
#     yaml_path = "config/r1_conf.yaml"

#     # Caricare la configurazione dal file YAML
#     config = ErgoCubConfig.from_yaml(yaml_path)

#     # Stampare la configurazione caricata per verificarne il contenuto
#     print("Configurazione caricata:")
#     print(config)
