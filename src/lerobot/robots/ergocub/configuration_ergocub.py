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
