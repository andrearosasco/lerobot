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

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("metaquest_rail")
@dataclass
class MetaQuestRailConfig(TeleoperatorConfig):
    name: str = "metaquest_rail"
    
    # Connection timeout and retry settings
    connection_timeout: float = 10.0  # seconds to wait for connection
    retry_attempts: int = 3  # number of connection attempts
    
    # List of control boards for actions (kept for compatibility)
    control_boards: List[str] = field(
        default_factory=lambda: ["right_hand", "gripper"]
    )
