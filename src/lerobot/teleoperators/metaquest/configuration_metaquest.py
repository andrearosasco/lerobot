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


@TeleoperatorConfig.register_subclass("metaquest")
@dataclass
class MetaQuestConfig(TeleoperatorConfig):
    name: str = "metaquest"
    # YARP remote prefix for the action server
    remote_prefix: str = "/metaControllClient"
    # YARP local prefix for the action client. A session ID will be appended.
    local_prefix: str = "/metaquest_dashboard"
    # List of control boards for actions
    control_boards: List[str] = field(
        default_factory=lambda: ["neck", "left_arm", "right_arm", "fingers"]
    )
