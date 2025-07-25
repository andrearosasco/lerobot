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
