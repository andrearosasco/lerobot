from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("metareader")
@dataclass
class MetaReaderConfig(TeleoperatorConfig):
    name: str = "metareader"
    port: int = 5005
    tcp_port: int = 5005
    read_timeout_s: float = 0.02
    connection_timeout_s: float = 10.0
    auto_adb_reverse: bool = True
    no_advertise: bool = False
    require_tracked_right_hand: bool = True
    pinch_close_distance_m: float = 0.03
    pinch_open_distance_m: float = 0.09
