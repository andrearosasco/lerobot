from dataclasses import dataclass, field

from ..configs import GripperConfig

TIPS = ("thumb", "index", "middle", "ring", "little")
TIP_ACTIONS = {f"action.fingertip.{tip}.{axis}": float for tip in TIPS for axis in "xyz"}


@GripperConfig.register_subclass("xhand")
@dataclass
class XHandConfig(GripperConfig):
    protocol: str = "RS485"
    serial_port: str = "/dev/ttyUSB0"
    baud_rate: int = 3000000
    hand_id: int = 0
    control_mode: int = 3
    kp: int = 100
    ki: int = 0
    kd: int = 0
    tor_max: int = 300
    poll_force_update: bool = True
    startup_delay_s: float = 1.0
    urdf_path: str | None = None
    palm_link_name: str = "palm"
    niter: int = 20000
    tip_link_names: dict[str, str] = field(default_factory=lambda: {tip: f"{tip}_tip" for tip in TIPS})
    open_positions: list[float] = field(default_factory=lambda: [0.0] * 12)
    closed_positions: list[float] = field(default_factory=lambda: [0.8] * 12)

    @property
    def type(self) -> str:
        return "xhand"