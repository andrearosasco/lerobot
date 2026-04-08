from dataclasses import dataclass, field

from ..configs import GripperConfig

TIPS = ("thumb", "index", "middle", "ring", "little")
TIP_ACTIONS = {f"action.{tip}.position.{axis}": float for tip in TIPS for axis in "xyz"}
DEFAULT_TIP_LINK_NAMES = {
    "thumb": "right_hand_thumb_rota_tip",
    "index": "right_hand_index_rota_tip",
    "middle": "right_hand_mid_tip",
    "ring": "right_hand_ring_tip",
    "little": "right_hand_pinky_tip",
}
DEFAULT_TIP_SCALE_FACTORS = {
    "thumb": 1.40,
    "index": 1.20,
    "middle": 1.12,
    "ring": 1.20,
    "little": 1.60,
}
COMMAND_INDEX_BY_DRIVER_NAME = {
    "right_hand_thumb_bend_link": 0,
    "right_hand_thumb_rota_link1": 1,
    "right_hand_thumb_rota_link2": 2,
    "right_hand_index_bend_link": 3,
    "right_hand_index_rota_link1": 4,
    "right_hand_index_rota_link2": 5,
    "right_hand_mid_link1": 6,
    "right_hand_mid_link2": 7,
    "right_hand_ring_link1": 8,
    "right_hand_ring_link2": 9,
    "right_hand_pinky_link1": 10,
    "right_hand_pinky_link2": 11,
}


@GripperConfig.register_subclass("xhand")
@dataclass
class XHandConfig(GripperConfig):
    hand_id: int = 0
    control_mode: int = 3
    use_delta_actions: bool = False
    read_tactile_sensors: bool = False
    kp: int = 50
    ki: int = 0
    kd: int = 0
    tor_max: int = 300
    poll_force_update: bool = True
    urdf_path: str | None = 'ergocub2-design-hand/robotera/xhand1/urdf/Xhand-urdf/xhand_right/urdf/xhand_right.urdf'
    palm_link_name: str = "right_hand_ee_link"
    niter: int = 20000
    tip_link_names: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_TIP_LINK_NAMES))
    tip_scale_factors: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TIP_SCALE_FACTORS))
    enable_tip_scale_tuner: bool = False
    visualize: bool = False

    @property
    def type(self) -> str:
        return "xhand"
