import numpy as np
from dataclasses import dataclass
import draccus
from ..configs import GripperConfig
from lerobot.configs.types import FeatureType, PolicyFeature

@GripperConfig.register_subclass("dummy")
@dataclass
class DummyGripperConfig(GripperConfig):
    @property
    def type(self) -> str:
        return "dummy"

class DummyGripper:
    def __init__(self, config: DummyGripperConfig = None):
        self.config = config if config else DummyGripperConfig()

    def get_end_effector_transform(self, arm_type: str) -> np.ndarray:
        identity = np.eye(4)
        return {"dummy": identity, "panda": identity}[arm_type].copy()

    def connect(self):
        pass

    def disconnect(self):
        pass

    def close(self):
        pass

    def reset(self):
        pass

    def apply_commands(self, action=None, speed: float = None, force: float = None):
        pass

    @property
    def action_features(self) -> dict:
        return {}

    @property
    def features(self) -> dict:
        return {}

    def get_sensors(self):
        return {'grip_joint_pos': np.array([0.0])}
