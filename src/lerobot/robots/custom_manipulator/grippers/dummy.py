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

    def connect(self):
        pass

    def disconnect(self):
        pass

    def close(self):
        pass

    def reset(self):
        pass

    def apply_commands(self, width: float, speed: float = None, force: float = None):
        pass

    @property
    def features(self) -> dict:
        return {
            "gripper.pos": float,
        }

    def get_sensors(self):
        return {'grip_joint_pos': np.array([0.0])}
