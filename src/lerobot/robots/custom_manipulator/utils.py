from lerobot.robots.custom_manipulator.grippers.panda_gripper import PandaGripperConfig
from .arms.dummy import DummyArm, DummyArmConfig
from .arms.panda import Panda, PandaConfig
from .grippers.config_xhand import XHandConfig
from .grippers.robotiq import Robotiq, RobotiqConfig
from .grippers.dummy_gripper import DummyGripper, DummyGripperConfig

import numpy as np

def make_arm_from_config(config):
    if isinstance(config, PandaConfig):
        return Panda(config)
    elif isinstance(config, DummyArmConfig) or config is None:
        return DummyArm(config)
    else:
        raise ValueError(f"Unknown arm config type: {type(config)}")

def make_gripper_from_config(config):
    if isinstance(config, RobotiqConfig):
        return Robotiq(config)
    elif isinstance(config, DummyGripperConfig) or config is None:
        return DummyGripper(config)
    elif isinstance(config, PandaGripperConfig):
        from .grippers.panda_gripper import PandaGripper
        return PandaGripper(config)
    elif isinstance(config, XHandConfig):
        try:
            from .grippers.xhand import XHand
        except ModuleNotFoundError as exc:
            if exc.name == "xhand_controller":
                raise ModuleNotFoundError(
                    "xHand support requires the optional `xhand_controller` package."
                ) from exc
            raise
        return XHand(config)
    else:
        raise ValueError(f"Unknown gripper config type: {type(config)}")

def rotation_6d_to_matrix(d6):
    a1, a2 = d6[:3], d6[3:]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=0)

def matrix_to_rotation_6d(matrix):
    return matrix[:2, :].reshape(6)
