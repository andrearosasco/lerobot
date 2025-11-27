from lerobot.robots.custom_manipulator.grippers.panda_gripper import PandaGripperConfig
from .arms.panda import Panda, PandaConfig
from .grippers.robotiq import Robotiq, RobotiqConfig
from .grippers.dummy import DummyGripper, DummyGripperConfig

def make_arm_from_config(config):
    if isinstance(config, PandaConfig):
        return Panda(config)
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
    else:
        raise ValueError(f"Unknown gripper config type: {type(config)}")
