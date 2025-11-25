import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.task import Future
from robotiq_85_msgs.msg import GripperStat, GripperCmd
from dataclasses import dataclass
import draccus
from ..configs import GripperConfig
from lerobot.configs.types import FeatureType, PolicyFeature

@GripperConfig.register_subclass("robotiq")
@dataclass
class RobotiqConfig(GripperConfig):
    max_width: float = 0.85
    force: float = 5.0
    speed: float = 0.0565

    @property
    def type(self) -> str:
        return "robotiq"

class Robotiq(Node):
    def __init__(self, config: RobotiqConfig = None, **kwargs):
        super().__init__('robotiq_action_client')
        self.config = config if config else RobotiqConfig()
        self.create_subscription(GripperStat, "/gripper/stat", self._state_topic_callback, 1)
        self.gripper_pub = self.create_publisher(GripperCmd, '/gripper/cmd', 1)
        self.gripper_state = None

    def apply_commands(self, width:float, speed:float=None, force:float=None):
        cmd_msg = GripperCmd()
        cmd_msg.position = width
        cmd_msg.force = force if force is not None else self.config.force
        cmd_msg.speed = speed if speed is not None else self.config.speed
        self.gripper_pub.publish(cmd_msg)

    def get_sensors(self):
        self.gripper_state = Future()
        # This spins until a message is received on the topic
        rclpy.spin_until_future_complete(self, self.gripper_state)
        return {'grip_joint_pos': np.array([self.gripper_state.result().position])}

    def connect(self):
        pass

    def close(self):
        self.reset()

    def reset(self, width=0.1, **kwargs):
        self.apply_commands(width=0.0)
        time.sleep(2)
        self.apply_commands(width=1.0)

    def _state_topic_callback(self, msg):
        if self.gripper_state is not None and not self.gripper_state.done():
            self.gripper_state.set_result(msg)
            self.gripper_state.done()

    @property
    def features(self) -> dict:
        return {
            "gripper.pos": float,
        }
