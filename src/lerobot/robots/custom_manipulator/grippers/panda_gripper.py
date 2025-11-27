import time
import numpy as np
import rclpy
from rclpy.node import Node
from panda_interface.srv import ApplyCommandsGripper, ConnectGripper, GetSensorsGripper
from panda_interface.msg import PandaGripperCommand
from dataclasses import dataclass
from ..configs import GripperConfig

@GripperConfig.register_subclass("panda_gripper")
@dataclass
class PandaGripperConfig(GripperConfig):
    max_width: float = 1.0
    force: float = 20.0
    speed: float = 0.5

    @property
    def type(self) -> str:
        return "panda_gripper"

class PandaGripper(Node):
    interfaces = {
        'apply_commands_gripper': ApplyCommandsGripper,
        'get_sensors_gripper': GetSensorsGripper,
        'connect_gripper': ConnectGripper,
    }

    def __init__(self, config: PandaGripperConfig = None, **kwargs):
        super().__init__('panda_gripper_client_lerobot')
        self.config = config if config else PandaGripperConfig()
        
        self.client_names = {}
        for name, type in PandaGripper.interfaces.items():
            client = self.create_client(type, name)
            self.client_names[name] = client

    def connect(self):
        for name, client in self.client_names.items():
             if not client.wait_for_service(timeout_sec=1.0):
                 self.get_logger().warn(f'service {name} not available')

        request = PandaGripper.interfaces['connect_gripper'].Request()
        future = self.client_names['connect_gripper'].call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result().success

    def apply_commands(self, gripper_state: float, speed: float = None, force: float = None):
        request = PandaGripper.interfaces['apply_commands_gripper'].Request()
        request.command = PandaGripperCommand(width=float(gripper_state == 0.0))
        
        future = self.client_names['apply_commands_gripper'].call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result().success

    def get_sensors(self):
        request = PandaGripper.interfaces['get_sensors_gripper'].Request()
        future = self.client_names['get_sensors_gripper'].call_async(request)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        return {'gripper': res.state.width}

    def close(self):
        pass

    def reset(self):
        self.apply_commands(gripper_state=1.0)
        time.sleep(1)
        self.apply_commands(gripper_state=0.0)

    @property
    def features(self) -> dict:
        return {
            "gripper": float,
        }
