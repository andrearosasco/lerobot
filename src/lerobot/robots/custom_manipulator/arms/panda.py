import time
import numpy as np
import rclpy
import roboticstoolbox as rtb
from scipy.spatial.transform import Rotation as R
from spatialmath import SE3
from rclpy.node import Node
from panda_interface.srv import ApplyCommands, Connect, GetSensors, Close
from panda_interface.msg import PandaCommand
from dataclasses import dataclass
import draccus
from ..configs import ArmConfig
from lerobot.configs.types import FeatureType, PolicyFeature

@ArmConfig.register_subclass("panda")
@dataclass
class PandaConfig(ArmConfig):
    # Add any configuration parameters here if needed
    @property
    def type(self) -> str:
        return "panda"

def _min_jerk_spaces(N: int, T: float):
    """
    Generates a 1-dim minimum jerk trajectory from 0 to 1 in N steps & T seconds.
    Assumes zero velocity & acceleration at start & goal.
    """
    assert N > 1, "Number of planning steps must be larger than 1."

    t_traj = np.linspace(0, 1, N)
    p_traj = 10 * t_traj**3 - 15 * t_traj**4 + 6 * t_traj**5
    pd_traj = (30 * t_traj**2 - 60 * t_traj**3 + 30 * t_traj**4) / T
    pdd_traj = (60 * t_traj - 180 * t_traj**2 + 120 * t_traj**3) / (T**2)

    return p_traj, pd_traj, pdd_traj

def generate_joint_space_min_jerk(start, goal, time_to_go: float, dt: float):
    """
    Primitive joint space minimum jerk trajectory planner.
    """
    steps =  int(time_to_go/dt)

    p_traj, pd_traj, pdd_traj = _min_jerk_spaces(steps, time_to_go)

    D = goal - start
    q_traj = start[None, :] + D[None, :] * p_traj[:, None]
    qd_traj = D[None, :] * pd_traj[:, None]
    qdd_traj = D[None, :] * pdd_traj[:, None]

    waypoints = [
        {
            "time_from_start": i * dt,
            "position": q_traj[i, :],
            "velocity": qd_traj[i, :],
            "acceleration": qdd_traj[i, :],
        }
        for i in range(steps)
    ]

    return waypoints

class Panda(Node):
    interfaces = {
        'apply_commands': ApplyCommands,
        'get_sensors': GetSensors,
        'connect': Connect,
        'close': Close
    }

    def __init__(self, config: PandaConfig = None, **kwargs):
        super().__init__('panda_client')
        self.config = config if config else PandaConfig()
        self.robot_model = rtb.models.Panda()

        self.client_names = {}
        for name, type in Panda.interfaces.items():
            client = self.create_client(type, name)
            self.client_names[name] = client

    def connect(self):
        for name, client in self.client_names.items():
            while not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'service {name} not available, waiting again...')

        request = Panda.interfaces['connect'].Request()
        self.future = self.client_names['connect'].call_async(request)

        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

    def apply_commands(self, action=None, q_desired=None, kp=None, kd=None, gain=4.):
        if action is not None:
            # Extract action components
            eef_pos = np.array([
                action["position.x"],
                action["position.y"],
                action["position.z"]
            ])
            
            # Orientation is axis-angle
            axis_angle = np.array([
                action["orientation.x"],
                action["orientation.y"],
                action["orientation.z"]
            ])
            
            # Convert axis-angle to rotation matrix
            eef_rot = R.from_rotvec(axis_angle).as_matrix()
            
            # Get current joint positions for IK seed
            request = Panda.interfaces['get_sensors'].Request()
            self.future = self.client_names['get_sensors'].call_async(request)
            rclpy.spin_until_future_complete(self, self.future)
            state = self.future.result().state
            qpos = np.array(state.position)
            
            # IK
            q_desired = self.compute_ik(eef_pos, eef_rot, q_seed=qpos)

        request = Panda.interfaces['apply_commands'].Request()
        # Ensure q_desired is a list or array
        if isinstance(q_desired, np.ndarray):
            q_desired = q_desired.tolist()
            
        request.command = PandaCommand(position=q_desired, gain=gain)
        self.future = self.client_names['apply_commands'].call_async(request)
        return

    def get_sensors(self):
        request = Panda.interfaces['get_sensors'].Request()

        self.future = self.client_names['get_sensors'].call_async(request)
        rclpy.spin_until_future_complete(self, self.future)

        state = self.future.result().state
        
        q = np.array(state.position)

        Te = self.robot_model.fkine(q)
        pos = Te.t
        rot_matrix = Te.R
        rot_axis_angle = R.from_matrix(rot_matrix).as_rotvec()

        pos = {f"position.{k}": v for k,v in zip(["x", "y", "z"], pos)}
        ori = {f"orientation.{k}": v for k,v in zip(["x", "y", "z"], rot_axis_angle)}

        return {**pos, **ori}

    def compute_ik(self, position, orientation, q_seed=None):
        Tep = SE3.Rt(R=orientation, t=position)
        sol = self.robot_model.ik_LM(Tep, q0=q_seed)
        return sol[0]

    def close(self):
        pass

    def reset(self):
        # Define home position
        home_pos = np.array([0.0, 0.0, 0.0, -2, 0.0, 2, 0.7854])
        
        # Get current position
        request = Panda.interfaces['get_sensors'].Request()
        self.future = self.client_names['get_sensors'].call_async(request)
        rclpy.spin_until_future_complete(self, self.future)
        state = self.future.result().state
        current_pos = np.array(state.position)
        
        # Generate trajectory
        dt = 0.1
        time_to_go = 5.0 # 4 seconds to move home
        waypoints = generate_joint_space_min_jerk(current_pos, home_pos, time_to_go, dt)
        
        # Execute
        for wp in waypoints:
            self.apply_commands(q_desired=wp['position'])
            time.sleep(dt)

    @property
    def features(self) -> dict:
        pos = {f"position.{d}": float for d in ["x", "y", "z"]}
        ori = {f"orientation.{d}": float for d in ["x", "y", "z"]}
        return {**pos, **ori}
