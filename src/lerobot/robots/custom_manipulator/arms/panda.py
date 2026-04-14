import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy

os.environ.setdefault("OSQP_ALGEBRA_BACKEND", "builtin")
import pinocchio as pin
import draccus
from lerobot.configs.types import FeatureType, PolicyFeature
from panda_interface.msg import PandaCommand
from panda_interface.srv import ApplyCommands, Close, Connect, GetSensors
from pink import Configuration
from pink.solve_ik import solve_ik
from pink.tasks import FrameTask, PostureTask
from rclpy.node import Node
import xacro

if not hasattr(np, "disp"):
    np.disp = lambda message, device=None, linefeed=True: print(message, end="\n" if linefeed else "")
if not hasattr(np, "int"):
    np.int = int

from scipy.spatial.transform import Rotation as R

from ..configs import ArmConfig
from .panda_utils import PandaDebugTools

HOME_ROT = R.from_rotvec([np.pi, 0.0, 0.0])

@ArmConfig.register_subclass("panda")
@dataclass
class PandaConfig(ArmConfig):
    visualize: bool = False
    use_delta_actions: bool = False

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
        self.end_effector_transform = np.eye(4)
        self._pin_model = None
        self._pin_data = None
        self._ik_task = None
        self._posture_task = None
        self._wrist_frame_id = None
        self._joint_names = None
        self._q_nom = None
        self._debug = None
        self._debug_urdf_path = None

        base = Path(__file__).resolve().parent / "franka_description"
        self.urdf_path = str(base / "robots" / "panda.urdf")
        self._debug = PandaDebugTools(
            urdf_path=self.urdf_path,
            enable_rerun_visualization=self.config.visualize,
        )

        self.client_names = {}
        for name, type in Panda.interfaces.items():
            client = self.create_client(type, name)
            self.client_names[name] = client

    def set_end_effector_transform(self, transform):
        self.end_effector_transform = np.array(transform, dtype=float, copy=True)

    def _ensure_kinematics(self):
        if self._pin_model is not None:
            return

        base = Path(__file__).resolve().parent / "franka_description"
        urdf_path = base / "robots" / "panda.urdf"
        full = pin.buildModelFromUrdf(str(urdf_path))

        q0 = pin.neutral(full)
        if q0.shape[0] >= 2:
            q0[-2:] = 0.04
        jids = []
        for joint_name in ("panda_finger_joint1", "panda_finger_joint2"):
            joint_id = full.getJointId(joint_name)
            if joint_id > 0 and joint_id not in jids:
                jids.append(joint_id)
        self._pin_model = pin.buildReducedModel(full, jids, q0) if jids else full
        self._pin_data = self._pin_model.createData()
        self._joint_names = tuple(self._pin_model.names[1:])
        wrist_frame = "panda_link8"
        self._wrist_frame_id = self._pin_model.getFrameId(wrist_frame)
        if self._wrist_frame_id >= len(self._pin_model.frames):
            self._wrist_frame_id = self._pin_model.getFrameId("panda_hand")
        self._ik_task = FrameTask(wrist_frame, position_cost=1.0, orientation_cost=1.0)
        self._posture_task = PostureTask(cost=0.05, gain=0.1)

    def _joint_state_dict(self, q) -> dict[str, float]:
        self._ensure_kinematics()
        return {name: float(value) for name, value in zip(self._joint_names, np.asarray(q, dtype=float))}

    def _forward_kinematics(self, q):
        self._ensure_kinematics()
        q = np.asarray(q, dtype=float)
        pin.forwardKinematics(self._pin_model, self._pin_data, q)
        pin.updateFramePlacements(self._pin_model, self._pin_data)
        oMf = self._pin_data.oMf[self._wrist_frame_id]
        wrist_position = np.asarray(oMf.translation, dtype=float).copy()
        wrist_rotation = np.asarray(oMf.rotation, dtype=float)
        tool_translation = self.end_effector_transform[:3, 3]
        tool_rotation = self.end_effector_transform[:3, :3]
        eef_position = wrist_position + wrist_rotation @ tool_translation
        eef_rotation = wrist_rotation @ tool_rotation
        eef_orientation = (HOME_ROT.inv() * R.from_matrix(eef_rotation)).as_rotvec()
        return eef_position, eef_orientation

    def connect(self):
        for name, client in self.client_names.items():
            while not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'service {name} not available, waiting again...')

        request = Panda.interfaces['connect'].Request()
        print("[panda] Calling /connect service...", flush=True)
        self.future = self.client_names['connect'].call_async(request)

        rclpy.spin_until_future_complete(self, self.future)
        print("[panda] /connect service returned.", flush=True)
        return self.future.result()

    def apply_commands(self, action=None, q_desired=None, kp=None, kd=None, gain=4.):
        debug_state_q = q_desired
        if action is not None:
            # Get current joint positions for IK seed
            request = Panda.interfaces['get_sensors'].Request()
            self.future = self.client_names['get_sensors'].call_async(request)
            rclpy.spin_until_future_complete(self, self.future)
            state = self.future.result().state
            qpos = np.array(state.position)
            debug_state_q = qpos

            eef_pos = np.array([
                action["position.x"],
                action["position.y"],
                action["position.z"]
            ])
            axis_angle = np.array([
                action["orientation.x"],
                action["orientation.y"],
                action["orientation.z"]
            ])
            if self.config.use_delta_actions:
                current_pos, current_axis_angle = self._forward_kinematics(qpos)
                eef_pos = current_pos + eef_pos
                target_rot = R.from_rotvec(axis_angle) * R.from_rotvec(current_axis_angle)
            else:
                target_rot = R.from_rotvec(axis_angle)

            axis_angle = target_rot.as_rotvec()
            eef_rot = (HOME_ROT * target_rot).as_matrix()
            tool_translation = self.end_effector_transform[:3, 3]
            tool_rotation = self.end_effector_transform[:3, :3]
            wrist_rot = eef_rot @ tool_rotation.T
            wrist_pos = eef_pos - wrist_rot @ tool_translation

            # IK
            q_desired = self.compute_ik(wrist_pos, wrist_rot, q_seed=qpos)

        request = Panda.interfaces['apply_commands'].Request()
        # Ensure q_desired is a list or array
        if isinstance(q_desired, np.ndarray):
            q_desired = q_desired.tolist()

        if q_desired is not None:
            wrist_position, wrist_orientation = self._forward_kinematics(debug_state_q)
            self._debug.log_state(
                joints=self._joint_state_dict(debug_state_q),
                state_eef_position=[float(v) for v in wrist_position],
                state_eef_orientation=[float(v) for v in wrist_orientation],
                target_eef_position=eef_pos if action is not None else None,
                target_eef_orientation=axis_angle if action is not None else None,
            )
            
        request.command = PandaCommand(position=q_desired, gain=gain)
        self.future = self.client_names['apply_commands'].call_async(request)
        return

    def get_sensors(self):
        request = Panda.interfaces['get_sensors'].Request()

        self.future = self.client_names['get_sensors'].call_async(request)
        rclpy.spin_until_future_complete(self, self.future)

        state = self.future.result().state
        
        q = np.array(state.position)

        pos, rot_axis_angle = self._forward_kinematics(q)

        pos = {f"position.{k}": v for k,v in zip(["x", "y", "z"], pos)}
        ori = {f"orientation.{k}": v for k,v in zip(["x", "y", "z"], rot_axis_angle)}

        return {**pos, **ori}

    def compute_ik(self, position, orientation, q_seed=None):
        self._ensure_kinematics()
        if q_seed is None:
            eps = 1e-6
            q_seed = np.clip(
                np.zeros(self._pin_model.nq),
                self._pin_model.lowerPositionLimit + eps,
                self._pin_model.upperPositionLimit - eps,
            )
        q_seed = np.asarray(q_seed)
        if self._q_nom is None:
            self._q_nom = q_seed.copy()
        cfg = Configuration(self._pin_model, self._pin_data, q_seed, copy_data=True, forward_kinematics=True)
        self._ik_task.set_target(pin.SE3(orientation, position))
        self._posture_task.set_target(self._q_nom)
        for _ in range(5):
            v = solve_ik(
                cfg,
                [self._ik_task, self._posture_task],
                dt=0.1,
                solver="proxqp",
                damping=0.1,
                safety_break=False,
            )
            cfg.integrate_inplace(v, 0.1)
        return cfg.q

    def close(self):
        if self._debug is not None:
            self._debug.close()

    def reset(self):
        # Define home position
        home_pos = np.array([0.0, 0.0, 0.0, -2, 0.0, 2, 0.0])
        
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
        print(f"[panda] Reset trajectory: {len(waypoints)} waypoints.", flush=True)
        for wp in waypoints:
            self.apply_commands(q_desired=wp['position'])
            time.sleep(dt)
        print("[panda] Reset trajectory complete.", flush=True)

    @property
    def features(self) -> dict:
        pos = {f"position.{d}": float for d in ["x", "y", "z"]}
        ori = {f"orientation.{d}": float for d in ["x", "y", "z"]}
        return {**pos, **ori}
