#!/usr/bin/env python

# Copyright 2024 Istituto Italiano di Tecnologia. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from typing import Dict
import threading
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
import yarp
from scipy.spatial.transform import Rotation as R
from .urdf_utils import resolve_ergocub_urdf
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.model.kinematics import RobotKinematics
from pysquale import BimanualIK, PoseInput
import pinocchio as pin
import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
from pink.utils import custom_configuration_vector, get_joint_idx
from lerobot.motors.ergocub.config_real import Config as cfg

import rerun as rr
from uuid import uuid4
from urdf_parser_py import urdf as urdf_parser
from lerobot.debug.urdf_logger import URDFLogger

logger = logging.getLogger(__name__)


class ErgoCubBimanualController:
    """
    Bimanual controller for ergoCub that controls both hands through a single port.
    Uses /mc-ergocub-cartesian-bimanual/rpc:i with hand side specified as string parameter.
    """
    
    def __init__(self, remote_prefix: str, local_prefix: str, use_left_hand: bool = True, use_right_hand: bool = True, use_torso: bool = True):
        """
        Initialize bimanual controller.
        
        Args:
            remote_prefix: Remote YARP prefix (e.g., "/ergocubSim")
            local_prefix: Local YARP prefix (e.g., "/lerobot/session_id")
            use_left_hand: Whether to control left hand
            use_right_hand: Whether to control right hand
            pd_rate_hz: Streaming frequency for Position Direct (e.g., 100 Hz)
            traj_duration_s: Nominal duration of generated trajectories to reach a target
        """
        self.remote_prefix = remote_prefix
        self.local_prefix = local_prefix

        self.use_left_hand = use_left_hand
        self.use_right_hand = use_right_hand

        self.initial_position = np.array([-5.15586422e-01,  5.15563259e-01,  8.66521676e-04,  7.84699903e-01,
                                    2.41915094e-06,  2.11463754e-05, -1.72567871e-03, -5.15615291e-01,
                                    5.15524823e-01,  8.81244420e-04,  7.84702351e-01,  6.89402045e-07,
                                    2.80294707e-05, -1.71659003e-03, -4.78971553e-03,  1.54077055e-02,
                                    7.14345317e-04])
        self.joint_names = cfg.right_joints + cfg.left_joints + cfg.torso_joints

        if not use_left_hand:
            cfg.left_joints = np.array([])
        if not use_right_hand:
            cfg.right_joints= np.array([])
        # if not use_torso:  # TODO make this work
        #     cfg.torso_joints= np.array([])

        self.ik = BimanualIK(**cfg.to_dict())
        self.robot = pin.RobotWrapper.BuildFromURDF(
            filename=cfg.urdf,
            package_dirs=["."],
            root_joint=None,
        )

        # Define tasks for the inverse kinematics solver
        self.tasks = {}
        
        # A FrameTask tries to match the 6D pose of a specific frame (the hand)
        self.tasks["left"] = FrameTask(
            "l_hand_palm",
            position_cost=1.0,      # High cost for position tracking
            orientation_cost=1.0,   # High cost for orientation tracking
        )
       
        self.tasks["right"] = FrameTask(
            "r_hand_palm",
            position_cost=1.0,
            orientation_cost=1.0,
        )

        self.posture_task = PostureTask(
            cost=1e-3,  # Very low cost compared to the hand tasks
        )
        # Set the target posture to the robot's neutral configuration defined in the URDF
        self.posture_task.set_target(custom_configuration_vector(self.robot, **dict(zip(self.joint_names, self.initial_position))))
        
        # YARP ports
        self.left_encoders_port = yarp.BufferedPortBottle()
        self.right_encoders_port = yarp.BufferedPortBottle()
        self.torso_encoders_port = yarp.BufferedPortBottle()
        # use a control board remapper to control the joints
        # prepare mapping from joint name -> index in the single remapped controlboard
        # placeholders for YARP driver / interfaces (opened in connect)
        self._driver = None
        self._ipos = None
        # Use Position Direct mode interfaces
        self._iposd = None
        self._icmd = None
        self._ienc = None

        self._is_connected = False

        # Executor-based short streaming task management
        self._pd_executor: ThreadPoolExecutor | None = None
        self._last_future: Future | None = None
        
        # Initialize kinematics solvers for both hands if needed
        self.kinematics_solvers = {}
        
        if use_left_hand:
            left_joint_names = ["torso_roll", "torso_pitch", "torso_yaw", "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow", "l_wrist_yaw", "l_wrist_roll", "l_wrist_pitch"]
            self.kinematics_solvers["left"] = RobotKinematics(cfg.urdf, "l_hand_palm", left_joint_names)
            
        if use_right_hand:
            right_joint_names = ["torso_roll", "torso_pitch", "torso_yaw", "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow", "r_wrist_yaw", "r_wrist_roll", "r_wrist_pitch"]
            self.kinematics_solvers["right"] = RobotKinematics(cfg.urdf, "r_hand_palm", right_joint_names)

        # Persistent PositionDirect publisher state (created on connect)
        # self._pd_thread: threading.Thread | None = None
        # self._pd_stop_evt = threading.Event()
        # self._pd_new_goal_evt = threading.Event()
        # self._pd_lock = threading.Lock()
        # self._pd_dt = 0.01  # 100 Hz by default
        # self._nominal_vel = 1.0  # rad/s cap for planning
        # self._min_traj_T = 0.25
        # self._max_traj_T = 1.0
        # self._active_traj: dict | None = None  # {q0, q1, T, t0}
        # self._last_q_ref: np.ndarray | None = None
        # self._q_goal: np.ndarray | None = None
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    def connect(self) -> None:
        """Connect to YARP bimanual control port."""
        if self._is_connected:
            raise DeviceAlreadyConnectedError("ErgoCubBimanualController already connected")
        
        # Connect encoder ports for both hands
        if self.use_left_hand:
            left_encoders_local = f"{self.local_prefix}/bimanual/left_encoders:i"
            if not self.left_encoders_port.open(left_encoders_local):
                raise ConnectionError(f"Failed to open left encoders port {left_encoders_local}")
            
            left_encoders_remote = f"{self.remote_prefix}/left_arm/state:o"
            while not yarp.Network.connect(left_encoders_remote, left_encoders_local):
                logger.warning(f"Failed to connect {left_encoders_remote} -> {left_encoders_local}, retrying...")
                time.sleep(1)
        
        if self.use_right_hand:
            right_encoders_local = f"{self.local_prefix}/bimanual/right_encoders:i"
            if not self.right_encoders_port.open(right_encoders_local):
                raise ConnectionError(f"Failed to open right encoders port {right_encoders_local}")
            
            right_encoders_remote = f"{self.remote_prefix}/right_arm/state:o"
            while not yarp.Network.connect(right_encoders_remote, right_encoders_local):
                logger.warning(f"Failed to connect {right_encoders_remote} -> {right_encoders_local}, retrying...")
                time.sleep(1)
        
        # Connect torso encoders
        torso_encoders_local = f"{self.local_prefix}/bimanual/torso_encoders:i"
        if not self.torso_encoders_port.open(torso_encoders_local):
            raise ConnectionError(f"Failed to open torso encoders port {torso_encoders_local}")
        
        torso_encoders_remote = f"{self.remote_prefix}/torso/state:o"
        while not yarp.Network.connect(torso_encoders_remote, torso_encoders_local):
            logger.warning(f"Failed to connect {torso_encoders_remote} -> {torso_encoders_local}, retrying...")
            time.sleep(1)

        self.rec = rr.RecordingStream(application_id="metacub_dashboard", recording_id=rr.rid)
        rr.spawn(recording=self.rec, memory_limit="50%")
        self.rec.connect_grpc()
        # self.rec.set_time("real_time", duration=0.0)
        urdf = urdf_parser.URDF.from_xml_file(cfg.urdf)
        urdf.path = cfg.urdf
        self.urdf_logger = URDFLogger(urdf, self.rec)
        self.urdf_logger.init()

        # Open controlboard remapper PolyDriver for position control of all joints
        props = yarp.Property()
        props.put("robot", "ergocubSim")
        props.put("device", "remotecontrolboardremapper")
        props.put("localPortPrefix", f"{self.local_prefix}/bimanual/controlboard")
        remote_control_boards = yarp.Bottle()
        remote_control_boards_list = remote_control_boards.addList()
        for control_board in ["/ergocubSim/torso", "/ergocubSim/right_arm", "/ergocubSim/left_arm"]:
            remote_control_boards_list.addString(control_board)
        props.put("remoteControlBoards", remote_control_boards.get(0))
        axesNames = yarp.Bottle()
        axesNames_list = axesNames.addList()
        for joint in self.joint_names:
            axesNames_list.addString(joint)
        props.put("axesNames", axesNames.get(0))
        self._driver = yarp.PolyDriver()
        self._driver.open(props)

        self._iposd = self._driver.viewIPositionDirect()
        self._icmd = self._driver.viewIControlMode()
        self._ienc = self._driver.viewIEncoders()

        # Set all joints to POSITION_DIRECT mode
        for j in range(len(self.joint_names)):
            self._icmd.setControlMode(j, yarp.VOCAB_CM_POSITION_DIRECT)

        # Replace short-burst executor with persistent streaming thread
        # self._pd_executor = None
        # self._last_future = None

        # Initialize last reference to current encoders and start PD loop
        # self._last_q_ref = self._read_current_joints()
        # self._pd_stop_evt.clear()
        # self._pd_thread = threading.Thread(target=self._pd_loop, name="pd-stream", daemon=True)
        # self._pd_thread.start()

        self._is_connected = True
        logger.info("ErgoCubBimanualController connected")

    
    def disconnect(self) -> None:
        """Disconnect from YARP ports."""
        if not self.is_connected:
            raise DeviceNotConnectedError("ErgoCubBimanualController not connected")

        # Stop PD loop first
        if self._pd_thread is not None:
            self._pd_stop_evt.set()
            self._pd_new_goal_evt.set()
            try:
                self._pd_thread.join(timeout=2.0)
            finally:
                self._pd_thread = None

        # Close input ports
        if self.use_left_hand:
            self.left_encoders_port.close()
        if self.use_right_hand:
            self.right_encoders_port.close()
        self.torso_encoders_port.close()
        
        self._is_connected = False
        logger.info("ErgoCubBimanualController disconnected")
        # Shut down executor to avoid further streaming (kept for backward compatibility)
        if self._pd_executor is not None:
            try:
                self._pd_executor.shutdown(wait=False, cancel_futures=True)
            finally:
                self._pd_executor = None
                self._last_future = None
        # Close driver if opened
        if self._driver is not None:
            self._driver.close()
    
    def read_current_state(self) -> Dict[str, float]:
        """Read current state for both hands."""
        if not self.is_connected:
            raise DeviceNotConnectedError("ErgoCubBimanualController not connected")
        
        state = {}
        
        # Read torso encoders
        torso_bottle = self.torso_encoders_port.read(False)
        if torso_bottle:
            torso_encoders = [torso_bottle.get(i).asFloat64() for i in range(min(3, torso_bottle.size()))]
        else:
            torso_encoders = [0.0, 0.0, 0.0]
        
        # Read and compute poses for each hand
        for side in ["left", "right"]:
            if (side == "left" and not self.use_left_hand) or (side == "right" and not self.use_right_hand):
                continue
                
            # Read hand encoders with busy wait
            encoders_port = self.left_encoders_port if side == "left" else self.right_encoders_port
            read_attempts = 0
            while (hand_bottle := encoders_port.read(False)) is None:
                read_attempts += 1
                if read_attempts % 1000 == 0:  # Warning every 1000 attempts
                    logger.warning(f"Still waiting for {side} hand encoder data (attempt {read_attempts})")
                time.sleep(0.001)  # 1 millisecond sleep
            
            # Read all available encoders (hand + fingers)
            all_encoders = [hand_bottle.get(i).asFloat64() for i in range(hand_bottle.size())]
            # First 7 are hand joints, last 6 are finger joints
            hand_encoders = all_encoders[:7] if len(all_encoders) >= 7 else [0.0] * 7
            finger_encoders = all_encoders[7:13] if len(all_encoders) >= 13 else [0.0] * 6
            
            # Combine torso + hand encoders
            joint_positions = np.array(torso_encoders + hand_encoders)
            
            # Compute forward kinematics
            if side in self.kinematics_solvers:
                pose_matrix = self.kinematics_solvers[side].forward_kinematics(joint_positions)
                position = pose_matrix[:3, 3]
                rotation = R.from_matrix(pose_matrix[:3, :3])
                quaternion = rotation.as_quat(canonical=True, scalar_first=True)  # [x, y, z, w]
                
                # Add to state
                state.update({
                    f"{side}_hand.position.x": position[0].item(),
                    f"{side}_hand.position.y": position[1].item(),
                    f"{side}_hand.position.z": position[2].item(),
                    f"{side}_hand.orientation.qw": quaternion[0].item(),
                    f"{side}_hand.orientation.qx": quaternion[1].item(),
                    f"{side}_hand.orientation.qy": quaternion[2].item(),
                    f"{side}_hand.orientation.qz": quaternion[3].item(),
                })
                
                # Add actual finger values from encoders
                finger_joint_names = ["thumb_add", "thumb_oc", "index_add", "index_oc", "middle_oc", "ring_pinky_oc"]
                for i, joint in enumerate(finger_joint_names):
                    state[f"{side}_fingers.{joint}"] = finger_encoders[i] if i < len(finger_encoders) else 0.0
        
        return state
    
    def send_command(self, q: np.ndarray = None) -> None:
        """Set a new target joint vector for the persistent Position Direct streaming loop.

        q should be a 1D numpy array with length equal to the number of joints in
        `self.joint_names` (left + right + torso order used above). Units: radians.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError("ErgoCubBimanualController not connected")
        if q is None:
            return
        if q.size != len(self.joint_names):
            raise ValueError(f"Expected q of length {len(self.joint_names)}, got {q.size}")

        # Publish goal for the PD loop; preempt any ongoing trajectory
        with self._pd_lock:
            self._q_goal = np.asarray(q, dtype=float).copy()
        self._pd_new_goal_evt.set()
        # Non-blocking; the PD thread handles interpolation and timing

    
    def send_commands(self, commands: dict[str, float]) -> None:
        """Send commands to bimanual controller."""
        if not self.is_connected:
            raise DeviceNotConnectedError("ErgoCubBimanualController not connected")
        
        # Filter commands for each hand
        left_hand_cmds = {k.split(".", 1)[1]: v for k, v in commands.items() if k.startswith("left_hand.")}
        right_hand_cmds = {k.split(".", 1)[1]: v for k, v in commands.items() if k.startswith("right_hand.")}

        # Prepare inputs
        left_pose = PoseInput() if left_hand_cmds else None
        right_pose = PoseInput() if right_hand_cmds else None
        
        # Send hand commands
        for hand_name, hand_cmds in [("left", left_hand_cmds), ("right", right_hand_cmds)]:
            if hand_cmds:
                # Extract pose components - position.x, position.y, position.z, orientation qw,qx,qy,qz
                pos_keys = ["position.x", "position.y", "position.z"]
                # Build scalar-first so we can rotate slice with one-liner above
                quat_keys = ["orientation.qw", "orientation.qx", "orientation.qy", "orientation.qz"]
                
                if all(key in hand_cmds for key in pos_keys + quat_keys):
                    pose = np.array([hand_cmds[key] for key in pos_keys + quat_keys])
                    if hand_name == "left":
                        left_pose.pos = pose[:3]
                        left_pose.quat = pose[3:]
                        self.tasks['left'].set_target(pin.SE3(R.from_quat(left_pose.quat, scalar_first=True).as_matrix(), left_pose.pos))
                    else:
                        right_pose.pos = pose[:3]
                        right_pose.quat = pose[3:]
                        self.tasks['right'].set_target(pin.SE3(R.from_quat(right_pose.quat, scalar_first=True).as_matrix(), right_pose.pos))

        # Compute inverse kinematics and send commands
        # Provide current joints (radians) to IK, as required
        q_curr = self._read_current_joints()
        s = time.perf_counter()
        # q = self.ik.solve_ik(right_pose=right_pose, left_pose=left_pose, q=q_curr, vel_threshold=np.pi / 180)
                # 1. Get current robot configuration

        current_joints_dict = {name: value for name, value in zip(self.joint_names, q_curr)}
        conf_vector = custom_configuration_vector(self.robot, **current_joints_dict)
        configuration = pink.Configuration(self.robot.model, self.robot.data, conf_vector)
        

        dt = 6e-3
        linear_tolerance = 1e-3  # 1 millimeter
        angular_tolerance = 1.7e-2 # 1 degree in radians (1 * np.pi / 180)
        for _ in np.arange(20):
            velocity = solve_ik(configuration, [self.tasks['left'], self.tasks['right']], dt, solver="proxqp", safety_break=False)
            conf_vector = configuration.integrate(velocity, dt)
            configuration = pink.Configuration(self.robot.model, self.robot.data, conf_vector)

            l_pos_error = np.linalg.norm(self.tasks['left'].compute_error(configuration)[:3])
            l_rot_error = np.linalg.norm(self.tasks['left'].compute_error(configuration)[3:])
            r_pos_error = np.linalg.norm(self.tasks['right'].compute_error(configuration)[:3])
            r_rot_error = np.linalg.norm(self.tasks['right'].compute_error(configuration)[3:])

            if (l_pos_error < linear_tolerance and 
                l_rot_error < angular_tolerance and
                r_pos_error < linear_tolerance and
                r_rot_error < angular_tolerance):
                break
            # time.sleep(dt)

        q = np.array([configuration.q[get_joint_idx(self.robot.model, name)[0]] for name in self.joint_names])

        print(time.perf_counter() - s)
        # self.send_command(q)
        pos = yarp.Vector(len(self.joint_names))
        # Send command (degrees)
        for idx, angle in enumerate(q):
            pos.set(idx, angle / np.pi * 180.0)
        
        for idx, angle in enumerate(q_curr):
            self.urdf_logger.log(self.joint_names[idx], angle)

        self._iposd.setPositions(pos.data())
        
        # Handle finger commands (bimanual controller doesn't actually control fingers, 
        # but we need to accept the commands to avoid errors during recording)
        for side in ["left", "right"]:
            finger_cmds = {k.split(".", 1)[1]: v for k, v in commands.items() if k.startswith(f"{side}_fingers.")}
            # For now, just log that finger commands were received but not processed
            if finger_cmds:
                logger.debug(f"Received {side} finger commands: {finger_cmds} (not processed by bimanual controller)")

    def reset(self) -> None:
        """Reset the bimanual controller"""
        # self.send_command(self.initial_position)
        pos = yarp.Vector(len(self.joint_names))
        # Send command (degrees)
        for idx, angle in enumerate(self.initial_position):
            pos.set(idx, angle / np.pi * 180.0)
            self.urdf_logger.log(self.joint_names[idx], angle)

        self._iposd.setPositions(pos.data())

    @property
    def motor_features(self) -> dict[str, type]:
        """Get motor features for bimanual controller."""
        features = {}
        
        # Left hand
        if self.use_left_hand:
            for coord in ["x", "y", "z"]:
                features[f"left_hand.position.{coord}"] = float
            for coord in ["qw", "qx", "qy", "qz"]:
                features[f"left_hand.orientation.{coord}"] = float
        
        # Right hand
        if self.use_right_hand:
            for coord in ["x", "y", "z"]:
                features[f"right_hand.position.{coord}"] = float
            for coord in ["qw", "qx", "qy", "qz"]:
                features[f"right_hand.orientation.{coord}"] = float
        
        return features

    # Internal: read current joint positions (radians) from remapper encoders
    def _read_current_joints(self) -> np.ndarray:
        n_axes = len(self.joint_names)
        vec = yarp.Vector(n_axes)
        ok = self._ienc.getEncoders(vec.data())
        if not ok:
            return np.zeros(n_axes, dtype=float)
        return np.radians(np.array([vec.get(i) for i in range(n_axes)], dtype=float))

    def _min_jerk_spaces(self, N: int, T: float):
        """Generates a 1-D minimum-jerk trajectory scalar from 0 to 1 in N steps over T seconds."""
        assert N > 1, "Number of planning steps must be larger than 1."
        t_traj = np.linspace(0.0, 1.0, N)
        p_traj = 10 * t_traj**3 - 15 * t_traj**4 + 6 * t_traj**5
        pd_traj = (30 * t_traj**2 - 60 * t_traj**3 + 30 * t_traj**4) / T
        pdd_traj = (60 * t_traj - 180 * t_traj**2 + 120 * t_traj**3) / (T**2)
        return p_traj, pd_traj, pdd_traj

    def _generate_joint_space_min_jerk(self, start: np.ndarray, goal: np.ndarray, time_to_go: float, dt: float):
        """Primitive joint-space minimum-jerk trajectory planner (positions, velocities, accelerations)."""
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        steps = int(max(2, round(time_to_go / dt)))
        T = steps * dt
        p_traj, pd_traj, pdd_traj = self._min_jerk_spaces(steps, T)
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

    # Persistent PD loop and helpers
    def _plan_traj(self, q0: np.ndarray, q1: np.ndarray) -> dict:
        D = np.asarray(q1, dtype=float) - np.asarray(q0, dtype=float)
        max_delta = float(np.max(np.abs(D))) if D.size > 0 else 0.0
        # Duration based on per-joint max delta and nominal velocity
        T = max(self._min_traj_T, min(self._max_traj_T, max_delta / max(self._nominal_vel, 1e-6)))
        return {"q0": np.asarray(q0, float), "q1": np.asarray(q1, float), "T": T, "t0": time.perf_counter()}

    def _pd_loop(self) -> None:
        if self._iposd is None:
            return
        n_axes = len(self.joint_names)
        pos = yarp.Vector(n_axes)
        # Timing init
        dt = float(self._pd_dt)
        next_t = time.perf_counter()
        overruns = 0
        # Local copy of last reference
        q_ref = np.array(self._last_q_ref if self._last_q_ref is not None else np.zeros(n_axes), dtype=float)

        # Min-jerk scalar function
        def jerk_profile(s: float) -> float:
            return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5

        while not self._pd_stop_evt.is_set():
            now = time.perf_counter()
            # Handle new goal (preempt)
            if self._pd_new_goal_evt.is_set():
                self._pd_new_goal_evt.clear()
                with self._pd_lock:
                    q_goal = None if self._q_goal is None else self._q_goal.copy()
                if q_goal is not None:
                    # Use current encoders for start to reduce jumps
                    q0 = self._read_current_joints()
                    self._active_traj = self._plan_traj(q0, q_goal)

            # Update current reference
            if self._active_traj is not None:
                T = self._active_traj["T"]
                t0 = self._active_traj["t0"]
                s = (now - t0) / T if T > 0 else 1.0
                if s >= 1.0:
                    q_ref = self._active_traj["q1"].copy()
                    self._active_traj = None
                else:
                    s = max(0.0, min(1.0, s))
                    p = jerk_profile(s)
                    q_ref = self._active_traj["q0"] + p * (self._active_traj["q1"] - self._active_traj["q0"])
            # else: hold last q_ref

            # Send command (degrees)
            for j in range(n_axes):
                pos.set(j, float(np.degrees(q_ref[j])))
            self._iposd.setPositions(pos.data())
            for name, angle in zip(self.joint_names, q_ref):
                self.urdf_logger.log(name, angle)

            # Hybrid sleep + short spin to target next tick
            next_t += dt
            now2 = time.perf_counter()
            sleep_dur = next_t - now2 - 0.001  # leave ~1ms for spin
            if sleep_dur > 0:
                try:
                    time.sleep(sleep_dur)
                except Exception:
                    pass
            # short spin for precision
            while True:
                now3 = time.perf_counter()
                if now3 >= next_t:
                    break
            # Detect large overruns (diagnostics)
            lag = now3 - next_t
            if lag > 0.01:  # >10ms late
                overruns += 1
                if overruns % 50 == 0:
                    logger.debug(f"PD loop overrun count={overruns}, last lag={lag*1000:.1f} ms")

        # Exit: optionally hold last reference once more
        try:
            for j in range(n_axes):
                pos.set(j, float(np.degrees(q_ref[j])))
            self._iposd.setPositions(pos.data())
        except Exception:
            pass