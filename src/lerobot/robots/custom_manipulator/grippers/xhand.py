# pyright: reportMissingImports=false
import time
from klampt import IKObjective, IKSolver, WorldModel
from klampt.math import so3, vectorops
import numpy as np
from xhand_controller import xhand_control
from .config_xhand import COMMAND_INDEX_BY_DRIVER_NAME, TIP_ACTIONS, TIPS, XHandConfig
from .xhand_utils import XHandDebugTools

PALM_TO_TARGET_ROT = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])

class XHand:
    end_effector_transforms = {
        "panda": np.array(
            [
                [1.0, 0.0, 0.0, 0.0455 + 0.065],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.036],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
    }

    def __init__(self, config: XHandConfig | None = None):
        self.config = config or XHandConfig()
        
        self.xhand = xhand_control.XHandControl()
        world = WorldModel()
        world.loadRobot(self.config.urdf_path)
        self.model = world.robot(0)
        self.world = world # keep a reference to prevent garbage collection

        self.joints = {self.model.driver(i).getName(): self.model.driver(i) for i in range(self.model.numDrivers())}
        self.links = [joint.getAffectedLink() for joint in self.joints.values()]

        # Reference frames
        self.palm_frame = self.model.link(self.config.palm_link_name)
        self.tips = {tip: self.model.link(self.config.tip_link_names[tip]) for tip in self.config.tip_link_names}

        # Setting up IK solver
        self.solver = IKSolver(self.model)
        self.solver.setActiveDofs(self.links)
        self.solver.setMaxIters(self.config.niter)
        self.solver.setTolerance(1e-3)

        self._debug = XHandDebugTools(self.config.urdf_path, self.config.tip_scale_factors, self.config.enable_tip_scale_tuner, 
                                      self.config.visualize, self.palm_frame, self.model.link("right_hand_link"), self.tips)

    def get_end_effector_transform(self, arm_type: str) -> np.ndarray:
        return self.end_effector_transforms[arm_type].copy()

    def connect(self):
        print("[xhand] Enumerating EtherCAT devices...", flush=True)
        ports = self.xhand.enumerate_devices("EtherCAT")
        print(f"[xhand] Enumerated ports: {ports}", flush=True)
        reply = self.xhand.open_ethercat(ports[0])
        print("[xhand] open_ethercat returned.", flush=True)

        if reply.error_code != 0:
            raise RuntimeError(f"Failed to open xHand device: {reply.error_message}")

    def disconnect(self):
        if self.xhand is not None:
            self.xhand.close_device()
        self.xhand = None
        self._debug.close()
    close = disconnect

    def reset(self):
        # reset tactile sensors
        print("[xhand] Resetting tactile sensors...", flush=True)
        for i in [17, 18, 19, 20, 21]:
            err_struct = self.xhand.reset_sensor(0,i)

        for joint in self.joints.values():
            joint.setValue(0.0)
        print("[xhand] Sending zero command...", flush=True)
        self._send(self.joints)
        print("[xhand] Reset complete.", flush=True)

    def _get_fingertips(self, joints: list[float]) -> dict[str, float]:
        previous_joint_values = {name: self.joints[name].getValue() for name in COMMAND_INDEX_BY_DRIVER_NAME}
        for name, idx in COMMAND_INDEX_BY_DRIVER_NAME.items():
            self.joints[name].setValue(float(joints[idx]))
        self.model.setConfig(self.model.getConfig())

        palm_r, palm_t = self.palm_frame.getTransform()
        palm_r_inv = so3.inv(palm_r)

        tip_positions = {
            tip: PALM_TO_TARGET_ROT.T @ np.array(
                so3.apply(palm_r_inv, vectorops.sub(link.getTransform()[1], palm_t))
            ) / self.config.tip_scale_factors[tip]
            for tip, link in self.tips.items()
        }

        fingertips = {
            f"{tip}.position.{axis}": float(value)
            for tip, tip_position in tip_positions.items()
            for axis, value in zip("xyz", tip_position)
        }

        for name, value in previous_joint_values.items():
            self.joints[name].setValue(value)
        self.model.setConfig(self.model.getConfig())
        return fingertips


    def apply_commands(self, action: dict | None = None, **kwargs):
        del kwargs
        if not action:
            return
        if self.config.use_delta_actions:
            fingertips = self._get_fingertips(self._read())
            action = action | {key: action[key] + fingertips[key] for key in fingertips if key in action}

        # self._debug.poll()
        targets = {
            tip: (PALM_TO_TARGET_ROT @ (np.array([float(action[f"{tip}.position.{axis}"]) for axis in "xyz"]) * self.config.tip_scale_factors[tip])).tolist()
            for tip in self.tips
        }

        self.solver.clear()
        palm_index = self.palm_frame.getIndex()
        for tip in self.tips:
            objective = IKObjective()
            objective.setRelativePoint(self.tips[tip].getIndex(), palm_index, [0, 0, 0], targets[tip])
            self.solver.add(objective)
        self.solver.solve()

        self._debug.log_targets(targets)
        return self._send(self.joints)

    @property
    def action_features(self) -> dict: 
        return TIP_ACTIONS
    
    @property
    def features(self) -> dict:
        features = {f"{tip}.position.{axis}": float for tip in TIPS for axis in "xyz"}
        if self.config.read_tactile_sensors:
            features.update({f"{tip}.force.{axis}": float for tip in TIPS for axis in "xyz"})
        return features

    def get_sensors(self):
        reply, state = self.xhand.read_state(
            self.config.hand_id,
            self.config.poll_force_update,
        )
        if reply.error_code != 0:
            raise RuntimeError(f"Failed to read xHand state: {reply.error_message}")

        joints = [float(finger.position) for finger in state.finger_state]

        previous_joint_values = {
            name: self.joints[name].getValue()
            for name in COMMAND_INDEX_BY_DRIVER_NAME
        }

        for name, idx in COMMAND_INDEX_BY_DRIVER_NAME.items():
            self.joints[name].setValue(joints[idx])

        forces = {}
        if self.config.read_tactile_sensors:
            forces = {
                f"{tip}.force.{axis}": float(
                    getattr(state.sensor_data[i].calc_force, f"f{axis}")
                )
                for i, tip in enumerate(TIPS)
                for axis in "xyz"
            }

        fingertip_values = self._get_fingertips(joints)
        sensors = fingertip_values | forces

        self._debug.log_state(
            {name: joints[idx] for name, idx in COMMAND_INDEX_BY_DRIVER_NAME.items()},
            fingertip_values,
            forces=forces,
        )

        for name, value in previous_joint_values.items():
            self.joints[name].setValue(value)

        return sensors

    def _send(self, joints) -> None:
        if self.xhand is None:
            raise RuntimeError("xHand is not connected.")
        cmd = xhand_control.HandCommand_t()
        for name, joint in joints.items():
            i = COMMAND_INDEX_BY_DRIVER_NAME[name]
            finger = cmd.finger_command[i]
            finger.id = i
            finger.kp = self.config.kp
            finger.ki = self.config.ki
            finger.kd = self.config.kd
            finger.position = float(joint.getValue())
            finger.tor_max = self.config.tor_max
            finger.mode = self.config.control_mode
        reply = self.xhand.send_command(self.config.hand_id, cmd)
        if reply.error_code != 0:
            raise RuntimeError(f"Failed to send xHand command: {reply.error_message}")

    def _read(self) -> list[float]:
        if self.xhand is None:
            raise RuntimeError("xHand is not connected.")
        reply, state = self.xhand.read_state(self.config.hand_id, self.config.poll_force_update)
        if reply.error_code != 0:
            raise RuntimeError(f"Failed to read xHand state: {reply.error_message}")
        obs = [float(finger.position) for finger in state.finger_state]
        return obs
