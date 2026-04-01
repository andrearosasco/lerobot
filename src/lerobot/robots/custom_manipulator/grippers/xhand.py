# pyright: reportMissingImports=false
import time
from klampt import IKObjective, IKSolver, WorldModel
import numpy as np
from xhand_controller import xhand_control
from .config_xhand import COMMAND_INDEX_BY_DRIVER_NAME, TIP_ACTIONS, XHandConfig
# from .xhand_utils import XHandDebugTools

PALM_TO_TARGET_ROT = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])

class XHand:
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

        # self._debug = XHandDebugTools(
        #     urdf_path=self.config.urdf_path,
        #     tip_scale_factors=self.config.tip_scale_factors,
        #     enable_tip_scale_tuner=self.config.enable_tip_scale_tuner,
        #     enable_rerun_visualization=self.config.enable_rerun_visualization,
        #     palm_frame=self.palm_frame,
        #     root_frame=self.model.link("right_hand_link"),
        #     tips=self.tips,
        # )
        # self._debug.log_state(self.joints, {})

    def connect(self):
        ports = self.xhand.enumerate_devices("EtherCAT")
        reply = self.xhand.open_ethercat(ports[0])

        if reply.error_code != 0:
            raise RuntimeError(f"Failed to open xHand device: {reply.error_message}")

    def disconnect(self):
        if self.xhand is not None:
            self.xhand.close_device()
        self.xhand = None
        # self._debug.close()
    close = disconnect

    def reset(self):
        for joint in self.joints.values():
            joint.setValue(0.0)
        self._send(self.joints)

    def apply_commands(self, action: dict | None = None, **kwargs):
        del kwargs
        if not action:
            return

        # self._debug.poll()
        targets = {
            tip: (PALM_TO_TARGET_ROT @ (np.array([float(action[f"fingertip.{tip}.{axis}"]) for axis in "xyz"]) * self.config.tip_scale_factors[tip])).tolist()
            for tip in self.tips
        }

        self.solver.clear()
        palm_index = self.palm_frame.getIndex()
        for tip in self.tips:
            objective = IKObjective()
            objective.setRelativePoint(self.tips[tip].getIndex(), palm_index, [0, 0, 0], targets[tip])
            self.solver.add(objective)
        self.solver.solve()

        # self._debug.log_state(self.joints, targets)
        return self._send(self.joints)

    @property
    def action_features(self) -> dict: 
        return TIP_ACTIONS
    @property
    def features(self) -> dict: 
        return {f"xhand.joint_{i}": float for i in range(self.model.numDrivers())}

    def get_sensors(self):
        joints = self._read()
        return {f"xhand.joint_{i}": float(v) for i, v in enumerate(joints)}

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
