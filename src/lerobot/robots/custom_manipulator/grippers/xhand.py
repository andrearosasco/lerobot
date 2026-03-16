# pyright: reportMissingImports=false
import time
from klampt import WorldModel
from klampt.math import so3, vectorops
from klampt.model import ik
import numpy as np
import rerun as rr
from xhand_controller import xhand_control
from .config_xhand import TIPS, TIP_ACTIONS, XHandConfig

class XHand:
    def __init__(self, config: XHandConfig | None = None):
        self.config = config or XHandConfig()
        self._device = None
        self._last = list(self.config.open_positions)
        self._world = None
        self.robot = None
        self.ik_dofs = []
        self._palm_link_name = ""
        self._tip_link_names = {}
        self._rr_ready = False
        self._rr_model_path = "xhand/model"
        self._rr_step = 0

        self._world = WorldModel()
        self._world.loadRobot(self.config.urdf_path)
        self.robot = self._world.robot(0)
        self.ik_dofs = [self.robot.driver(i).getName() for i in range(self.robot.numDrivers())]
        names = {self.robot.link(i).getName() for i in range(self.robot.numLinks())}
        self._palm_link_name = self._resolve_link_name(self.config.palm_link_name, ("right_hand_link", "right_hand_ee_link", "base0"), names)
        defaults = {
            "thumb": "right_hand_thumb_rota_tip",
            "index": "right_hand_index_rota_tip",
            "middle": "right_hand_mid_tip",
            "ring": "right_hand_ring_tip",
            "little": "right_hand_pinky_tip",
        }
        self._tip_link_names = {
            tip: self._resolve_link_name(self.config.tip_link_names.get(tip, defaults[tip]), (defaults[tip],), names)
            for tip in TIPS
        }
        self._init_rerun()
        self._log_rerun_state((self._last + [0.0] * self.robot.numDrivers())[: self.robot.numDrivers()], {})

    def _init_rerun(self):
        if self._rr_ready:
            return
        rr.init("xhand_debug", spawn=True)
        rr.log("xhand/urdf_path", rr.TextDocument(str(self.config.urdf_path), media_type="text/plain"), static=True)
        try:
            rr.log_file_from_path(str(self.config.urdf_path), entity_path_prefix=self._rr_model_path, static=True)
        except Exception:
            rr.log(self._rr_model_path, rr.TextDocument("URDF model import failed in Rerun; logging link transforms only.", media_type="text/plain"), static=True)
        self._rr_ready = True

    def _log_rerun_state(self, joints: list[float], world_targets: dict[str, list[float]]):
        if not self._rr_ready or self.robot is None:
            return
        self._rr_step += 1
        rr.set_time_sequence("step", self._rr_step)
        nd = self.robot.numDrivers()
        j = (list(joints) + [0.0] * nd)[:nd]
        self.robot.setConfig(self.robot.configFromDrivers(j))
        for i in range(nd):
            rr.log(f"xhand/joints/{self.robot.driver(i).getName()}", rr.Scalars(float(j[i])))
        for i in range(self.robot.numLinks()):
            link = self.robot.link(i)
            r, t = link.getTransform()
            rr.log(
                f"{self._rr_model_path}/{link.getName()}",
                rr.Transform3D(translation=[float(t[0]), float(t[1]), float(t[2])], mat3x3=so3.matrix(r), axis_length=0.01),
            )
        if world_targets:
            points = [world_targets[tip] for tip in TIPS if tip in world_targets]
            labels = [tip for tip in TIPS if tip in world_targets]
            if points:
                rr.log("xhand/targets", rr.Points3D(points, labels=labels, radii=0.005))

    @staticmethod
    def _resolve_link_name(name: str, fallbacks: tuple[str, ...], available: set[str]) -> str:
        for candidate in (name, *fallbacks):
            if candidate in available:
                return candidate
        raise RuntimeError(f"Invalid link name '{name}'. Available links do not include requested/fallback names.")

    def connect(self):
        if self._device is not None:
            return True
        self._device = xhand_control.XHandControl()
        if self.config.protocol.upper() == "ETHERCAT":
            ports = self._device.enumerate_devices("EtherCAT")
            reply = self._device.open_ethercat(ports[0])
        else:
            reply = self._device.open_serial(self.config.serial_port, self.config.baud_rate)
        if reply.error_code != 0:
            raise RuntimeError(f"Failed to open xHand device: {reply.error_message}")
        time.sleep(self.config.startup_delay_s)
        return True

    def disconnect(self):
        if self._device is not None:
            self._device.close_device()
        self._device = None
    close = disconnect

    def reset(self):
        self._send(self.config.open_positions)

    def apply_commands(self, action: dict | None = None, **kwargs):
        del kwargs
        if not action or sum(list(action.values())) == 0.0:
            return

        targets = {
            tip: tuple(float(action[f"fingertip.{tip}.{axis}"]) for axis in "xyz")
            for tip in TIPS
            if action and all(f"fingertip.{tip}.{axis}" in action for axis in "xyz")
        }

        if targets and self.robot is not None:
            seed = self._read()
            nd = self.robot.numDrivers()
            seed = (seed + [0.0] * nd)[:nd]
            self.robot.setConfig(self.robot.configFromDrivers(seed))
            palm_r, palm_t = self.robot.link(self._palm_link_name).getTransform()
            local_rot = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
            world_targets = {
                tip: [
                    float(v)
                    for v in vectorops.add(
                        palm_t,
                        so3.apply(palm_r, local_rot @ target),
                    )
                ]
                for tip, target in targets.items()
            }
            goals = [
                ik.objective(self.robot.link(self._tip_link_names[tip]), local=[0, 0, 0], world=world_targets[tip])
                for tip, target in targets.items()
            ]
            ik.solve(goals, iters=self.config.niter, tol=1e-3, activeDofs=self.ik_dofs)
            cmd = [self.robot.driver(i).getValue() for i in range(self.robot.numDrivers())]
            self._log_rerun_state(cmd, world_targets)
            # return self._send(cmd)

    @property
    def action_features(self) -> dict: return {"action.gripper": float, **TIP_ACTIONS}
    @property
    def features(self) -> dict: return {"gripper": float, **{f"xhand.joint_{i}": float for i in range(len(self._last))}}

    def get_sensors(self):
        joints = self._read()
        grip = sum(min(1.0, max(0.0, (joint - closed) / (opened - closed or 1e-6))) for joint, opened, closed in zip(joints, self.config.open_positions, self.config.closed_positions, strict=True)) / len(self.config.open_positions)
        return {"gripper": grip, **{f"xhand.joint_{i}": float(v) for i, v in enumerate(joints)}}

    def _send(self, joints: list[float]) -> None:
        if self._device is None:
            raise RuntimeError("xHand is not connected.")
        cmd = xhand_control.HandCommand_t()
        for i, position in enumerate(joints[:12]):
            finger = cmd.finger_command[i]
            finger.id = i
            finger.kp = self.config.kp
            finger.ki = self.config.ki
            finger.kd = self.config.kd
            finger.position = float(position)
            finger.tor_max = self.config.tor_max
            finger.mode = self.config.control_mode
        reply = self._device.send_command(self.config.hand_id, cmd)
        if reply.error_code != 0:
            raise RuntimeError(f"Failed to send xHand command: {reply.error_message}")
        self._last = list(joints[:12])

    def _read(self) -> list[float]:
        if self._device is None:
            return list(self._last)
        reply, state = self._device.read_state(self.config.hand_id, self.config.poll_force_update)
        if reply.error_code == 0:
            self._last = [float(finger.position) for finger in state.finger_state][:12]
        return list(self._last)
