# pyright: reportMissingImports=false
import numpy as np
import rerun as rr
from klampt.math import so3, vectorops
from rerun.urdf import UrdfTree

from .config_xhand import TIPS
from ..rerun_blueprint_utils import send_custom_manipulator_blueprint

PALM_TO_TARGET_ROT = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])


class XHandDebugTools:
    def __init__(
        self,
        urdf_path: str,
        tip_scale_factors: dict[str, float],
        enable_tip_scale_tuner: bool,
        enable_rerun_visualization: bool,
        palm_frame,
        root_frame,
        tips,
    ):
        self.tip_scale_factors = tip_scale_factors
        self._step = 0
        self._tip_scale_root = None
        self._enable_rerun_visualization = enable_rerun_visualization
        self.palm_frame = palm_frame
        self.root_frame = root_frame
        self.tips = tips
        self.urdf_tree = None
        self._root_frame_id = "tf#/xhand/right_hand_link"

        if self._enable_rerun_visualization:
            if not rr.is_enabled():
                rr.init("custom_manipulator_debug", spawn=True)
            self.urdf_tree = UrdfTree.from_file_path(urdf_path, entity_path_prefix="xhand", frame_prefix="tf#/xhand/")
            self.urdf_tree.log_urdf_to_recording()
            send_custom_manipulator_blueprint()

        if enable_tip_scale_tuner:
            import tkinter as tk

            self._tip_scale_root = tk.Tk()
            self._tip_scale_root.title("xhand_tip_scales")
            for tip in TIPS:
                scale = tk.Scale(
                    self._tip_scale_root,
                    from_=0,
                    to=300,
                    orient=tk.HORIZONTAL,
                    label=tip,
                    command=lambda value, tip_name=tip: self.tip_scale_factors.__setitem__(tip_name, float(value) / 100.0),
                )
                scale.set(int(self.tip_scale_factors[tip] * 100))
                scale.pack(fill="x")
            self.poll()

    def poll(self):
        if self._tip_scale_root is not None:
            self._tip_scale_root.update_idletasks()
            self._tip_scale_root.update()

    def close(self):
        if self._tip_scale_root is not None:
            self._tip_scale_root.destroy()
            self._tip_scale_root = None

    def log_state(
        self,
        joints,
        fingertip_values: dict[str, float],
        forces: dict[str, float] | None = None,
    ):
        if not self._enable_rerun_visualization or self.urdf_tree is None:
            return

        self._step += 1
        rr.set_time("step", sequence=self._step)

        for i, joint in enumerate(self.urdf_tree.joints()):
            if joint.child_link not in joints:
                continue

            value = joints[joint.child_link]
            if hasattr(value, "getValue"):
                value = value.getValue()
            rr.log("xhand/transforms", joint.compute_transform(value))
            rr.log(f"/xhand/joints/{i}", rr.Scalars([value]))

        root_r, root_t = self.root_frame.getTransform(); root_r_inv = so3.inv(root_r)
        palm_r, palm_t = self.palm_frame.getTransform()
        palm_r_root = so3.mul(root_r_inv, palm_r)
        palm_t_root = so3.apply(root_r_inv, vectorops.sub(palm_t, root_t))
        root_tip_positions = {
            tip: [float(v) for v in vectorops.add(palm_t_root, so3.apply(palm_r_root, (PALM_TO_TARGET_ROT @ (np.array([fingertip_values[f"{tip}.position.{axis}"] for axis in "xyz"]) * self.tip_scale_factors[tip])).tolist()))]
            for tip in TIPS
        }
        self._log_points("/tips", root_tip_positions, "tip", [80, 170, 255], 0.004)
        if forces:
            rr.log("/forces", rr.Arrows3D(origins=[root_tip_positions[tip] for tip in TIPS], vectors=[[0.002 * float(v) for v in so3.apply(so3.mul(root_r_inv, self.tips[tip].getTransform()[0]), [forces[f"{tip}.force.{axis}"] for axis in "xyz"])] for tip in TIPS], colors=[[255, 80, 80]] * len(TIPS), radii=0.002), rr.CoordinateFrame(self._root_frame_id))

    def log_targets(self, target_positions: dict[str, list[float]] | None):
        if not self._enable_rerun_visualization or self.urdf_tree is None or not target_positions:
            return

        root_r, root_t = self.root_frame.getTransform(); root_r_inv = so3.inv(root_r)
        palm_r, palm_t = self.palm_frame.getTransform()
        palm_r_root = so3.mul(root_r_inv, palm_r)
        palm_t_root = so3.apply(root_r_inv, vectorops.sub(palm_t, root_t))
        root_targets = {
            tip: [float(v) for v in vectorops.add(palm_t_root, so3.apply(palm_r_root, target_positions[tip]))]
            for tip in target_positions
        }
        self._log_points("/targets", root_targets, "target", [255, 80, 80], 0.005)

    def _log_points(self, path: str, points_by_tip: dict[str, list[float]], prefix: str, color: list[int], radius: float):
        if not points_by_tip:
            return

        visible_tips = [tip for tip in TIPS if tip in points_by_tip]
        rr.log(
            path,
            rr.Points3D(
                [points_by_tip[tip] for tip in visible_tips],
                labels=[f"{prefix}_{tip}" for tip in visible_tips],
                radii=radius,
                colors=color,
            ),
            rr.CoordinateFrame(self._root_frame_id),
        )
