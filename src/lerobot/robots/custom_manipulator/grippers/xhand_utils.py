# pyright: reportMissingImports=false
import rerun as rr
from klampt.math import so3, vectorops
from rerun.urdf import UrdfTree

from .config_xhand import TIPS


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

        if self._enable_rerun_visualization:
            rr.init("xhand_debug", spawn=True)
            rr.log_file_from_path(urdf_path, static=True)
            self.urdf_tree = UrdfTree.from_file_path(urdf_path)

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
        joints: list[float],
        target_positions: dict[str, list[float]],
    ):
        if not self._enable_rerun_visualization or self.urdf_tree is None:
            return

        self._step += 1
        rr.set_time("step", sequence=self._step)

        for i, joint in enumerate(self.urdf_tree.joints()):
            if joint.child_link not in joints:
                continue

            value = joints[joint.child_link].getValue()
            rr.log("transforms", joint.compute_transform(value))
            rr.log(f"/joints/{i}", rr.Scalars([value]))

        root_targets = {}
        root_tip_positions = {}
        if target_positions:
            root_r, root_t = self.root_frame.getTransform()
            palm_r, palm_t = self.palm_frame.getTransform()
            root_r_inv = so3.inv(root_r)
            palm_r_root = so3.mul(root_r_inv, palm_r)
            palm_t_root = so3.apply(root_r_inv, vectorops.sub(palm_t, root_t))
            root_targets = {
                tip: [float(v) for v in vectorops.add(palm_t_root, so3.apply(palm_r_root, target_positions[tip]))]
                for tip in target_positions
            }
            root_tip_positions = {
                tip: [float(v) for v in so3.apply(root_r_inv, vectorops.sub(self.tips[tip].getTransform()[1], root_t))]
                for tip in target_positions
            }

        self._log_points("/targets", root_targets, "target", [255, 80, 80], 0.005)
        self._log_points("/tips", root_tip_positions, "tip", [80, 170, 255], 0.004)

    @staticmethod
    def _log_points(path: str, points_by_tip: dict[str, list[float]], prefix: str, color: list[int], radius: float):
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
            rr.CoordinateFrame("right_hand_link"),
        )
