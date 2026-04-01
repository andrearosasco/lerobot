# pyright: reportMissingImports=false
import rerun as rr
from rerun.urdf import UrdfTree


class PandaDebugTools:
    def __init__(
        self,
        urdf_path: str,
        enable_rerun_visualization: bool,
    ):
        self._step = 0
        self._enable_rerun_visualization = enable_rerun_visualization
        self.urdf_tree = None

        if self._enable_rerun_visualization:
            rr.init("panda_debug", spawn=True)
            rr.log_file_from_path(urdf_path, static=True)
            self.urdf_tree = UrdfTree.from_file_path(urdf_path)

    def close(self):
        pass

    def log_state(
        self,
        joints: dict[str, float],
        wrist_position: list[float],
        wrist_orientation: list[float],
        target_position: list[float] | None,
        target_orientation: list[float] | None,
    ):
        if not self._enable_rerun_visualization or self.urdf_tree is None:
            return

        self._step += 1
        rr.set_time("step", sequence=self._step)

        for i, joint in enumerate(self.urdf_tree.joints()):
            if joint.name not in joints:
                continue

            value = joints[joint.name]
            rr.log("transforms", joint.compute_transform(value))
            rr.log(f"/joints/{i}", rr.Scalars([value]))

        self._log_points("/targets", {"wrist": target_position} if target_position is not None else {}, "target", [255, 80, 80], 0.01)
        self._log_points("/wrist", {"wrist": wrist_position}, "wrist", [80, 170, 255], 0.01)
        self._log_orientation("/targets/orientation", target_orientation, "target")
        self._log_orientation("/wrist/orientation", wrist_orientation, "wrist")

    @staticmethod
    def _log_points(path: str, points: dict[str, list[float]], prefix: str, color: list[int], radius: float):
        if not points:
            return

        names = list(points)
        rr.log(
            path,
            rr.Points3D(
                [[float(v) for v in points[name]] for name in names],
                labels=[f"{prefix}_{name}" for name in names],
                radii=radius,
                colors=color,
            ),
            rr.CoordinateFrame("panda_link0"),
        )

    @staticmethod
    def _log_orientation(path: str, orientation: list[float] | None, prefix: str):
        if orientation is None:
            return

        for axis, value in zip("xyz", orientation):
            rr.log(f"{path}/{prefix}.{axis}", rr.Scalars([float(value)]))
