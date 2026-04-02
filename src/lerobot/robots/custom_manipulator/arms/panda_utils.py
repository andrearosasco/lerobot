# pyright: reportMissingImports=false
import rerun as rr
from rerun.urdf import UrdfTree
from scipy.spatial.transform import Rotation as R

HOME_ROT = R.from_rotvec([3.141592653589793, 0.0, 0.0])


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
        state_eef_position: list[float],
        state_eef_orientation: list[float],
        target_eef_position: list[float] | None,
        target_eef_orientation: list[float] | None,
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

        self._log_points("/target_eef", {"eef": target_eef_position} if target_eef_position is not None else {}, "target", [255, 80, 80], 0.01)
        self._log_points("/state_eef", {"eef": state_eef_position}, "state", [80, 170, 255], 0.01)
        if target_eef_position is not None and target_eef_orientation is not None:
            rr.log("/target_eef/frame", rr.Arrows3D(origins=[target_eef_position] * 3, vectors=(0.04 * (HOME_ROT * R.from_rotvec(target_eef_orientation)).as_matrix().T).tolist(), colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]], radii=0.002), rr.CoordinateFrame("panda_link0"))
        rr.log("/state_eef/frame", rr.Arrows3D(origins=[state_eef_position] * 3, vectors=(0.04 * (HOME_ROT * R.from_rotvec(state_eef_orientation)).as_matrix().T).tolist(), colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]], radii=0.002), rr.CoordinateFrame("panda_link0"))
        self._log_orientation("/target_eef/orientation", target_eef_orientation, "target_eef")
        self._log_orientation("/state_eef/orientation", state_eef_orientation, "state_eef")

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
