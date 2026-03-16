#!/usr/bin/env python3
from __future__ import annotations

import argparse
import errno
import math
import os
import time
import shutil
import subprocess
import sys
from pathlib import Path


def ensure_viewer_environment() -> None:
    if os.environ.get("CONDA_DEFAULT_ENV") == "websockets_working":
        return
    if os.environ.get("METAREADER_VIEWER_BOOTSTRAPPED") == "1":
        return

    direct_python = Path.home() / "miniconda3" / "envs" / "websockets_working" / "bin" / "python"
    child_env = dict(os.environ)
    child_env["METAREADER_VIEWER_BOOTSTRAPPED"] = "1"
    if direct_python.exists():
        raise SystemExit(subprocess.call([str(direct_python), __file__, *sys.argv[1:]], env=child_env))

    conda_binary = shutil.which("conda")
    if conda_binary is None:
        return

    env_list = subprocess.run([conda_binary, "env", "list"], capture_output=True, text=True, check=False)
    if env_list.returncode != 0 or "websockets_working" not in env_list.stdout:
        return

    command = [conda_binary, "run", "-n", "websockets_working", "python", __file__, *sys.argv[1:]]
    raise SystemExit(subprocess.call(command, env=child_env))


ensure_viewer_environment()

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from metareader import MetaReader
from metareader import HandState
from metareader import Pose
from metareader import TelemetryFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show live Quest head and hand poses in a 3D plot.")
    parser.add_argument("--port", type=int, default=5005, help="Quest transport port. Default: 5005")
    parser.add_argument("--tcp-port", type=int, default=5005, help="Host TCP port for adb reverse. Default: 5005")
    parser.add_argument("--trail-length", type=int, default=120, help="Number of head samples to keep in the trail. Default: 120")
    parser.add_argument("--radius", type=float, default=0.45, help="Half-width of the displayed 3D volume in meters. Default: 0.45")
    parser.add_argument("--no-auto-adb-reverse", action="store_true", help="Disable automatic adb reverse setup.")
    parser.add_argument("--no-advertise", action="store_true", help="Disable Avahi advertisement in wireless mode.")
    return parser.parse_args()


def set_axes_cube(ax, center: tuple[float, float, float], radius: float) -> None:
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def pose_position(pose: Pose | None) -> tuple[float, float, float] | None:
    if pose is None or not pose.valid:
        return None
    return pose.position


def vector_add(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def vector_sub(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def vector_scale(vector: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return (vector[0] * scale, vector[1] * scale, vector[2] * scale)


def vector_cross(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def vector_norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])


def normalize_vector(vector: tuple[float, float, float]) -> tuple[float, float, float] | None:
    norm = vector_norm(vector)
    if norm <= 1e-6:
        return None
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def rotation_matrix_to_quaternion(
    rotation: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
) -> tuple[float, float, float, float]:
    r00, r01, r02 = rotation[0]
    r10, r11, r12 = rotation[1]
    r20, r21, r22 = rotation[2]
    trace = r00 + r11 + r22

    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return (
            (r21 - r12) / scale,
            (r02 - r20) / scale,
            (r10 - r01) / scale,
            0.25 * scale,
        )
    if r00 > r11 and r00 > r22:
        scale = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        return (
            0.25 * scale,
            (r01 + r10) / scale,
            (r02 + r20) / scale,
            (r21 - r12) / scale,
        )
    if r11 > r22:
        scale = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        return (
            (r01 + r10) / scale,
            0.25 * scale,
            (r12 + r21) / scale,
            (r02 - r20) / scale,
        )

    scale = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
    return (
        (r02 + r20) / scale,
        (r12 + r21) / scale,
        0.25 * scale,
        (r10 - r01) / scale,
    )


def tracked_fingertip_positions(hand: HandState) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    for fingertip in hand.fingertips.values():
        if fingertip.tracked and fingertip.pose is not None:
            points.append(fingertip.pose.position)
    return points


def synthesize_hand_pose_from_fingertips(hand: HandState) -> Pose | None:
    points = tracked_fingertip_positions(hand)
    if len(points) < 2:
        return None

    origin = (0.0, 0.0, 0.0)
    for point in points:
        origin = vector_add(origin, point)
    origin = vector_scale(origin, 1.0 / len(points))

    index_tip = hand.fingertips.get("index_tip")
    little_tip = hand.fingertips.get("little_tip")
    middle_tip = hand.fingertips.get("middle_tip")

    across = None
    if index_tip and index_tip.pose and index_tip.pose.valid and little_tip and little_tip.pose and little_tip.pose.valid:
        across = normalize_vector(vector_sub(index_tip.pose.position, little_tip.pose.position))
    elif len(points) >= 2:
        across = normalize_vector(vector_sub(points[0], points[-1]))

    forward = None
    if middle_tip and middle_tip.pose and middle_tip.pose.valid:
        forward = normalize_vector(vector_sub(middle_tip.pose.position, origin))
    elif hand.wrist.pose is not None and hand.wrist.pose.valid:
        forward = normalize_vector(vector_sub(origin, hand.wrist.pose.position))

    if across is None or forward is None:
        return Pose(position=origin, orientation=(0.0, 0.0, 0.0, 1.0), valid=True)

    normal = normalize_vector(vector_cross(across, forward))
    if normal is None:
        return Pose(position=origin, orientation=(0.0, 0.0, 0.0, 1.0), valid=True)

    forward = normalize_vector(vector_cross(normal, across))
    if forward is None:
        return Pose(position=origin, orientation=(0.0, 0.0, 0.0, 1.0), valid=True)

    orientation = rotation_matrix_to_quaternion((across, forward, normal))
    return Pose(position=origin, orientation=orientation, valid=True)


def hand_reference_pose(hand: HandState) -> tuple[Pose | None, str]:
    for joint in (hand.palm, hand.wrist):
        if joint.tracked and joint.pose is not None:
            return joint.pose, "palm" if joint is hand.palm else "wrist"

    synthesized = synthesize_hand_pose_from_fingertips(hand)
    if synthesized is not None:
        return synthesized, "fingertips"

    return None, "none"


def quaternion_to_rotation_matrix(quaternion: tuple[float, float, float, float]) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    x, y, z, w = quaternion
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        return (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )

    x /= norm
    y /= norm
    z /= norm
    w /= norm

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def collect_hand_points(frame: TelemetryFrame) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    left_points: list[tuple[float, float, float]] = []
    right_points: list[tuple[float, float, float]] = []

    for hand, target in ((frame.left_hand, left_points), (frame.right_hand, right_points)):
        for joint in (hand.wrist, hand.palm):
            if joint.tracked and joint.pose is not None:
                target.append(joint.pose.position)
        for fingertip in hand.fingertips.values():
            if fingertip.tracked and fingertip.pose is not None:
                target.append(fingertip.pose.position)

    return left_points, right_points


def format_pose_position(pose: Pose | None) -> str:
    if pose is None or not pose.valid:
        return "none"
    x, y, z = pose.position
    return f"({x:.3f}, {y:.3f}, {z:.3f})"


def format_joint_position(hand: HandState, joint_name: str) -> str:
    joint = getattr(hand, joint_name)
    if not joint.tracked or joint.pose is None:
        return "none"
    x, y, z = joint.pose.position
    return f"({x:.3f}, {y:.3f}, {z:.3f})"


def draw_connection(ax, start: tuple[float, float, float], end: tuple[float, float, float], color: str) -> None:
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        [start[2], end[2]],
        color=color,
        linewidth=2.5,
        linestyle="--",
        alpha=0.9,
    )


def expanded_radius(
    minimum_radius: float,
    positions: list[tuple[float, float, float]],
    margin: float = 0.12,
) -> float:
    if not positions:
        return minimum_radius

    max_extent = 0.0
    for x, y, z in positions:
        max_extent = max(max_extent, abs(x), abs(y), abs(z))
    return max(minimum_radius, max_extent + margin)


def draw_reference_frame(
    ax,
    pose: Pose,
    axis_length: float,
    axis_colors: tuple[str, str, str],
    origin_color: str,
    origin_size: float,
    label: str | None = None,
) -> None:
    rotation = quaternion_to_rotation_matrix(pose.orientation)
    origin_x, origin_y, origin_z = pose.position

    ax.scatter([origin_x], [origin_y], [origin_z], color=origin_color, s=origin_size, depthshade=False)
    if label is not None:
        ax.text(origin_x, origin_y, origin_z, label, color=origin_color)

    for axis_index, color in enumerate(axis_colors):
        direction = (
            rotation[0][axis_index],
            rotation[1][axis_index],
            rotation[2][axis_index],
        )
        end_x = origin_x + direction[0] * axis_length
        end_y = origin_y + direction[1] * axis_length
        end_z = origin_z + direction[2] * axis_length
        ax.plot(
            [origin_x, end_x],
            [origin_y, end_y],
            [origin_z, end_z],
            color=color,
            linewidth=3.0,
            solid_capstyle="round",
        )


def draw_frame(
    ax,
    frame: TelemetryFrame,
    radius: float,
    view_center: tuple[float, float, float],
) -> None:
    ax.cla()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=22, azim=42)

    left_points, right_points = collect_hand_points(frame)
    left_hand_pose, left_hand_source = hand_reference_pose(frame.left_hand)
    right_hand_pose, right_hand_source = hand_reference_pose(frame.right_hand)
    left_tracked_count = len(tracked_fingertip_positions(frame.left_hand))
    right_tracked_count = len(tracked_fingertip_positions(frame.right_hand))

    visible_positions = [frame.head_pose.position]
    if left_hand_pose is not None:
        visible_positions.append(left_hand_pose.position)
    if right_hand_pose is not None:
        visible_positions.append(right_hand_pose.position)
    visible_positions.extend(left_points)
    visible_positions.extend(right_points)
    visible_radius = expanded_radius(radius, visible_positions)

    set_axes_cube(ax, view_center, visible_radius)
    draw_reference_frame(
        ax,
        frame.head_pose,
        axis_length=0.10,
        axis_colors=("#ff4d4d", "#39d98a", "#4da6ff"),
        origin_color="#ffffff",
        origin_size=28,
        label="H",
    )

    if left_hand_pose is not None:
        draw_connection(ax, frame.head_pose.position, left_hand_pose.position, color="#ff9f1c")
        draw_reference_frame(
            ax,
            left_hand_pose,
            axis_length=0.14,
            axis_colors=("#ff9f1c", "#ffd166", "#ffbf69"),
            origin_color="#ff9f1c",
            origin_size=90,
            label="L",
        )

    if right_hand_pose is not None:
        draw_connection(ax, frame.head_pose.position, right_hand_pose.position, color="#c77dff")
        draw_reference_frame(
            ax,
            right_hand_pose,
            axis_length=0.14,
            axis_colors=("#7b2cbf", "#c77dff", "#e0aaff"),
            origin_color="#c77dff",
            origin_size=90,
            label="R",
        )

    if left_points:
        xs, ys, zs = zip(*left_points)
        ax.scatter(xs, ys, zs, color="#39d98a", s=32, label="left hand", depthshade=False)

    if right_points:
        xs, ys, zs = zip(*right_points)
        ax.scatter(xs, ys, zs, color="#ff6b6b", s=32, label="right hand", depthshade=False)

    ax.legend(loc="upper left")
    ax.set_title(
        f"MetaReader Live View\n"
        f"mode={frame.transport} connection={frame.status.connection_state} target={frame.status.target_host}\n"
        f"LH tracked={frame.left_hand.tracked} src={left_hand_source} tips={left_tracked_count} | "
        f"RH tracked={frame.right_hand.tracked} src={right_hand_source} tips={right_tracked_count} radius={visible_radius:.2f}m"
    )


def draw_waiting(ax, radius: float, message: str) -> None:
    ax.cla()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_axes_cube(ax, (0.0, 0.0, 0.0), radius)
    ax.set_title(message)


def main() -> int:
    args = parse_args()

    try:
        reader = MetaReader(
            port=args.port,
            tcp_port=args.tcp_port,
            no_advertise=args.no_advertise,
            auto_adb_reverse=not args.no_auto_adb_reverse,
        )
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            print(
                "MetaReader live viewer could not bind the receiver port because it is already in use. "
                "Stop any existing ./receiver.sh or other live viewer process first, or run this viewer with a different host port.",
                file=sys.stderr,
            )
            return 1
        raise

    with reader:
        print(f"MetaReader viewer mode: {reader.mode.name} ({reader.mode.detail})", flush=True)
        print("Waiting for Quest frames. Close the plot window or press Ctrl+C to stop.", flush=True)
        figure = plt.figure("MetaReader Live View")
        axis = figure.add_subplot(111, projection="3d")
        state = {"received_first_frame": False, "last_debug_print_time": 0.0}
        draw_waiting(axis, args.radius, f"MetaReader Live View\nwaiting for frames in {reader.mode.name} mode")

        def update(_frame_index: int):
            frame = reader.read_latest(timeout=0.02)
            if frame is None:
                return []

            if not state["received_first_frame"]:
                print(
                    f"Received first frame: seq={frame.sequence} transport={frame.transport} "
                    f"head={frame.head_pose.position}",
                    flush=True,
                )
                state["received_first_frame"] = True

            now = time.monotonic()
            if now - state["last_debug_print_time"] >= 1.0:
                print(
                    "palms"
                    f" seq={frame.sequence}"
                    f" left_palm={format_joint_position(frame.left_hand, 'palm')}"
                    f" right_palm={format_joint_position(frame.right_hand, 'palm')}",
                    flush=True,
                )
                state["last_debug_print_time"] = now

            draw_frame(axis, frame, args.radius, (0.0, 0.0, 0.0))
            return []

        animation = FuncAnimation(figure, update, interval=20, cache_frame_data=False)
        setattr(figure, "_meta_reader_animation", animation)

        try:
            plt.show()
        except KeyboardInterrupt:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
