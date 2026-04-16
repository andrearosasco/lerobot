#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CubRobotProfile:
    robot_model: str
    yarp_robot_name: str
    urdf_env_var: str
    urdf_fallback_filename: str
    torso_joint_names: tuple[str, ...]
    torso_state_indices: tuple[int, ...]
    left_arm_joint_names: tuple[str, ...]
    right_arm_joint_names: tuple[str, ...]
    left_hand_frame_name: str
    right_hand_frame_name: str
    neck_joint_names: tuple[str, ...]
    neck_state_indices: tuple[int, ...]
    head_frame_name: str


CUB_ROBOT_PROFILES: dict[str, CubRobotProfile] = {
    "ergocub": CubRobotProfile(
        robot_model="ergocub",
        yarp_robot_name="ergoCubSN002",
        urdf_env_var="ERGOCUB_URDF_PATH",
        urdf_fallback_filename="model.urdf",
        torso_joint_names=("torso_roll", "torso_pitch", "torso_yaw"),
        torso_state_indices=(0, 1, 2),
        left_arm_joint_names=(
            "l_shoulder_pitch",
            "l_shoulder_roll",
            "l_shoulder_yaw",
            "l_elbow",
            "l_wrist_yaw",
            "l_wrist_roll",
            "l_wrist_pitch",
        ),
        right_arm_joint_names=(
            "r_shoulder_pitch",
            "r_shoulder_roll",
            "r_shoulder_yaw",
            "r_elbow",
            "r_wrist_yaw",
            "r_wrist_roll",
            "r_wrist_pitch",
        ),
        left_hand_frame_name="l_hand_palm",
        right_hand_frame_name="r_hand_palm",
        neck_joint_names=("neck_pitch", "neck_roll", "neck_yaw"),
        neck_state_indices=(0, 1, 2),
        head_frame_name="head",
    ),
    "r1": CubRobotProfile(
        robot_model="r1",
        yarp_robot_name="R1SN003",
        urdf_env_var="R1_URDF_PATH",
        urdf_fallback_filename="model.urdf",
        torso_joint_names=("torso_yaw_joint",),
        torso_state_indices=(-1,),
        left_arm_joint_names=(
            "l_shoulder_pitch",
            "l_shoulder_roll",
            "l_shoulder_yaw",
            "l_elbow",
            "l_wrist_yaw",
            "l_wrist_roll",
            "l_wrist_pitch",
        ),
        right_arm_joint_names=(
            "r_shoulder_pitch",
            "r_shoulder_roll",
            "r_shoulder_yaw",
            "r_elbow",
            "r_wrist_yaw",
            "r_wrist_roll",
            "r_wrist_pitch",
        ),
        left_hand_frame_name="l_hand_palm",
        right_hand_frame_name="r_hand_palm",
        neck_joint_names=("neck_pitch_joint", "neck_yaw_joint"),
        neck_state_indices=(0, 1),
        head_frame_name="head_link",
    ),
}


def get_cub_robot_profile(robot_model: str) -> CubRobotProfile:
    try:
        return CUB_ROBOT_PROFILES[robot_model]
    except KeyError as exc:
        raise ValueError(f"Unsupported cub robot model: {robot_model!r}") from exc
