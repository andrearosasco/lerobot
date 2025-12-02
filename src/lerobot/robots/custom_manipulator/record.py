#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass, field, asdict
from pathlib import Path
from pprint import pformat

from lerobot.robots.custom_manipulator.grippers.panda_gripper import PandaGripperConfig
from scipy.spatial.transform import Rotation as R

from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts, build_dataset_frame
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import (
    make_default_processors,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
)
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.custom_manipulator.config_custom_manipulator import CustomManipulatorConfig
from lerobot.robots.custom_manipulator.arms.panda import PandaConfig
from lerobot.robots.custom_manipulator.grippers.robotiq import RobotiqConfig
from lerobot.robots.custom_manipulator.custom_manipulator import CustomManipulator
from lerobot.robots.custom_manipulator.processor.metaquest_processor import MetaQuestRelativeMotionProcessor
from lerobot.teleoperators.metaquest.metaquest_rail.configuration_metaquest import MetaQuestRailConfig
from lerobot.teleoperators.metaquest.metaquest_rail.metaquest import MetaQuestRail
from lerobot.utils.control_utils import sanity_check_dataset_name, sanity_check_dataset_robot_compatibility, is_headless
from lerobot.utils.utils import log_say, init_logging, get_safe_torch_device
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.robot_utils import busy_wait
from lerobot.policies.utils import make_robot_action
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.teleoperators import Teleoperator
from lerobot.teleoperators.so100_leader import SO100Leader
from lerobot.teleoperators.so101_leader import SO101Leader
from lerobot.teleoperators.koch_leader import KochLeader
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.utils.control_utils import predict_action
from typing import Any

import rerun as rr

#  +---------------------------------------------------------------------------------------+
#  |                                          record_loop                                  |
#  +---------------------------------------------------------------------------------------+
#  | Legend: [variable] (function)                                                         |
#  |                                                                                       |
#  |      [Robot]                                                                          |
#  |         |                                                                             |
#  |         v                                                                             |
#  |       [obs] ---------------------------------------+                                  |
#  |         |                                          |                                  |
#  |         v                                          |                                  |
#  | (robot_observation_processor)                      |                                  |
#  |         |                                          |                                  |
#  |         v                                          |                                  |
#  |   [obs_processed]                                  |                                  |
#  |         |                                          |                                  |
#  |         v                                          |                                  |
#  | (build_dataset_frame)                              |                                  |
#  |         |                                          |                                  |
#  |         v                                          |                                  |
#  | [observation_frame] -------------------------------|---------------------------+      |
#  |         |                                          |                           |      |
#  |         | (if Policy)                              |                           |      |
#  |         v                                          |                           |      |
#  |  (predict_action)       [Teleop]                   |                           |      |
#  |         |                  |                       |                           |      |
#  |         v                  v                       |                           |      |
#  |  [action_values]         [act]                     |                           |      |
#  |         |                  |                       |                           |      |
#  |         v                  v                       |                           |      |
#  | (make_robot_action) (teleop_action_processor) <----+                           |      |
#  |         |                  |                       |                           |      |
#  |         v                  v                       |                           |      |
#  | [act_processed_policy] [act_processed_teleop]      |                           |      |
#  |         |                  |                       |                           |      |
#  |         +--------+---------+                       |                           |      |
#  |                  |                                 |                           |      |
#  |                  v                                 |                           |      |
#  |           [action_values] -->(build_dataset_frame)-|-->[action_frame]------+   |      |
#  |                  |                                 |                       |   |      |
#  |                  v                                 |                       v   v      |
#  |        (robot_action_processor) <------------------+                     [Dataset]    |
#  |                  |                                                                    |
#  |                  v                                                                    |
#  |        [robot_action_to_send]                                                         |
#  |                  |                                                                    |
#  |                  v                                                                    |
#  |               [Robot]                                                                 |
#  |                                                                                       |
#  +---------------------------------------------------------------------------------------+

@safe_stop_image_writer
def record_loop(
    robot: CustomManipulator,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs after teleop
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs before robot
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],  # runs after robot
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
):
    """
    Custom record_loop forked from lerobot.scripts.lerobot_record.record_loop.
    
    Modifications:
    - Supports 'pause' behavior: frames are only added to dataset when 'is_engaged' is True.
    - Supports 'early exit' via 'exit_episode' (A button).
    - Supports 'discard episode' via 'discard_episode' (B button).
    
    Note: This function will not automatically receive updates from the upstream lerobot_record.py.
    """
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Get robot observation
        obs = robot.get_observation()

        # Applies a pipeline to the raw robot observation, default is IdentityProcessor
        obs_processed = robot_observation_processor(obs)

        if policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        # Get action from either policy or teleop
        if policy is not None and preprocessor is not None and postprocessor is not None:
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
            )

            act_processed_policy: RobotAction = make_robot_action(action_values, dataset.features)

        elif policy is None and isinstance(teleop, Teleoperator):
            act = teleop.get_action()

            rr.log('oculus_frame', rr.Transform3D(translation=[act["position.x"], act["position.y"], act["position.z"]],
                                                  mat3x3=R.from_rotvec([act["orientation.x"], act["orientation.y"], act["orientation.z"]]).as_matrix(),
                                                  axis_length=0.1)
            )

            # Check for exit signal from teleop (A button)
            if act.pop("exit_episode"):
                break

            # Check for discard signal from teleop (B button)
            if act.pop("discard_episode"):
                events["rerecord_episode"] = True
                break

            if not act.pop("is_engaged"):
                teleop_action_processor.reset()
                continue

            # Applies a pipeline to the raw teleop action, default is IdentityProcessor
            act_processed_teleop = teleop_action_processor((act, obs))

        else:
            logging.info(
                "No policy or teleoperator provided, skipping action generation."
                "This is likely to happen when resetting the environment without a teleop device."
                "The robot won't be at its rest position at the start of the next episode."
            )
            continue

        # Applies a pipeline to the action, default is IdentityProcessor
        if policy is not None and act_processed_policy is not None:
            action_values = act_processed_policy
            robot_action_to_send = robot_action_processor((act_processed_policy, obs))
        else:
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

        # Write to dataset ONLY if engaged

        
        _sent_action = robot.send_action(robot_action_to_send)

        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            log_rerun_data(observation=obs_processed, action=action_values)

        dt_s = time.perf_counter() - start_loop_t
        target_dt_s = 1 / fps
        
        # Warn if control frequency drops below target fps
        if dt_s > target_dt_s:
            actual_fps = 1 / dt_s
            logging.warning(
                f"Control frequency dropped below target: {actual_fps:.1f} Hz (actual) vs {fps} Hz (target). "
                f"Loop took {dt_s*1000:.1f}ms vs target {target_dt_s*1000:.1f}ms."
            )
        
        busy_wait(target_dt_s - dt_s)

        timestamp = time.perf_counter() - start_episode_t

@dataclass
class DatasetRecordConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str = "ar0s/pick-turtle"
    # A short but accurate description of the task performed during the recording
    single_task: str = "Pick up the turtle"
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second.
    fps: int = 10
    # Number of seconds for data recording for each episode.
    episode_time_s: int | float = 60000
    # Number of seconds for resetting the environment after each episode.
    reset_time_s: int | float = 0
    # Number of episodes to record.
    num_episodes: int = 10
    # Encode frames in the dataset into video
    video: bool = True
    # Upload dataset to Hugging Face hub.
    push_to_hub: bool = True
    # Upload on private repository on the Hugging Face hub.
    private: bool = False
    # Add tags to your dataset on the hub.
    tags: list[str] | None = None
    # Number of subprocesses handling the saving of frames as PNG.
    num_image_writer_processes: int = 0
    # Number of threads writing the frames as png images on disk, per camera.
    num_image_writer_threads_per_camera: int = 4
    # Number of episodes to record before batch encoding videos
    video_encoding_batch_size: int = 1
    # Rename map for the observation to override the image and state keys
    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")

@dataclass
class RecordConfig:
    robot: CustomManipulatorConfig = field(
        default_factory=lambda: CustomManipulatorConfig(
            arm=PandaConfig(),
            gripper=PandaGripperConfig(),
            cameras={
                "wrist": RealSenseCameraConfig(serial_number_or_name="728612070403", use_depth=False, width=640, height=480, fps=30),
                "left": RealSenseCameraConfig(serial_number_or_name="123622270882", use_depth=False, width=640, height=480, fps=30),
            }
        )
    )
    teleop: MetaQuestRailConfig = field(default_factory=MetaQuestRailConfig)
    dataset: DatasetRecordConfig = field(default_factory=DatasetRecordConfig)
    
    display_data: bool = True
    play_sounds: bool = True
    resume: bool = True

@parser.wrap()
def record(cfg: RecordConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    
    if cfg.display_data:
        init_rerun(session_name="recording_custom_manipulator")

    # Initialize robot and teleop from config
    robot = CustomManipulator(cfg.robot)
    teleop = MetaQuestRail(cfg.teleop)

    # Create processors
    # We replace the default teleop_action_processor with our custom one
    _, robot_action_processor, robot_observation_processor = make_default_processors()
    
    teleop_action_processor = RobotProcessorPipeline(
        steps=[MetaQuestRelativeMotionProcessor()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action
    )

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(
                action=robot.action_features
            ),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )
        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
    else:
        sanity_check_dataset_name(cfg.dataset.repo_id, None)
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

    robot.connect()
    teleop.connect()

    events = {
        "stop_recording": False,
        "rerecord_episode": False,
        "exit_early": False,
    }
    listener = None
    
    robot.reset()

    with VideoEncodingManager(dataset):
        recorded_episodes = 0
        while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
            log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
            record_loop(
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                dataset=dataset,
                control_time_s=cfg.dataset.episode_time_s,
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data,
            )

            if not events["stop_recording"] and (
                (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
            ):
                log_say("Reset the environment", cfg.play_sounds)
                robot.reset()
                teleop_action_processor.reset()

            if events["rerecord_episode"]:
                log_say("Re-record episode", cfg.play_sounds)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            recorded_episodes += 1

    log_say("Stop recording", cfg.play_sounds, blocking=True)

    robot.disconnect()
    teleop.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()

    if cfg.dataset.push_to_hub:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    log_say("Exiting", cfg.play_sounds)
    return dataset

if __name__ == "__main__":
    record()
