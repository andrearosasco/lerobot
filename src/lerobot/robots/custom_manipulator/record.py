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

import importlib
import logging
import time
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

from pyparsing import Optional

from scipy.spatial.transform import Rotation as R

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.utils.feature_utils import combine_feature_dicts, build_dataset_frame
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import ProcessorStepRegistry, RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.custom_manipulator.custom_manipulator import CustomManipulator
from lerobot.robots.custom_manipulator.episode_start_overlay import make_episode_start_overlay
from lerobot.robots.custom_manipulator.record_config import (
    RecordConfig,
    get_missing_policy_source_message,
    get_policy_loading_source,
)
from lerobot.common.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.utils import log_say, init_logging
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
from lerobot.utils.robot_utils import precise_sleep
from lerobot.policies.utils import make_robot_action
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.teleoperators import Teleoperator, make_teleoperator_from_config
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.common.control_utils import predict_action
from typing import Any, List

import rerun as rr

# import debugpy
# debugpy.listen(5678)
# print('Waiting for client...')
# debugpy.wait_for_client()

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
    overlay_viewer = None,
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
    teleop_was_engaged = False
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Get robot observation
        obs = robot.get_observation()
        if overlay_viewer is not None:
            overlay_viewer.show(obs)

        # Applies a pipeline to the raw robot observation, default is IdentityProcessor
        obs_processed = robot_observation_processor(obs)

        if policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        teleop_engaged = False
        selected_action = None

        if isinstance(teleop, Teleoperator):
            act = teleop.get_action()

            rr.log('oculus_frame', rr.Transform3D(translation=[act["position.x"], act["position.y"], act["position.z"]],
                                                  mat3x3=R.from_rotvec([act["orientation.x"], act["orientation.y"], act["orientation.z"]]).as_matrix(),
                                                  )
            )

            # Check for exit signal from teleop (A button)
            if act.pop("exit_episode"):
                break

            # Check for discard signal from teleop (B button)
            if act.pop("discard_episode"):
                events["rerecord_episode"] = True
                break

            teleop_engaged = bool(act.pop("is_engaged"))

        # Get action from either policy or teleop
        if teleop_engaged:
            selected_action = teleop_action_processor((act, obs))
        else:
            if isinstance(teleop, Teleoperator):
                teleop_action_processor.reset()

                if teleop_was_engaged and policy is not None and preprocessor is not None and postprocessor is not None:
                    policy.reset()
                    preprocessor.reset()
                    postprocessor.reset()

                if policy is None:
                    teleop_was_engaged = teleop_engaged
                    continue

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
                selected_action = make_robot_action(action_values, dataset.features)
            else:
                logging.info(
                    "No policy or teleoperator provided, skipping action generation."
                    "This is likely to happen when resetting the environment without a teleop device."
                    "The robot won't be at its rest position at the start of the next episode."
                )
                teleop_was_engaged = teleop_engaged
                continue

        # Applies a pipeline to the action, default is IdentityProcessor
        action_values = selected_action
        robot_action_to_send = robot_action_processor((selected_action, obs))

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

        teleop_was_engaged = teleop_engaged
        
        precise_sleep(target_dt_s - dt_s)

        timestamp = time.perf_counter() - start_episode_t


def _instantiate_processor_step(step_spec: Any):
    if isinstance(step_spec, str):
        step_class = ProcessorStepRegistry.get(step_spec)
        return step_class()

    if isinstance(step_spec, dict):
        if "registry_name" in step_spec:
            step_class = ProcessorStepRegistry.get(step_spec["registry_name"])
        elif "class" in step_spec:
            module_path, class_name = step_spec["class"].rsplit(".", 1)
            module = importlib.import_module(module_path)
            step_class = getattr(module, class_name)
        else:
            raise ValueError(
                f"Invalid processor step config {step_spec!r}. Expected a string, or a dict with "
                f"'registry_name' or 'class'."
            )
        return step_class(**step_spec.get("config", {}))

    raise TypeError(f"Unsupported processor step spec: {step_spec!r}")


def _build_robot_processor_pipeline(
    processor_cfg: dict[str, Any],
    *,
    to_transition,
    to_output,
) -> RobotProcessorPipeline:
    steps = [_instantiate_processor_step(step_spec) for step_spec in processor_cfg.get("steps", [])]
    return RobotProcessorPipeline(
        steps=steps,
        to_transition=to_transition,
        to_output=to_output,
    )


def _resolve_resume_root(cfg: RecordConfig) -> Path:
    if cfg.dataset.root is not None:
        return Path(cfg.dataset.root)

    root = HF_LEROBOT_HOME / cfg.dataset.repo_id
    logging.info(
        "Resuming recording without `dataset.root`; using the default local dataset path: %s",
        root,
    )
    return root

@parser.wrap(config_path='cfgs/record.yaml')
def record(cfg: RecordConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    
    if cfg.display_data:
        init_rerun(session_name="recording_custom_manipulator")

    # Initialize robot and teleop from config
    robot = CustomManipulator(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    teleop_action_processor = _build_robot_processor_pipeline(
        cfg.teleop_action_processor,
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    robot_action_processor = _build_robot_processor_pipeline(
        cfg.robot_action_processor,
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    robot_observation_processor = _build_robot_processor_pipeline(
        cfg.robot_observation_processor,
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
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
        num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
        dataset = LeRobotDataset.resume(
            cfg.dataset.repo_id,
            root=_resolve_resume_root(cfg),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
            if num_cameras > 0
            else 0,
        )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
    else:
        sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
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

    # Load pretrained policy
    if cfg.policy is not None:
        policy_source = get_policy_loading_source(cfg.policy)
        if policy_source is None:
            raise ValueError(get_missing_policy_source_message(cfg.policy))
        logging.info("Loading pretrained policy '%s' from '%s'.", cfg.policy.type, policy_source)

    policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)

    preprocessor = None
    postprocessor = None
    if cfg.policy is not None:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
                "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
            },
        )

    overlay_viewer = None
    if cfg.display_data and not is_headless():
        overlay_viewer = make_episode_start_overlay(dataset)

    robot.connect()
    if cfg.teleop is not None:
        teleop.connect()

    listener, events = init_keyboard_listener()
    
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
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=dataset,
                control_time_s=cfg.dataset.episode_time_s,
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data,
                overlay_viewer=overlay_viewer,
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
            if overlay_viewer is not None:
                overlay_viewer.on_episode_saved(dataset)
            recorded_episodes += 1

    log_say("Stop recording", cfg.play_sounds, blocking=True)

    if overlay_viewer is not None:
        overlay_viewer.close()

    robot.disconnect()
    if cfg.teleop is not None:
        teleop.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()

    if cfg.dataset.push_to_hub:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    log_say("Exiting", cfg.play_sounds)
    return dataset

if __name__ == "__main__":
    record()
