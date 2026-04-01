from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.robots.custom_manipulator.config_custom_manipulator import CustomManipulatorConfig
from lerobot.teleoperators import TeleoperatorConfig
from lerobot.teleoperators.metareader import MetaReaderConfig  # noqa: F401
from lerobot.teleoperators.metaquest.metaquest_rail.metaquest import MetaQuestRailConfig  # noqa: F401
from lerobot.robots.custom_manipulator.processor.metaquest_processor import (  # noqa: F401
    AbsolutePoseToDeltaPose,
    MetaQuestRelativeMotionProcessor,
)


@dataclass
class DatasetRecordConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str = "steb6/pick_bread"
    # A short but accurate description of the task performed during the recording
    single_task: str = "Pick the bread and place it in the plate."
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second.
    fps: int = 10
    # Number of seconds for data recording for each episode.
    episode_time_s: int | float = 2000
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
    robot: CustomManipulatorConfig
    dataset: DatasetRecordConfig
    # Whether to control the robot with a teleoperator
    teleop: TeleoperatorConfig | None = None
    # Whether to control the robot with a policy
    policy: PreTrainedConfig | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False
    teleop_action_processor: dict[str, Any] = field(
        default_factory=lambda: {"steps": ["metaquest_relative_motion_processor"]}
    )
    robot_action_processor: dict[str, Any] = field(default_factory=lambda: {"steps": []})
    robot_observation_processor: dict[str, Any] = field(default_factory=lambda: {"steps": []})

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.teleop is None and self.policy is None:
            raise ValueError("Choose a policy, a teleoperator or both to control the robot")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]
