from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.robots.custom_manipulator.config_custom_manipulator import CustomManipulatorConfig
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators import TeleoperatorConfig
from lerobot.teleoperators.metareader import MetaReaderConfig  # noqa: F401
from lerobot.teleoperators.metaquest.metaquest_rail.metaquest import MetaQuestRailConfig  # noqa: F401
from lerobot.robots.custom_manipulator.processor.metaquest_processor import (  # noqa: F401
    ArmAbsoluteToDelta,
    ClutchProcessor,
    HandAbsoluteToDelta,
)


def get_policy_loading_source(policy: PreTrainedConfig | None) -> str | None:
    """Return the configured pretrained source for a policy, if any."""
    if policy is None or policy.pretrained_path is None:
        return None
    return str(policy.pretrained_path)


def get_missing_policy_source_message(policy: PreTrainedConfig) -> str:
    repo_hint = str(policy.repo_id or "<hub-repo-or-local-dir>")
    return (
        f"Custom manipulator policy execution requires a pretrained checkpoint. "
        f"A policy of type '{policy.type}' was configured without `policy.pretrained_path`. "
        f"`policy.repo_id={policy.repo_id!r}` does not load weights in this script. "
        f"Use `--policy.path={repo_hint}` or set `policy.pretrained_path: {repo_hint}` in the config."
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
    robot: RobotConfig
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
        default_factory=lambda: {"steps": ["clutch_processor"]}
    )
    robot_action_processor: dict[str, Any] = field(default_factory=lambda: {"steps": []})
    robot_observation_processor: dict[str, Any] = field(default_factory=lambda: {"steps": []})

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = Path(policy_path)

        if self.teleop is None and self.policy is None:
            raise ValueError("Choose a policy, a teleoperator or both to control the robot")

        if not isinstance(self.robot, CustomManipulatorConfig):
            raise ValueError(
                "Custom manipulator recording expects robot.type='custom_manipulator', "
                f"got {self.robot.type!r}."
            )

        if self.policy is not None and get_policy_loading_source(self.policy) is None:
            raise ValueError(get_missing_policy_source_message(self.policy))

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]
