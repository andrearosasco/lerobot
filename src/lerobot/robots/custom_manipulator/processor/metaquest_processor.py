import numpy as np
from scipy.spatial.transform import Rotation as R

from lerobot.processor import ProcessorStep, EnvTransition, ProcessorStepRegistry
from lerobot.configs.types import PipelineFeatureType, PolicyFeature

FINGERTIP_KEYS = [(f"{tip}.position.{axis}", f"{tip}.position.{axis}") for tip in ("thumb", "index", "middle", "ring", "little") for axis in "xyz"]


@ProcessorStepRegistry.register("clutch_processor")
class ClutchProcessor(ProcessorStep):
    def __init__(self):
        self.reset()

    def reset(self):
        self.vr_origin_pos = None
        self.vr_origin_rot = None
        self.eef_origin_pos = None
        self.eef_origin_rot = None
        self.engaged_prev = False

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition["action"]
        obs = transition["observation"]
        # action and obs are dicts
        
        # Extract VR pose (axis-angle)
        vr_pos = np.array([action["position.x"], action["position.y"], action["position.z"]])
        vr_rot_vec = np.array([action["orientation.x"], action["orientation.y"], action["orientation.z"]])
        vr_rot = R.from_rotvec(vr_rot_vec)
        
        # Extract Robot pose (axis-angle)
        eef_pos = np.array([obs[f"position.{d}"] for d in ['x', 'y', 'z']])
        eef_rot_vec = np.array([obs[f"orientation.{d}"] for d in ['x', 'y', 'z']])
        eef_rot = R.from_rotvec(eef_rot_vec)

        if not self.engaged_prev:
            # Rising edge: set origins
            self.vr_origin_pos = vr_pos
            self.vr_origin_rot = vr_rot
            self.eef_origin_pos = eef_pos
            self.eef_origin_rot = eef_rot
        
        # Calculate relative motion
        rel_pos = vr_pos - self.vr_origin_pos
        target_pos = self.eef_origin_pos + rel_pos
        
        # Calculate relative rotation in the world frame from the clutch pose.
        # R_delta_global = R_current * R_origin^T
        rel_rot_global = vr_rot * self.vr_origin_rot.inv()

        # Apply the world-frame delta on the left of the robot EEF origin.
        # R_target = R_delta_global * R_eef_origin
        target_rot = rel_rot_global * self.eef_origin_rot
        
        # Convert back to axis angle for action
        target_rot_vec = target_rot.as_rotvec()
        
        new_action = action.copy()
        new_action["position.x"] = target_pos[0]
        new_action["position.y"] = target_pos[1]
        new_action["position.z"] = target_pos[2]
        new_action["orientation.x"] = target_rot_vec[0]
        new_action["orientation.y"] = target_rot_vec[1]
        new_action["orientation.z"] = target_rot_vec[2]
        
        self.engaged_prev = True
        transition["action"] = new_action
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("arm_absolute_to_delta")
class ArmAbsoluteToDelta(ProcessorStep):
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition["action"]
        obs = transition["observation"]

        target_pos = np.array([action["position.x"], action["position.y"], action["position.z"]])
        target_rot = R.from_rotvec([action["orientation.x"], action["orientation.y"], action["orientation.z"]])
        current_pos = np.array([obs["position.x"], obs["position.y"], obs["position.z"]])
        current_rot = R.from_rotvec([obs["orientation.x"], obs["orientation.y"], obs["orientation.z"]])

        delta_pos = target_pos - current_pos
        delta_rot = target_rot * current_rot.inv()
        delta_rot_vec = delta_rot.as_rotvec()

        new_action = action.copy()
        new_action["position.x"] = delta_pos[0]
        new_action["position.y"] = delta_pos[1]
        new_action["position.z"] = delta_pos[2]
        new_action["orientation.x"] = delta_rot_vec[0]
        new_action["orientation.y"] = delta_rot_vec[1]
        new_action["orientation.z"] = delta_rot_vec[2]
        transition["action"] = new_action
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("hand_absolute_to_delta")
class HandAbsoluteToDelta(ProcessorStep):
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition["action"]
        obs = transition["observation"]

        if not all(action_key in action for action_key, _ in FINGERTIP_KEYS) or not all(obs_key in obs for _, obs_key in FINGERTIP_KEYS):
            return transition

        transition["action"] = action | {
            action_key: action[action_key] - obs[obs_key] for action_key, obs_key in FINGERTIP_KEYS
        }
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
