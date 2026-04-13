import numpy as np
from scipy.spatial.transform import Rotation as R

from lerobot.processor import ProcessorStep, EnvTransition
from lerobot.configs.types import PipelineFeatureType, PolicyFeature

class MetaQuestRelativeMotionProcessor(ProcessorStep):
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
        
        # Calculate relative rotation in global frame (VR)
        # R_delta_global = R_current * R_origin^T
        rel_rot_global = vr_rot * self.vr_origin_rot.inv()
        
        # Apply to robot origin in local frame (EEF)
        # R_target = R_eef_origin * R_delta_global
        target_rot = self.eef_origin_rot * rel_rot_global
        
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
