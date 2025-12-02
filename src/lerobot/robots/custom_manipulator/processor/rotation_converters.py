import numpy as np
from scipy.spatial.transform import Rotation as R
from lerobot.processor import ProcessorStep, EnvTransition
from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.robots.custom_manipulator.utils import matrix_to_rotation_6d, rotation_6d_to_matrix

class AxisAngleToRot6D(ProcessorStep):
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition["action"]
        
        # Extract axis-angle
        rot_vec = np.array([action["orientation.x"], action["orientation.y"], action["orientation.z"]])
        rot = R.from_rotvec(rot_vec)
        matrix = rot.as_matrix()
        
        # Convert to 6D
        rot_6d = matrix_to_rotation_6d(matrix)
        
        new_action = action.copy()
        # Remove axis-angle keys
        del new_action["orientation.x"]
        del new_action["orientation.y"]
        del new_action["orientation.z"]
        
        # Add 6D keys
        for i in range(6):
            new_action[f"orientation_6d.{i}"] = rot_6d[i]
            
        transition["action"] = new_action
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # TODO: Update features to reflect 6D rotation
        return features

class Rot6DToAxisAngle(ProcessorStep):
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition["action"]
        
        # Extract 6D
        rot_6d = np.array([action[f"orientation_6d.{i}"] for i in range(6)])
        
        # Convert to matrix then axis-angle
        matrix = rotation_6d_to_matrix(rot_6d)
        rot = R.from_matrix(matrix)
        rot_vec = rot.as_rotvec()
        
        new_action = action.copy()
        # Remove 6D keys
        for i in range(6):
            del new_action[f"orientation_6d.{i}"]
            
        # Add axis-angle keys
        new_action["orientation.x"] = rot_vec[0]
        new_action["orientation.y"] = rot_vec[1]
        new_action["orientation.z"] = rot_vec[2]
        
        transition["action"] = new_action
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # TODO: Update features to reflect axis-angle
        return features
