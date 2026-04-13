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
            new_action[f"orientation_6d.{i}"] = rot_6d[i].item()
            
        transition["action"] = new_action
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        if PipelineFeatureType.ACTION in features:
            action_features = features[PipelineFeatureType.ACTION]

            # Check if we have the axis-angle keys
            if "action.orientation.x" in action_features:
                action_features_keys = list(action_features.keys())
                action_features_values = list(action_features.values())

                start = action_features_keys.index("action.orientation.x")
                end = action_features_keys.index("action.orientation.z")

                # Add 6D keys
                rot6d = {}
                dtype = action_features["action.orientation.x"]
                for i in range(6):
                    rot6d[f"action.orientation_6d.{i}"] = PolicyFeature(type=dtype, shape=())

                updated_keys = action_features_keys[:start] + list(rot6d.keys()) + action_features_keys[end+1:]
                updated_values = action_features_values[:start] + list(rot6d.values()) + action_features_values[end+1:]

                updated_action_features = dict(zip(updated_keys, updated_values))
                features[PipelineFeatureType.ACTION] = updated_action_features

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
        if PipelineFeatureType.ACTION in features:
            action_features = features[PipelineFeatureType.ACTION]

            # Check if we have the 6D keys
            if "orientation_6d.0" in action_features:
                # Get the type from one of them
                dtype = action_features["orientation_6d.0"].type

                # Remove 6D keys
                for i in range(6):
                    key = f"orientation_6d.{i}"
                    if key in action_features:
                        del action_features[key]

                # Add axis-angle keys
                for key in ["orientation.x", "orientation.y", "orientation.z"]:
                    action_features[key] = PolicyFeature(type=dtype, shape=())

        return features
