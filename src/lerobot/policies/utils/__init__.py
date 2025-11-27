#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Utilities for policy implementations."""

# Re-export all functions from the original utils.py module for backward compatibility
import sys
import importlib.util
from pathlib import Path

# Load the original utils.py as a module
_utils_file = Path(__file__).parent.parent / "utils.py"
_spec = importlib.util.spec_from_file_location("_policies_utils_module", _utils_file)
_utils_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils_module)

# Re-export everything from the original utils.py
populate_queues = _utils_module.populate_queues
get_device_from_parameters = _utils_module.get_device_from_parameters
get_dtype_from_parameters = _utils_module.get_dtype_from_parameters
get_output_shape = _utils_module.get_output_shape
log_model_loading_keys = _utils_module.log_model_loading_keys
prepare_observation_for_inference = _utils_module.prepare_observation_for_inference
build_inference_frame = _utils_module.build_inference_frame
make_robot_action = _utils_module.make_robot_action
raise_feature_mismatch_error = _utils_module.raise_feature_mismatch_error
validate_visual_features_consistency = _utils_module.validate_visual_features_consistency

# Import language encoder utilities
from lerobot.policies.utils.language_encoder import (
    LanguageEncoder,
    LanguageProjection,
    filter_language_encoder_from_state_dict,
)

__all__ = [
    # From original utils.py
    "populate_queues",
    "get_device_from_parameters",
    "get_dtype_from_parameters",
    "get_output_shape",
    "log_model_loading_keys",
    "prepare_observation_for_inference",
    "build_inference_frame",
    "make_robot_action",
    "raise_feature_mismatch_error",
    "validate_visual_features_consistency",
    # Language encoding utilities
    "LanguageEncoder",
    "LanguageProjection",
    "filter_language_encoder_from_state_dict",
]

