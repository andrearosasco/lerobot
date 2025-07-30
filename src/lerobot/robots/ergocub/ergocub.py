#!/usr/bin/env python

# Copyright 2024 Istituto Italiano di Tecnologia. All rights reserved.
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
import uuid
from functools import cached_property
from typing import Any

import numpy as np
import yarp
from lerobot.cameras import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.ergocub import ErgoCubMotorsBus
from lerobot.robots.robot import Robot

from .configuration_ergocub import ErgoCubConfig

logger = logging.getLogger(__name__)


class ErgoCub(Robot):
    config_class = ErgoCubConfig
    name = "ergocub"
    
    def __init__(self, config: ErgoCubConfig):
        super().__init__(config)
        self.config = config
        self.session_id = uuid.uuid4()
        self._is_connected = False

        # Safety control variables for hand position checking
        self.is_arm_controlled = {"left": False, "right": False}
        self.position_tolerance = 0.1  # 10 cm tolerance like in metaControllClient (0.2 was 20cm)

        # Set YARP robot name for resource finding
        import os
        os.environ["YARP_ROBOT_NAME"] = "ergoCubSN002"

        yarp.Network.init()

        # Use custom camera creation for YARP cameras with prefixes
        self.cameras = {}
        for cam_name, cam_config in config.cameras.items():
            if cam_config.type == "yarp":
                from lerobot.cameras.yarp import YarpCamera
                self.cameras[cam_name] = YarpCamera(
                    cam_config, 
                    f"{config.local_prefix}/{self.session_id}"
                )
            else:
                # Use standard LeRobot camera factory for non-YARP cameras
                other_cameras = make_cameras_from_configs({cam_name: cam_config})
                self.cameras.update(other_cameras)

        # Create motors configuration for compatibility with MotorsBus interface
        motors = {}
        
        # Define motors for left arm (pose + fingers)
        if config.use_left_arm:
            motors.update({
                "left_arm": Motor(1, "ergocub_arm", MotorNormMode.RANGE_M100_100),
                "left_fingers": Motor(2, "ergocub_fingers", MotorNormMode.RANGE_M100_100),
            })
        
        # Define motors for right arm (pose + fingers)
        if config.use_right_arm:
            motors.update({
                "right_arm": Motor(3, "ergocub_arm", MotorNormMode.RANGE_M100_100),
                "right_fingers": Motor(4, "ergocub_fingers", MotorNormMode.RANGE_M100_100),
            })
        
        # Define motor for neck
        if config.use_neck:
            motors["neck"] = Motor(5, "ergocub_neck", MotorNormMode.RANGE_M100_100)

        # Initialize the new ErgoCub motors bus
        self.bus = ErgoCubMotorsBus(
            remote_prefix=config.remote_prefix,
            local_prefix=f"{config.local_prefix}/{self.session_id}",
            motors=motors,
            calibration=None,  # No calibration needed for ErgoCub
            use_left_arm=config.use_left_arm,
            use_right_arm=config.use_right_arm,
            use_neck=config.use_neck,
        )

    def connect(self, calibrate: bool = True):
        """
        Establish communication with the robot.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
            
        # Connect cameras using standard LeRobot interface
        for cam in self.cameras.values():
            cam.connect()
        
        # Connect motor bus (arm and neck controllers)
        self.bus.connect()
        self._is_connected = True
        
        if not self.is_calibrated and calibrate:
            logger.info("ErgoCub doesn't require calibration - skipping.")
            
        self.configure()
        logger.info("%s connected.", self)

    def disconnect(self):
        """Disconnect from the robot and perform any necessary cleanup."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
            
        # Reset arm control state for safety
        self.reset_arm_control()
            
        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()
        
        # Disconnect motor bus
        self.bus.disconnect()
        self._is_connected = False
        
        logger.info("%s disconnected.", self)

    def get_observation(self) -> dict[str, Any]:
        """
        Retrieve the current observation from the robot.

        Returns:
            dict[str, Any]: A flat dictionary representing the robot's current sensory state.
            
        Raises:
            DeviceNotConnectedError: if robot is not connected.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
            
        obs = {}
        
        # Read camera data using standard LeRobot interface
        for cam_name, cam in self.cameras.items():
            cam_data = cam.read()
            if "image" in cam_data:
                obs[cam_name] = cam_data["image"]
            if "depth" in cam_data:
                obs[f"{cam_name}_depth"] = cam_data["depth"]
        
        # Read motor data (poses and finger positions) using new motor bus
        motor_data = self.bus.sync_read("Present_Position")
        obs.update(motor_data)
        
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Send an action command to the robot.

        Args:
            action (dict[str, Any]): Dictionary representing the desired action.
                Expected format for new motor system:
                - left_arm.{x,y,z,qx,qy,qz,qw}: left arm position + orientation
                - right_arm.{x,y,z,qx,qy,qz,qw}: right arm position + orientation
                - left_fingers.{thumb_add,thumb_oc,index_add,index_oc,middle_oc,ring_pinky_oc}: left hand joints
                - right_fingers.{thumb_add,thumb_oc,index_add,index_oc,middle_oc,ring_pinky_oc}: right hand joints
                - neck.{qx,qy,qz,qw}: neck orientation

        Returns:
            dict[str, Any]: The action actually sent to the robot system.
            
        Raises:
            DeviceNotConnectedError: if robot is not connected.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        # Safety check: only move if hand positions are close to current positions
        if not self.check_hand_position_safety(action):
            logger.debug("Action blocked: hand positions not within safety tolerance")
            # Return the current action without executing it
            current_positions = self.bus.sync_read("Present_Position")
            return current_positions
            
        # Validate action format matches expected action_features
        expected_keys = set(self.action_features.keys())
        received_keys = set(action.keys())
        
        # Log any missing or extra keys for debugging
        missing_keys = expected_keys - received_keys
        extra_keys = received_keys - expected_keys
        
        if missing_keys:
            logger.warning("Missing action keys: %s", missing_keys)
        if extra_keys:
            logger.warning("Extra action keys: %s", extra_keys)
        
        # Convert to dict if it's an array for compatibility
        if isinstance(action, np.ndarray):
            action_dict = {}
            idx = 0
            
            # Map array to structured action format
            for key in sorted(self.action_features.keys()):
                if idx < len(action):
                    action_dict[key] = float(action[idx])
                    idx += 1
                else:
                    # Default values: 1.0 for quaternion w components, 0.0 for others
                    action_dict[key] = 1.0 if key.endswith('.qw') else 0.0
                    
            action = action_dict
        
        # Validate and sanitize action values
        validated_action = {}
        for key, expected_type in self.action_features.items():
            if key in action:
                try:
                    # Ensure value is of correct type
                    validated_action[key] = expected_type(action[key])
                    
                    # Clamp quaternion components to valid range [-1, 1]
                    if key.endswith(('.qx', '.qy', '.qz', '.qw')):
                        validated_action[key] = max(-1.0, min(1.0, validated_action[key]))
                        
                except (ValueError, TypeError) as e:
                    logger.warning("Invalid value for %s: %s (%s). Using default.", key, action[key], e)
                    validated_action[key] = 1.0 if key.endswith('.qw') else 0.0
            else:
                # Use default values for missing keys
                validated_action[key] = 1.0 if key.endswith('.qw') else 0.0
        
        # Normalize quaternions to ensure they are valid
        quaternion_prefixes = []
        if self.config.use_neck:
            quaternion_prefixes.append('neck')
        if self.config.use_left_arm:
            quaternion_prefixes.append('left_arm')
        if self.config.use_right_arm:
            quaternion_prefixes.append('right_arm')
            
        for prefix in quaternion_prefixes:
            qx_key = f"{prefix}.qx"
            qy_key = f"{prefix}.qy" 
            qz_key = f"{prefix}.qz"
            qw_key = f"{prefix}.qw"
            
            if all(key in validated_action for key in [qx_key, qy_key, qz_key, qw_key]):
                # Normalize quaternion
                qx, qy, qz, qw = validated_action[qx_key], validated_action[qy_key], validated_action[qz_key], validated_action[qw_key]
                norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
                
                if norm > 1e-6:  # Avoid division by zero
                    validated_action[qx_key] = qx / norm
                    validated_action[qy_key] = qy / norm  
                    validated_action[qz_key] = qz / norm
                    validated_action[qw_key] = qw / norm
                else:
                    # Use identity quaternion if norm is too small
                    validated_action[qx_key] = 0.0
                    validated_action[qy_key] = 0.0
                    validated_action[qz_key] = 0.0
                    validated_action[qw_key] = 1.0
        
        # Send commands via motor bus
        self.bus.sync_write("Goal_Position", validated_action)
        
        # Log the action for debugging (first few keys only)
        sample_keys = list(validated_action.keys())[:5]
        sample_action = {k: f"{validated_action[k]:.3f}" for k in sample_keys}
        logger.debug("Sending action (sample): %s, ...", sample_action)
        
        return validated_action

    @property
    def is_connected(self) -> bool:
        """Whether the robot is currently connected."""
        # Check if cameras are connected
        cameras_connected = all(cam.is_connected for cam in self.cameras.values()) if self.cameras else True
        
        # Check if motor bus is connected
        motors_connected = self.bus.is_connected
        
        return self._is_connected and cameras_connected and motors_connected

    @property
    def is_calibrated(self) -> bool:
        """ErgoCub doesn't require calibration."""
        return True

    def calibrate(self) -> None:
        """ErgoCub doesn't require calibration - no-op."""

    def configure(self) -> None:
        """Apply any one-time configuration - no-op for ErgoCub."""

    def get_current_hand_position(self, side: str) -> np.ndarray:
        """
        Get the current position of the specified hand from the robot's sensors.
        
        Args:
            side (str): "left" or "right"
            
        Returns:
            np.ndarray: Current hand position [x, y, z] in meters
        """
        try:
            # Read current motor positions
            current_positions = self.bus.sync_read("Present_Position")
            
            # Extract position for the specified arm
            arm_key = f"{side}_arm"
            if arm_key in current_positions:
                # Return position components [x, y, z]
                return np.array([
                    current_positions.get(f"{arm_key}.x", 0.0),
                    current_positions.get(f"{arm_key}.y", 0.0), 
                    current_positions.get(f"{arm_key}.z", 0.0)
                ])
            else:
                logger.warning("No position data available for %s arm", side)
                return np.zeros(3)
                
        except (RuntimeError, AttributeError) as e:
            logger.warning("Failed to get current position for %s arm: %s", side, e)
            return np.zeros(3)

    def check_hand_position_safety(self, action: dict[str, Any]) -> bool:
        """
        Check if the target hand positions are close enough to current positions to safely move.
        Based on the safety logic from metaControllClient.
        
        Args:
            action (dict[str, Any]): Target action containing hand positions
            
        Returns:
            bool: True if it's safe to move (both hands within tolerance or controlled)
        """
        arms_to_check = []
        if self.config.use_left_arm:
            arms_to_check.append("left")
        if self.config.use_right_arm:
            arms_to_check.append("right")
            
        for side in arms_to_check:
            if not self.is_arm_controlled[side]:
                # Get current and target positions
                current_pos = self.get_current_hand_position(side)
                target_pos = np.array([
                    action.get(f"{side}_arm.x", 0.0),
                    action.get(f"{side}_arm.y", 0.0),
                    action.get(f"{side}_arm.z", 0.0)
                ])
                
                # Calculate position error
                position_error = target_pos - current_pos
                max_error = np.max(np.abs(position_error))
                
                if max_error < self.position_tolerance:
                    self.is_arm_controlled[side] = True
                    logger.info("%s arm is now controlled (error: %.3fm < %.3fm)", 
                              side.capitalize(), max_error, self.position_tolerance)
                else:
                    logger.debug("%s arm not ready: position error %.3fm > %.3fm", 
                               side.capitalize(), max_error, self.position_tolerance)
            
        # Only move if all configured arms are controlled
        controlled_arms = [side for side in arms_to_check if self.is_arm_controlled[side]]
        return len(controlled_arms) == len(arms_to_check)

    def reset_arm_control(self) -> None:
        """
        Reset the arm control state, requiring position check before movement.
        Similar to the reset functionality in metaControllClient.
        """
        self.is_arm_controlled = {"left": False, "right": False}
        logger.info("Arm control reset: hands must be repositioned within tolerance before movement")

    def set_position_tolerance(self, tolerance: float) -> None:
        """
        Set the position tolerance for safety checking.
        
        Args:
            tolerance (float): Position tolerance in meters (default 0.1m = 10cm)
        """
        self.position_tolerance = max(0.01, tolerance)  # Minimum 1cm tolerance
        logger.info("Position tolerance set to %.3fm", self.position_tolerance)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Helper property to get motor features in SO100 format."""
        motors_ft = {}
        
        # Left arm pose (7 DOF)
        if self.config.use_left_arm:
            for coord in ["x", "y", "z", "qx", "qy", "qz", "qw"]:
                motors_ft[f"left_arm.{coord}"] = float
            
            # Left fingers (6 DOF)
            for joint in ["thumb_add", "thumb_oc", "index_add", "index_oc", "middle_oc", "ring_pinky_oc"]:
                motors_ft[f"left_fingers.{joint}"] = float
        
        # Right arm pose (7 DOF)
        if self.config.use_right_arm:
            for coord in ["x", "y", "z", "qx", "qy", "qz", "qw"]:
                motors_ft[f"right_arm.{coord}"] = float
            
            # Right fingers (6 DOF)
            for joint in ["thumb_add", "thumb_oc", "index_add", "index_oc", "middle_oc", "ring_pinky_oc"]:
                motors_ft[f"right_fingers.{joint}"] = float
        
        # Neck orientation (4 DOF)
        if self.config.use_neck:
            for coord in ["qx", "qy", "qz", "qw"]:
                motors_ft[f"neck.{coord}"] = float
        
        return motors_ft

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Helper property to get camera features in SO101 format."""
        cam_features = {}
        for cam_name, cam_config in self.config.cameras.items():
            cam_features[cam_name] = (cam_config.height, cam_config.width, 3)
            if hasattr(cam_config, 'use_depth') and cam_config.use_depth:
                cam_features[f"{cam_name}_depth"] = (cam_config.height, cam_config.width)
        return cam_features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """
        A dictionary describing the structure and types of the observations produced by the robot.
        Values are either float for single values or tuples for array shapes.
        """
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """
        A dictionary describing the structure and types of the actions expected by the robot.
        ErgoCub actions are pose commands (position + orientation) for arms and neck, plus finger positions.
        
        Returns action features in SO100-like format with dot notation.
        """
        return self._motors_ft  # Actions and observations have the same structure
