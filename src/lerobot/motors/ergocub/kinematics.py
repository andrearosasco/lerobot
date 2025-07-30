#!/usr/bin/env python

"""
Direct kinematics module for ErgoCub using placo library.

This module implements functions to compute end-effector poses 
from joint angles for the ErgoCub robot using the placo kinematics library.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
import logging
import yarp
import os

if TYPE_CHECKING:
    from lerobot.model.kinematics import RobotKinematics

logger = logging.getLogger(__name__)

# Global variables for kinematics solvers (initialized on first use)
_left_arm_kinematics: Optional['RobotKinematics'] = None
_right_arm_kinematics: Optional['RobotKinematics'] = None
_neck_kinematics: Optional['RobotKinematics'] = None

# URDF path will be found using YARP ResourceFinder
_urdf_path: Optional[str] = None


def _get_ergocub_urdf_path() -> str:
    """
    Get ErgoCub URDF path using YARP ResourceFinder.
    
    This function uses YARP's ResourceFinder to locate the URDF file
    automatically based on the robot configuration.
    
    Returns:
        str: Path to the ErgoCub URDF file
        
    Raises:
        FileNotFoundError: If URDF file cannot be found
    """
    global _urdf_path
    
    if _urdf_path is not None:
        return _urdf_path
    
    # Initialize YARP if not already done
    if not yarp.Network.checkNetwork():
        logger.warning("YARP network not available, initializing...")
        yarp.Network.init()
    
    # Set robot name environment variable if not set
    if "YARP_ROBOT_NAME" not in os.environ:
        os.environ["YARP_ROBOT_NAME"] = "ergoCubSN002"
        logger.info("Set YARP_ROBOT_NAME to ergoCubSN002")
    
    urdf_file = yarp.ResourceFinder().findFileByName("model.urdf")
    if not urdf_file:
        raise FileNotFoundError(
            "Could not find model.urdf using YARP ResourceFinder. "
            "Please ensure YARP_ROBOT_NAME is set correctly and robot files are installed."
        )
    
    _urdf_path = urdf_file
    logger.info(f"Found ErgoCub URDF at: {_urdf_path}")
    return _urdf_path


def _get_arm_kinematics(arm_side: str) -> 'RobotKinematics':
    """Get or initialize arm kinematics solver."""
    global _left_arm_kinematics, _right_arm_kinematics
    
    try:
        from lerobot.model.kinematics import RobotKinematics
    except ImportError as e:
        logger.error("Cannot import RobotKinematics. Please install placo library.")
        raise ImportError("placo library required for ErgoCub kinematics") from e
    
    if arm_side == "left":
        if _left_arm_kinematics is None:
            urdf_path = _get_ergocub_urdf_path()
            # Define left arm joint names for ErgoCub
            joint_names = [
                "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw",
                "l_elbow", "l_wrist_yaw", "l_wrist_roll", "l_wrist_pitch"
            ]
            _left_arm_kinematics = RobotKinematics(
                urdf_path=urdf_path,
                target_frame_name="l_hand_palm",  # Left hand end-effector frame
                joint_names=joint_names
            )
        return _left_arm_kinematics
    
    elif arm_side == "right":
        if _right_arm_kinematics is None:
            urdf_path = _get_ergocub_urdf_path()
            # Define right arm joint names for ErgoCub
            joint_names = [
                "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw",
                "r_elbow", "r_wrist_yaw", "r_wrist_roll", "r_wrist_pitch"
            ]
            _right_arm_kinematics = RobotKinematics(
                urdf_path=urdf_path,
                target_frame_name="r_hand_palm",  # Right hand end-effector frame
                joint_names=joint_names
            )
        return _right_arm_kinematics
    
    else:
        raise ValueError(f"Invalid arm_side: {arm_side}. Must be 'left' or 'right'")


def _get_neck_kinematics() -> 'RobotKinematics':
    """Get or initialize neck kinematics solver."""
    global _neck_kinematics
    
    try:
        from lerobot.model.kinematics import RobotKinematics
    except ImportError as e:
        logger.error("Cannot import RobotKinematics. Please install placo library.")
        raise ImportError("placo library required for ErgoCub kinematics") from e
    
    if _neck_kinematics is None:
        urdf_path = _get_ergocub_urdf_path()
        # Define neck joint names for ErgoCub
        joint_names = ["neck_pitch", "neck_roll", "neck_yaw"]
        _neck_kinematics = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name="head",  # Head end-effector frame
            joint_names=joint_names
        )
    return _neck_kinematics


def compute_arm_pose_from_joints(
    joint_positions: List[float], 
    arm_side: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute end-effector pose from joint positions for ErgoCub arm.
    
    Args:
        joint_positions: List of joint angles in radians [shoulder_pitch, shoulder_roll, 
                        shoulder_yaw, elbow, wrist_prosup, wrist_pitch, wrist_yaw]
        arm_side: "left" or "right"
        
    Returns:
        Tuple of (position, orientation):
        - position: 3D position vector [x, y, z] in meters
        - orientation: 3x3 rotation matrix or quaternion [x, y, z, w]
        
    Raises:
        ValueError: If arm_side is not 'left' or 'right'
        ImportError: If placo library is not available
    """
    if len(joint_positions) != 7:
        raise ValueError(f"Expected 7 joint positions for arm, got {len(joint_positions)}")
    
    # Get kinematics solver for the specified arm
    kinematics = _get_arm_kinematics(arm_side)
    
    # Compute forward kinematics
    T = kinematics.forward_kinematics(joint_positions)
    
    # Extract position and orientation from transformation matrix
    position = T[:3, 3]  # Translation vector
    rotation_matrix = T[:3, :3]  # Rotation matrix
    
    # Convert rotation matrix to quaternion (x, y, z, w)
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    
    return position, quaternion


def compute_neck_orientation_from_joints(joint_positions: List[float]) -> np.ndarray:
    """
    Compute head orientation from neck joint positions for ErgoCub.
    
    Args:
        joint_positions: List of neck joint angles in radians [neck_pitch, neck_roll, neck_yaw]
        
    Returns:
        3x3 rotation matrix representing head orientation
        
    Raises:
        ImportError: If placo library is not available
    """
    if len(joint_positions) != 3:
        raise ValueError(f"Expected 3 neck joint positions, got {len(joint_positions)}")
    
    # Get neck kinematics solver
    kinematics = _get_neck_kinematics()
    
    # Compute forward kinematics
    T = kinematics.forward_kinematics(joint_positions)
    
    # Extract rotation matrix
    rotation_matrix = T[:3, :3]
    
    return rotation_matrix


def compute_neck_quaternion_from_joints(joint_positions: List[float]) -> np.ndarray:
    """
    Compute head orientation as quaternion from neck joint positions for ErgoCub.
    
    Args:
        joint_positions: List of neck joint angles in radians [neck_pitch, neck_roll, neck_yaw]
        
    Returns:
        Quaternion as numpy array [qx, qy, qz, qw]
        
    Raises:
        ImportError: If placo library is not available
    """
    # Get rotation matrix
    rotation_matrix = compute_neck_orientation_from_joints(joint_positions)
    
    # Convert to quaternion
    return rotation_matrix_to_quaternion(rotation_matrix)


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion.
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Quaternion as numpy array [x, y, z, w]
    """
    # Shepperd's method for numerical stability
    trace = np.trace(R)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([x, y, z, w])


# Example usage and testing functions
def test_kinematics():
    """Test function to verify kinematics computation."""
    logger.info("Testing ErgoCub kinematics...")
    
    # Test left arm kinematics
    try:
        left_arm_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Zero configuration
        pos, quat = compute_arm_pose_from_joints(left_arm_joints, "left")
        logger.info(f"Left arm zero config - Position: {pos}, Quaternion: {quat}")
        
        # Test right arm kinematics  
        right_arm_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Zero configuration
        pos, quat = compute_arm_pose_from_joints(right_arm_joints, "right")
        logger.info(f"Right arm zero config - Position: {pos}, Quaternion: {quat}")
        
        # Test neck kinematics
        neck_joints = [0.0, 0.0, 0.0]  # Zero configuration
        rotation = compute_neck_orientation_from_joints(neck_joints)
        logger.info(f"Neck zero config - Rotation matrix:\n{rotation}")
        
        logger.info("Kinematics test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Kinematics test failed: {e}")
        return False


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_kinematics()
