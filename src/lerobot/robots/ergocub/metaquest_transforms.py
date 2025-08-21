#!/usr/bin/env python

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict, Any

# Fixed transforms from config (hardcoded for simplicity)
HEAD_I_ROOTLINK = np.array([
    [1.0, 0.0, 0.0, 0.005],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.547],
    [0.0, 0.0, 0.0, 1.0]
])

RHS_WO_HEAD_I_TRANSF = np.array([
    [0.0, 0.0, -1.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

LEFT_HAND_ADAPTER = np.eye(4)

RIGHT_HAND_ADAPTER = np.array([
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

HEAD_ADAPTER = np.array([
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])


def transform_metaquest_to_ergocub(action: Dict[str, Any]) -> Dict[str, Any]:
    """Transform MetaQuest action to ErgoCub coordinates.
    
    If action contains MetaQuest format keys (e.g. left_hand.position.x), transform them.
    If action contains robot format keys (e.g. left_arm.position.x), pass them through unchanged.
    """
    result = {}
    
    # Check if this is MetaQuest format or robot format
    has_metaquest_keys = any(key.startswith(('left_hand.', 'right_hand.', 'head.')) for key in action.keys())
    has_robot_keys = any(key.startswith(('left_arm.', 'right_arm.', 'neck.')) for key in action.keys())
    
    if has_robot_keys and not has_metaquest_keys:
        # Already in robot format - pass through unchanged
        return action.copy()
    
    # Transform hands from MetaQuest format
    for side in ['left', 'right']:
        # Check if hand data exists
        pos_keys = [f'{side}_hand.position.x', f'{side}_hand.position.y', f'{side}_hand.position.z']
        quat_keys = [f'{side}_hand.orientation.qx', f'{side}_hand.orientation.qy', 
                     f'{side}_hand.orientation.qz', f'{side}_hand.orientation.qw']
        
        if all(key in action for key in pos_keys + quat_keys):
            # Construct transform matrix from position and quaternion
            pos = np.array([action[key] for key in pos_keys])
            quat = np.array([action[key] for key in quat_keys])
            
            hand_transform = np.eye(4)
            hand_transform[:3, 3] = pos
            hand_transform[:3, :3] = R.from_quat(quat).as_matrix()
            
            adapter = LEFT_HAND_ADAPTER if side == 'left' else RIGHT_HAND_ADAPTER
            T = HEAD_I_ROOTLINK @ RHS_WO_HEAD_I_TRANSF @ hand_transform @ adapter
            pos, quat = T[:3, 3], R.from_matrix(T[:3, :3]).as_quat()
            result.update({
                f'{side}_arm.position.x': pos[0], f'{side}_arm.position.y': pos[1], f'{side}_arm.position.z': pos[2],
                f'{side}_arm.orientation.qx': quat[0], f'{side}_arm.orientation.qy': quat[1], 
                f'{side}_arm.orientation.qz': quat[2], f'{side}_arm.orientation.qw': quat[3]
            })
    
    # Transform head
    head_quat_keys = ['head.orientation.qx', 'head.orientation.qy', 'head.orientation.qz', 'head.orientation.qw']
    if all(key in action for key in head_quat_keys):
        head_quat = np.array([action[key] for key in head_quat_keys])
        oc_wrt_wo = np.eye(4)
        oc_wrt_wo[:3, :3] = R.from_quat(head_quat).as_matrix()
        T = HEAD_I_ROOTLINK @ RHS_WO_HEAD_I_TRANSF @ oc_wrt_wo @ HEAD_ADAPTER
        quat = R.from_matrix(T[:3, :3]).as_quat()
        result.update({
            'neck.orientation.qx': quat[0], 'neck.orientation.qy': quat[1], 
            'neck.orientation.qz': quat[2], 'neck.orientation.qw': quat[3]
        })
    
    # Copy fingers
    for side in ['left', 'right']:
        finger_keys = [k for k in action.keys() if k.startswith(f'{side}_fingers.')]
        for key in finger_keys:
            result[key] = action[key]
    
    return result

