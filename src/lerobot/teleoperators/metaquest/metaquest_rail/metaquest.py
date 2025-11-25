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

import time
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.teleoperators.teleoperator import Teleoperator
from .configuration_metaquest import MetaQuestRailConfig

try:
    from oculus_reader import OculusReader
except ImportError:
    raise ImportError("(Missing oculus_reader. HINT: Install and perform the setup instructions from https://github.com/rail-berkeley/oculus_reader)")

# VR ==> MJ mapping when teleOp user is behind the robot
def vrbehind2mj(pose):
    pos = np.zeros([3])
    pos[0] = +1.*pose[2][3]
    pos[1] = +1.*pose[0][3]
    pos[2] = +1.*pose[1][3]

    mat = np.zeros([3, 3])
    mat[0][:] = +1.*pose[2][:3]
    mat[1][:] = -1.*pose[0][:3]
    mat[2][:] = -1.*pose[1][:3]

    # Convert matrix to axis-angle
    r = R.from_matrix(mat)
    axis_angle = r.as_rotvec()
    return pos, axis_angle

class MetaQuestRail(Teleoperator):
    # Required class attributes for LeRobot compatibility
    config_class = MetaQuestRailConfig
    name = "metaquest_rail"
    
    def __init__(self, cfg: MetaQuestRailConfig):
        self.cfg = cfg
        self._is_connected = False
        self.oculus_reader = None
        
        # Call parent constructor after setting cfg
        super().__init__(cfg)

    def connect(self):
        """Connect to OculusReader."""
        print('Waiting for Oculus', end='')
        self.oculus_reader = OculusReader()
        
        oculus_reader_ready = False
        while not oculus_reader_ready:
            # Get the controller and headset positions and the button being pushed
            transformations, buttons = self.oculus_reader.get_transformations_and_buttons()
            if transformations or buttons:
                print('')
                oculus_reader_ready = True
            else:
                #  track both the current character and its index for easier backtracking later
                for index, char in enumerate('...'):
                    sys.stdout.write(char)  # write the next char to STDOUT
                    sys.stdout.flush()  # flush the output
                    time.sleep(0.3)  # wait to match our speed
                index += 1  # lists are zero indexed, we need to increase by one for the accurate count
                # backtrack the written characters, overwrite them with space, backtrack again:
                sys.stdout.write("\b" * index + " " * index + "\b" * index)
                sys.stdout.flush()  # flush the output

        print('Oculus Ready!')
        self._is_connected = True

    def disconnect(self):
        """Disconnect from OculusReader."""
        self.oculus_reader = None
        self._is_connected = False

    def get_action(self) -> dict[str, Any]:
        """
        Get raw MetaQuest poses transformed to MJ frame.
        Returns one cartesian position and a gripper state.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        transformations, buttons = self.oculus_reader.get_transformations_and_buttons()
        
        # Wait for right controller
        while not transformations or 'r' not in transformations:
            transformations, buttons = self.oculus_reader.get_transformations_and_buttons()
            # Add a small sleep to avoid busy loop if needed, but oculus_reader might be blocking or fast enough
            # time.sleep(0.001) 

        vr_pos, vr_axis_angle = vrbehind2mj(transformations['r'])
        
        # Gripper state from right trigger (0.0 to 1.0)
        # Assuming 'rightTrig' is the key for the right trigger
        gripper_state = buttons.get('rightTrig', [0.0])[0] if isinstance(buttons.get('rightTrig'), (list, tuple)) else buttons.get('rightTrig', 0.0)
        
        # Get RG button state (Right Grip/Grab)
        rg_state = buttons.get('RG', False)
        
        # Get A button state for exit
        a_state = buttons.get('A', False)

        # Get B button state for discard
        b_state = buttons.get('B', False)
        
        # Construct action dictionary
        action = {
            "position.x": vr_pos[0],
            "position.y": vr_pos[1],
            "position.z": vr_pos[2],
            "orientation.x": vr_axis_angle[0],
            "orientation.y": vr_axis_angle[1],
            "orientation.z": vr_axis_angle[2],
            "gripper": gripper_state,
            "is_engaged": rg_state,
            "exit_episode": a_state,
            "discard_episode": b_state
        }
        
        return action

    @property
    def action_features(self):
        """
        Return action features for MetaQuest Rail.
        Provides right hand position/orientation and gripper state.
        """
        return {
            # Right hand pose (mapped to MJ frame)
            "position.x": float,
            "position.y": float,
            "position.z": float,
            "orientation.x": float,
            "orientation.y": float,
            "orientation.z": float,
            
            # Gripper
            "gripper": float,
            
            # Engagement state
            "is_engaged": bool,
            
            # Exit signal
            "exit_episode": bool,
            
            # Discard signal
            "discard_episode": bool,
        }

    @property
    def feedback_features(self) -> dict:
        """MetaQuest teleop doesn't provide feedback features."""
        return {}

    @property
    def is_connected(self) -> bool:
        """Whether the teleoperator is currently connected."""
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        """MetaQuest teleop doesn't require calibration."""
        return True

    def calibrate(self) -> None:
        """MetaQuest teleop doesn't require calibration - no-op."""
        pass

    def configure(self) -> None:
        """Apply any one-time configuration - no-op for MetaQuest teleop."""
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """MetaQuest teleop doesn't support feedback - no-op."""
        pass
