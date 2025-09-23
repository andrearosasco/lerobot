#!/usr/bin/env python

# Keyboard-based teleoperator that mimics MetaQuest action schema with relative translations only.

from typing import Any

from lerobot.errors import DeviceNotConnectedError
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.control_utils import is_headless

from .configuration_metaquest_keyboard import BimanualKeyboardConfig


class BimanualKeyboard(Teleoperator):
    config_class = BimanualKeyboardConfig
    name = "bimanualkeyboard"

    def __init__(self, cfg: BimanualKeyboardConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self._connected = False

        # Key bindings: left hand (a,s,d,w,q,e) and right hand (j,k,l,i,u,o)
        self.left_keys = {
            "a": (-1, 0, 0),  # -x
            "d": (1, 0, 0),   # +x
            "s": (0, -1, 0),  # -y
            "w": (0, 1, 0),   # +y
            "q": (0, 0, -1),  # -z
            "e": (0, 0, 1),   # +z
        }
        self.right_keys = {
            "j": (-1, 0, 0),
            "l": (1, 0, 0),
            "k": (0, -1, 0),
            "i": (0, 1, 0),
            "u": (0, 0, -1),
            "o": (0, 0, 1),
        }

        # Current pressed map is filled by record.py global keyboard listener; we poll from there
        self._pressed: dict[str, bool] = {}

    # Schema mirrors MetaQuest: head pose, hands pose, fingers positions
    @property
    def action_features(self) -> dict:
        return {
            # Head pose (zeros)
            "head.position.x": float,
            "head.position.y": float,
            "head.position.z": float,
            "head.orientation.qw": float,
            "head.orientation.qx": float,
            "head.orientation.qy": float,
            "head.orientation.qz": float,
            # Left hand pose
            "left_hand.position.x": float,
            "left_hand.position.y": float,
            "left_hand.position.z": float,
            "left_hand.orientation.qw": float,
            "left_hand.orientation.qx": float,
            "left_hand.orientation.qy": float,
            "left_hand.orientation.qz": float,
            # Right hand pose
            "right_hand.position.x": float,
            "right_hand.position.y": float,
            "right_hand.position.z": float,
            "right_hand.orientation.qw": float,
            "right_hand.orientation.qx": float,
            "right_hand.orientation.qy": float,
            "right_hand.orientation.qz": float,
            # Left finger positions (zeros)
            "left_fingers.thumb.x": float,
            "left_fingers.thumb.y": float,
            "left_fingers.thumb.z": float,
            "left_fingers.index.x": float,
            "left_fingers.index.y": float,
            "left_fingers.index.z": float,
            "left_fingers.middle.x": float,
            "left_fingers.middle.y": float,
            "left_fingers.middle.z": float,
            "left_fingers.ring.x": float,
            "left_fingers.ring.y": float,
            "left_fingers.ring.z": float,
            "left_fingers.pinky.x": float,
            "left_fingers.pinky.y": float,
            "left_fingers.pinky.z": float,
            # Right finger positions (zeros)
            "right_fingers.thumb.x": float,
            "right_fingers.thumb.y": float,
            "right_fingers.thumb.z": float,
            "right_fingers.index.x": float,
            "right_fingers.index.y": float,
            "right_fingers.index.z": float,
            "right_fingers.middle.x": float,
            "right_fingers.middle.y": float,
            "right_fingers.middle.z": float,
            "right_fingers.ring.x": float,
            "right_fingers.ring.y": float,
            "right_fingers.ring.z": float,
            "right_fingers.pinky.x": float,
            "right_fingers.pinky.y": float,
            "right_fingers.pinky.z": float,
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        # Do not create our own listener; record.py already starts one. Just mark connected.
        # If headless, still allow connection; get_action will just return zeros unless keys are injected.
        self._connected = True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _read_pressed(self) -> dict[str, bool]:
        # We avoid importing pynput or starting a listener to prevent conflicts with record.py.
        # Instead, allow external code/tests to set self._pressed directly between calls.
        return self._pressed

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        pressed = self._read_pressed()
        step = float(self.cfg.step)

        # Compute relative translations from key states
        lx = ly = lz = 0.0
        rx = ry = rz = 0.0

        for key, (dx, dy, dz) in self.left_keys.items():
            if pressed.get(key, False):
                lx += dx * step
                ly += dy * step
                lz += dz * step
        for key, (dx, dy, dz) in self.right_keys.items():
            if pressed.get(key, False):
                rx += dx * step
                ry += dy * step
                rz += dz * step

        # Rotations are zero; finger positions zero. Head zeros too.
        # Order must match action_features keys
        vals = [
            # head (pos + quat)
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            # left hand (pos + quat)
            lx, ly, lz, 1.0, 0.0, 0.0, 0.0,
            # right hand (pos + quat)
            rx, ry, rz, 1.0, 0.0, 0.0, 0.0,
            # left fingers (5 tips x 3)
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            # right fingers
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]

        keys = list(self.action_features.keys())
        return dict(zip(keys, vals))

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        self._connected = False
