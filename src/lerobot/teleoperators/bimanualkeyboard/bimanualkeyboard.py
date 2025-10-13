#!/usr/bin/env python

from typing import Any

from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.teleoperators.teleoperator import Teleoperator

from .configuration_bimanualkeyboard import BimanualKeyboardConfig


class BimanualKeyboard(Teleoperator):
    config_class = BimanualKeyboardConfig
    name = "bimanualkeyboard"

    def __init__(self, cfg: BimanualKeyboardConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self._connected = False
        # key bindings
        self.left_keys = {
            "a": (-1, 0, 0),
            "d": (1, 0, 0),
            "s": (0, -1, 0),
            "w": (0, 1, 0),
            "q": (0, 0, -1),
            "e": (0, 0, 1),
        }
        self.right_keys = {
            "j": (-1, 0, 0),
            "l": (1, 0, 0),
            "k": (0, -1, 0),
            "i": (0, 1, 0),
            "u": (0, 0, -1),
            "o": (0, 0, 1),
        }
        self._pressed: dict[str, bool] = {}

    @property
    def action_features(self) -> dict:
        return {
            # head
            "head.position.x": float,
            "head.position.y": float,
            "head.position.z": float,
            "head.orientation.qw": float,
            "head.orientation.qx": float,
            "head.orientation.qy": float,
            "head.orientation.qz": float,
            # left hand
            "left_arm.position.x": float,
            "left_arm.position.y": float,
            "left_arm.position.z": float,
            "left_arm.orientation.qw": float,
            "left_arm.orientation.qx": float,
            "left_arm.orientation.qy": float,
            "left_arm.orientation.qz": float,
            # right hand
            "right_arm.position.x": float,
            "right_arm.position.y": float,
            "right_arm.position.z": float,
            "right_arm.orientation.qw": float,
            "right_arm.orientation.qx": float,
            "right_arm.orientation.qy": float,
            "right_arm.orientation.qz": float,
            # fingers
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
        self._connected = True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _read_pressed(self) -> dict[str, bool]:
        return self._pressed

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        pressed = self._read_pressed()
        step = float(self.cfg.step)
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

        vals = [
            # head
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            # left hand
            lx, ly, lz, 1.0, 0.0, 0.0, 0.0,
            # right hand
            rx, ry, rz, 1.0, 0.0, 0.0, 0.0,
            # left fingers (zeros)
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            # right fingers (zeros)
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
