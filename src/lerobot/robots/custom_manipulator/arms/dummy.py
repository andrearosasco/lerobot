from dataclasses import dataclass

from ..configs import ArmConfig


@ArmConfig.register_subclass("dummy")
@dataclass
class DummyArmConfig(ArmConfig):
    @property
    def type(self) -> str:
        return "dummy"


class DummyArm:
    def __init__(self, config: DummyArmConfig | None = None):
        self.config = config if config else DummyArmConfig()
        self._state = {key: 0.0 for key in self.features}

    def connect(self):
        pass

    def disconnect(self):
        pass

    def close(self):
        pass

    def reset(self):
        self._state = {key: 0.0 for key in self.features}

    def apply_commands(self, action=None, q_desired=None, kp=None, kd=None, gain=4.0):
        if action is None:
            return

        for key in self._state:
            action_key = f"action.{key}"
            if action_key in action:
                self._state[key] = float(action[action_key])

    @property
    def features(self) -> dict:
        pos = {f"position.{axis}": float for axis in ["x", "y", "z"]}
        ori = {f"orientation.{axis}": float for axis in ["x", "y", "z"]}
        return {**pos, **ori}

    def get_sensors(self):
        return dict(self._state)
