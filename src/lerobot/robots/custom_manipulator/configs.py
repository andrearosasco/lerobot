import draccus
from dataclasses import dataclass
import abc

@dataclass
class ArmConfig(draccus.ChoiceRegistry, abc.ABC):
    pass

@dataclass
class GripperConfig(draccus.ChoiceRegistry, abc.ABC):
    pass
