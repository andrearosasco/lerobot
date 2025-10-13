#!/usr/bin/env python

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bimanualkeyboard")
@dataclass
class BimanualKeyboardConfig(TeleoperatorConfig):
    # Step size per tick for relative xyz deltas (meters)
    step: float = 0.01
