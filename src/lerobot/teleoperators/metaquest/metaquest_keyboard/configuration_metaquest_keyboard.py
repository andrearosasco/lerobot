#!/usr/bin/env python

# Copyright 2025

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bimanualkeyboard")
@dataclass
class BimanualKeyboardConfig(TeleoperatorConfig):
    # Step size applied per tick for relative xyz deltas (meters)
    step: float = 0.01
