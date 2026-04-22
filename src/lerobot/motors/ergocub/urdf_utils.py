#!/usr/bin/env python

"""Utilities for resolving the ergoCub URDF path consistently across controllers.

Logic:
1. If environment variable ROBOT_URDF_PATH is set, expand ~, resolve relative path, and use it if it exists.
2. Otherwise fall back to YARP's ResourceFinder with the filename 'model.urdf'.

This centralizes the behavior used by arm, bimanual, and neck controllers.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yarp

logger = logging.getLogger(__name__)


def resolve_robot_urdf(
    env_vars: tuple[str, ...] = ("ROBOT_URDF_PATH",),
    fallback_filename: str = "model.urdf",
) -> str:
    """Resolve path to a cub robot URDF file.

    Args:
        env_vars: Environment variables that may point to a URDF file, in priority order.
        fallback_filename: Filename to look up via YARP ResourceFinder if env var is unset/invalid.

    Returns:
        Absolute path (string) to the URDF file (env var path if valid, else ResourceFinder result).
    """
    for env_var in env_vars:
        urdf_env = os.environ.get(env_var)
        if not urdf_env:
            continue

        candidate = Path(urdf_env).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        if candidate.exists():
            logger.info(f"Using URDF from {env_var}: {candidate}")
            return str(candidate)
        logger.warning(
            f"{env_var} is set to '{urdf_env}' but file does not exist. Falling back to YARP ResourceFinder."
        )

    # Fallback
    urdf_file = yarp.ResourceFinder().findFileByName(fallback_filename)
    logger.info(
        f"No valid URDF env var found. Using YARP ResourceFinder: {fallback_filename} -> {urdf_file}"
    )
    return urdf_file


def resolve_ergocub_urdf(env_var: str = "ROBOT_URDF_PATH", fallback_filename: str = "model.urdf") -> str:
    return resolve_robot_urdf((env_var,), fallback_filename)


__all__ = ["resolve_ergocub_urdf", "resolve_robot_urdf"]
