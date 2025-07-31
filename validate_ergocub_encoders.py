#!/usr/bin/env python

"""
ErgoCub Robot Encoder Validation Script

This script instantiates an ErgoCub robot and continuously reads from its encoders
to validate the connection and data flow.

Usage:
    python validate_ergocub_encoders.py

Configuration matches the record.sh script:
    --robot.type=ergocub \
    --robot.remote_prefix="/ergocub" \
    --robot.local_prefix="/lerobot" \
    --robot.cameras='{ agentview: {"type": "yarp", "yarp_name": "depthCamera", "width": 1280, "height": 720, "fps": 30} }' \
    --robot.encoders_control_boards='[head,left_arm,right_arm,torso]'
"""

import logging
import time
from pprint import pformat

from lerobot.cameras.yarp.configuration_yarp import YarpCameraConfig
from lerobot.robots.ergocub import ErgoCubConfig, ErgoCub
from lerobot.utils.utils import init_logging


def create_ergocub_config() -> ErgoCubConfig:
    """Create ErgoCub configuration matching the record.sh script."""
    
    # Create camera configuration
    camera_config = YarpCameraConfig(
        yarp_name="depthCamera",
        width=1280,
        height=720,
        fps=30
    )
    
    config = ErgoCubConfig(
        name="ergocub",
        remote_prefix="/ergocub",
        local_prefix="/lerobot", 
        cameras={"agentview": camera_config},
        use_left_arm=True,
        use_right_arm=True,
        use_neck=True,
        encoders_control_boards=["head", "left_arm", "right_arm", "torso"]
    )
    
    return config


def validate_encoders():
    """Main validation function that connects to robot and reads encoders."""
    
    init_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting ErgoCub encoder validation...")
    
    # Create robot configuration
    config = create_ergocub_config()
    logger.info("Robot configuration:")
    logger.info(pformat(config.__dict__))
    
    # Create robot instance
    robot = ErgoCub(config)
    logger.info(f"Created robot: {robot}")
    
    try:
        # Connect to robot
        logger.info("Connecting to robot...")
        robot.connect(calibrate=False)  # Skip calibration for validation
        logger.info("Robot connected successfully!")
        
        # Read encoders continuously
        logger.info("Starting encoder reading loop...")
        logger.info("Press Ctrl+C to stop...")
        
        read_count = 0
        start_time = time.time()
        
        while True:
            try:
                # Get observation (includes encoder data)
                observation = robot.get_observation()
                read_count += 1
                
                # Print summary every 10 reads
                if read_count % 10 == 0:
                    elapsed_time = time.time() - start_time
                    read_rate = read_count / elapsed_time
                    
                    logger.info(f"Read #{read_count} (Rate: {read_rate:.1f} Hz)")
                    
                    # Print sample of encoder data
                    encoder_keys = [k for k in observation.keys() if not k.startswith('agentview')]
                    if encoder_keys:
                        logger.info("Sample encoder data:")
                        for i, key in enumerate(encoder_keys):  # Show first 8 encoder values
                            value = observation[key]
                            logger.info(f"  {key}: {value:.4f}")
                    
                    # Print camera data info if available
                    camera_keys = [k for k in observation.keys() if k.startswith('agentview')]
                    if camera_keys:
                        logger.info("Camera data:")
                        for key in camera_keys:
                            shape = observation[key].shape if hasattr(observation[key], 'shape') else "N/A"
                            logger.info(f"  {key}: shape={shape}")
                
                # Small delay to avoid overwhelming the system
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, stopping...")
                break
                
    except Exception as e:
        logger.error(f"Failed to connect or operate robot: {e}")
        return False
        
    finally:
        # Disconnect robot
        try:
            logger.info("Disconnecting robot...")
            robot.disconnect()
            logger.info("Robot disconnected successfully!")
        except Exception as e:
            logger.error(f"Error disconnecting robot: {e}")
    
    logger.info("ErgoCub encoder validation completed.")
    return True


def main():
    """Entry point for the validation script."""
    success = validate_encoders()
    if success:
        print("✅ ErgoCub encoder validation completed successfully!")
    else:
        print("❌ ErgoCub encoder validation failed!")
        exit(1)


if __name__ == "__main__":
    main()
