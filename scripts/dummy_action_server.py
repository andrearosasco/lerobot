#!/usr/bin/env python3
"""
Dummy Action Server for MetaCub Dashboard Testing

This script creates a YARP server that streams dummy action data in the format
expected by the ActionInterface. It publishes pose data for neck, arms, and fingers
through YARP port "/metaControllClient/action:o".

Usage:
    python dummy_action_server.py [--frequency FREQ] [--port-prefix PREFIX]

The dummy data includes:
- Neck: Rotating 3x3 rotation matrix
- Arms: Sinusoidal position and rotating orientation 
- Fingers: Individual finger positions with slight variations

This allows testing the MetaCub Dashboard ActionInterface without needing
the actual robot control system.
"""

import yarp
import time
import math
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R


class DummyActionServer:
    """Streams dummy action data through YARP in ActionInterface format."""
    
    def __init__(self, port_prefix="/metaControllClient", frequency=10.0):
        """
        Initialize the dummy action server.
        
        Args:
            port_prefix: YARP port prefix (default: "/metaControllClient")
            frequency: Publishing frequency in Hz (default: 30.0)
        """
        self.port_prefix = port_prefix
        self.frequency = frequency
        self.period = 1.0 / frequency
        
        # Initialize YARP
        yarp.Network.init()
        
        # Create output port
        self.port = yarp.BufferedPortBottle()
        self.port_name = f"{port_prefix}/action:o"
        self.port.open(self.port_name)
        
        # Create reset input port
        self.reset_port = yarp.BufferedPortBottle()
        self.reset_port_name = f"{port_prefix}/reset:i"
        self.reset_port.open(self.reset_port_name)
        
        print(f"Dummy action server started on port: {self.port_name}")
        print(f"Reset port available at: {self.reset_port_name}")
        print(f"Publishing at {frequency} Hz")
        print("Connect your ActionInterface to this port to receive data")
        
        # State for generating smooth animations
        self.time_start = time.time()
        self.seq_number = 0
        self.stamp = yarp.Stamp()
        
    def generate_neck_pose(self, t):
        """Generate rotating neck pose (3x3 rotation matrix)."""
        # Rotate around Z-axis over time
        angle = 0.5 * math.sin(0.3 * t)  # Slow neck rotation
        rotation = R.from_euler('z', angle)
        return rotation.as_matrix().flatten().tolist()
    
    def generate_arm_pose(self, t, is_left=True):
        """Generate arm pose (3 position + 4 quaternion)."""
        # Sinusoidal motion for position
        phase = 0 if is_left else math.pi  # Offset for left/right arms
        
        x = 0.3 + 0.1 * math.sin(0.5 * t + phase)
        y = 0.2 * (1 if is_left else -1) + 0.05 * math.cos(0.7 * t + phase)
        z = 0.1 + 0.03 * math.sin(0.9 * t + phase)
        
        # Rotating orientation
        roll = 0.1 * math.sin(0.4 * t + phase)
        pitch = 0.1 * math.cos(0.6 * t + phase)
        yaw = 0.2 * math.sin(0.3 * t + phase)
        
        rotation = R.from_euler('xyz', [roll, pitch, yaw])
        quat = rotation.as_quat()  # [x, y, z, w]
        
        return [x, y, z] + quat.tolist()
    
    def generate_finger_poses(self, t):
        """Generate finger poses (10 fingers × 3 positions each)."""
        finger_names = ['l_thumb', 'l_index', 'l_middle', 'l_ring', 'l_pinky',
                       'r_thumb', 'r_index', 'r_middle', 'r_ring', 'r_pinky']
        
        finger_poses = []
        for i, name in enumerate(finger_names):
            # Each finger moves slightly differently
            phase = i * 0.2
            side_mult = 1 if name.startswith('l_') else -1
            
            x = 0.02 + 0.005 * math.sin(0.8 * t + phase)
            y = side_mult * (0.01 + 0.003 * math.cos(0.6 * t + phase))
            z = 0.005 * math.sin(1.2 * t + phase)
            
            finger_poses.append([x, y, z])
        
        return finger_poses
    
    def check_reset(self):
        """Check for reset signals and acknowledge them."""
        bottle = self.reset_port.read(False)
        if bottle is not None:
            reset_value = bottle.get(0).asInt32()
            if reset_value == 1:
                print("🔄 Reset signal received - acknowledged")
                # Could reset internal state here if needed
                # For now, we just acknowledge the signal
                return True
        return False
    
    def fill_bottle(self, bottle, t):
        """Fill YARP bottle with all pose data."""
        bottle.clear()
        
        # Add neck data
        neck_bottle = bottle.addList()
        neck_bottle.addString("neck")
        neck_data = neck_bottle.addList()
        for val in self.generate_neck_pose(t):
            neck_data.addFloat64(val)
        
        # Add left arm data
        left_arm_bottle = bottle.addList()
        left_arm_bottle.addString("left_arm")
        left_arm_data = left_arm_bottle.addList()
        for val in self.generate_arm_pose(t, is_left=True):
            left_arm_data.addFloat64(val)
        
        # Add right arm data
        right_arm_bottle = bottle.addList()
        right_arm_bottle.addString("right_arm")
        right_arm_data = right_arm_bottle.addList()
        for val in self.generate_arm_pose(t, is_left=False):
            right_arm_data.addFloat64(val)
        
        # Add finger data
        fingers_bottle = bottle.addList()
        fingers_bottle.addString("fingers")
        fingers_data = fingers_bottle.addList()
        for finger_pos in self.generate_finger_poses(t):
            finger_bottle = fingers_data.addList()
            for val in finger_pos:
                finger_bottle.addFloat64(val)
    
    def run(self):
        """Main loop - publish data at specified frequency."""
        print("Starting to publish dummy action data...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                start_time = time.time()
                
                # Check for reset signals
                self.check_reset()
                
                # Get current time
                t = time.time() - self.time_start
                
                # Create and send bottle
                bottle = self.port.prepare()
                self.fill_bottle(bottle, t)
                
                # Set timestamp and sequence number
                self.stamp.update(yarp.now())
                
                self.port.setEnvelope(self.stamp)
                self.port.write()
                
                self.seq_number += 1
                
                # Maintain frequency
                elapsed = time.time() - start_time
                sleep_time = max(0, self.period - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Print status every second
                if self.seq_number % int(self.frequency) == 0:
                    print(f"Published {self.seq_number} messages, t={t:.1f}s")
                    
        except KeyboardInterrupt:
            print("\nShutting down dummy action server...")
        finally:
            self.close()
    
    def close(self):
        """Clean shutdown."""
        self.port.close()
        self.reset_port.close()
        yarp.Network.fini()
        print("Dummy action server stopped.")


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Dummy Action Server for MetaCub Dashboard Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dummy_action_server.py                    # Default settings
  python dummy_action_server.py --frequency 60     # 60 Hz publishing
  python dummy_action_server.py --port-prefix /test # Custom port prefix

The server publishes pose data for:
- neck: 3x3 rotation matrix (9 floats)
- left_arm, right_arm: position + quaternion (7 floats each)  
- fingers: 10 finger positions (3 floats each)

Connect ActionInterface with remote_prefix matching the port-prefix.
        """
    )
    
    parser.add_argument(
        "--frequency", "-f",
        type=float,
        default=10.0,
        help="Publishing frequency in Hz (default: 30.0)"
    )
    
    parser.add_argument(
        "--port-prefix", "-p",
        type=str,
        default="/metaControllClient",
        help="YARP port prefix (default: /metaControllClient)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.frequency <= 0:
        print("Error: Frequency must be positive")
        return 1
    
    if not args.port_prefix.startswith('/'):
        print("Error: Port prefix must start with '/'")
        return 1
        
    if args.port_prefix.endswith('/'):
        print("Error: Port prefix must not end with '/'")
        return 1
    
    # Create and run server
    server = DummyActionServer(
        port_prefix=args.port_prefix,
        frequency=args.frequency
    )
    
    server.run()
    return 0


if __name__ == "__main__":
    exit(main())
