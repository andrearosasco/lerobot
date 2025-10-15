#!/usr/bin/env python

"""
Evaluate a trained policy on the ErgoCub robot in continuous mode.

Usage:
    python evaluate_ergocub_policy.py --policy_path=steb6/ergocub-pick-mustard

This script will:
1. Load your trained policy from Hugging Face
2. Connect to the ErgoCub robot
3. Run policy inference continuously without saving data
"""

import argparse
import logging
import time
import numpy as np
from pathlib import Path

from lerobot.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.robots.ergocub import ErgoCubConfig, ErgoCub
from lerobot.cameras.yarp import YarpCameraConfig
from lerobot.utils.control_utils import init_keyboard_listener, predict_action
from lerobot.utils.utils import log_say, get_safe_torch_device
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data


def main():
    parser = argparse.ArgumentParser(
        description="Run policy on ErgoCub robot continuously"
    )
    parser.add_argument(
        "--policy_path",
        type=str,
        required=True,
        help="Path or Hugging Face repo ID of the trained policy (e.g., steb6/ergocub-pick-mustard)",
    )
    parser.add_argument("--fps", type=int, default=10, help="Control frequency in Hz")
    parser.add_argument(
        "--task_description",
        type=str,
        default="Policy evaluation",
        help="Task description for the evaluation",
    )
    parser.add_argument(
        "--remote_prefix",
        type=str,
        default="/ergocub",
        help="Remote prefix for ErgoCub robot",
    )
    parser.add_argument(
        "--local_prefix",
        type=str,
        default="/lerobot",
        help="Local prefix for ErgoCub robot",
    )
    parser.add_argument(
        "--noise_scheduler_type",
        type=str,
        default=None,
        help="Noise scheduler type for diffusion policies (e.g., DDIM, DDPM)",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Number of inference/denoising steps for diffusion policies",
    )
    parser.add_argument(
        "--crop_height",
        type=int,
        default=None,
        help="Image crop height for preprocessing",
    )
    parser.add_argument(
        "--crop_width",
        type=int,
        default=None,
        help="Image crop width for preprocessing",
    )
    parser.add_argument(
        "--action-delta-scale",
        type=float,
        default=0.1,
        help="Scale factor for action delta (e.g., 0.1 means 10% of the delta is applied each step)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    print(f"🤖 Starting ErgoCub continuous policy evaluation")
    print(f"   Policy: {args.policy_path}")
    print(f"   Control frequency: {args.fps} Hz")
    print(f"   Running continuously (Ctrl+C to stop)")
    print(f"   No data will be saved")

    # Create robot configuration
    robot_config = ErgoCubConfig(
        remote_prefix=args.remote_prefix,
        local_prefix=args.local_prefix,
        cameras={
            "agentview": YarpCameraConfig(
                yarp_name="depthCamera/rgbImage:o", width=1280, height=720, fps=30
            ),
            "infrared": YarpCameraConfig(
                yarp_name="depthCamera/ir:o", width=1280, height=720, fps=30
            ),
        }
    )

    # Initialize robot
    print("🔧 Initializing robot...")
    robot = ErgoCub(robot_config)

    # Load policy using the correct LeRobot method
    print(f"📥 Loading policy from {args.policy_path}...")
    try:
        # For pretrained models, use from_pretrained which loads weights AND normalization stats
        print("   Loading pretrained policy with normalization stats...")
        policy_class = None

        # Load policy configuration to determine the policy class
        policy_config = PreTrainedConfig.from_pretrained(args.policy_path)
        print(f"   Policy type: {policy_config.type}")

        # Import the specific policy class based on the config type
        if policy_config.type == "diffusion":
            from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

            policy_class = DiffusionPolicy
        elif policy_config.type == "act":
            from lerobot.policies.act.modeling_act import ACTPolicy

            policy_class = ACTPolicy
        elif policy_config.type == "tdmpc":
            from lerobot.policies.tdmpc.modeling_tdmpc import TDMPCPolicy

            policy_class = TDMPCPolicy
        elif policy_config.type == "vqbet":
            from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy

            policy_class = VQBeTPolicy
        elif policy_config.type == "sac":
            from lerobot.policies.sac.modeling_sac import SACPolicy

            policy_class = SACPolicy
        else:
            raise ValueError(f"Unsupported policy type: {policy_config.type}")

        # Load the pretrained policy (this loads weights + normalization stats)
        policy = policy_class.from_pretrained(args.policy_path)
        print("   ✅ Successfully loaded pretrained policy with normalization stats")

        # Configure inference settings for diffusion policies
        if policy_config.type == "diffusion" and hasattr(policy, 'noise_scheduler'):
            print("   🔧 Configuring diffusion policy inference settings...")
            
            # Set noise scheduler type if specified
            if args.noise_scheduler_type:
                print(f"   Setting noise scheduler type: {args.noise_scheduler_type}")
                if hasattr(policy.noise_scheduler, 'scheduler_type'):
                    policy.noise_scheduler.scheduler_type = args.noise_scheduler_type
                elif hasattr(policy, 'scheduler_type'):
                    policy.scheduler_type = args.noise_scheduler_type
                else:
                    print(f"   ⚠️  Warning: Cannot set scheduler type on this policy")
            
            # Set number of inference steps if specified
            if args.num_inference_steps:
                print(f"   Setting inference steps: {args.num_inference_steps}")
                if hasattr(policy, 'num_inference_steps'):
                    policy.num_inference_steps = args.num_inference_steps
                elif hasattr(policy.noise_scheduler, 'num_inference_steps'):
                    policy.noise_scheduler.num_inference_steps = args.num_inference_steps
                else:
                    print(f"   ⚠️  Warning: Cannot set inference steps on this policy")
        
        # Configure image preprocessing if crop dimensions are specified
        if args.crop_height and args.crop_width:
            print(f"   🖼️  Configuring image crop size: {args.crop_height}x{args.crop_width}")
            if hasattr(policy, 'config') and hasattr(policy.config, 'crop_shape'):
                policy.config.crop_shape = [args.crop_height, args.crop_width]
                print(f"   Set crop_shape to: {policy.config.crop_shape}")
            else:
                print(f"   ⚠️  Warning: Cannot set crop_shape on this policy")

        print("✅ Policy loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load policy: {e}")
        print(f"   Error details: {type(e).__name__}: {str(e)}")
        import traceback

        print("   Full traceback:")
        traceback.print_exc()
        return 1

    # Connect to robot
    print("🔌 Connecting to robot...")
    try:
        robot.connect()
        if not robot.is_connected:
            print("❌ Failed to connect to robot!")
            return 1
        print("✅ Robot connected successfully!")
    except Exception as e:
        print(f"❌ Error connecting to robot: {e}")
        return 1

    # Initialize rerun and keyboard listener
    print("🖥️  Initializing visualization and controls...")
    _init_rerun(session_name="ergocub_continuous_policy")
    listener, events = init_keyboard_listener()

    print("\n" + "=" * 50)
    print("🎯 CONTINUOUS POLICY EXECUTION")
    print("=" * 50)
    print("Controls:")
    print("  - Press 's' to stop")
    print("  - Press Ctrl+C to exit")
    print("=" * 50 + "\n")

    # Reset policy state
    if policy is not None:
        policy.reset()

    try:
        log_say("🚀 Starting continuous policy execution")
        print(f"Policy device: {next(policy.parameters()).device}")
        print(f"Robot connected: {robot.is_connected}")
        print(f"Robot action features: {list(robot.action_features.keys())}")
        print(
            f"Policy input features: {list(policy.config.input_features.keys()) if hasattr(policy.config, 'input_features') else 'None'}"
        )
        print(
            f"Policy output features: {list(policy.config.output_features.keys()) if hasattr(policy.config, 'output_features') else 'None'}"
        )

        # Simple infinite control loop
        start_time = time.perf_counter()
        loop_count = 0

        while not events["stop_recording"]:
            loop_start = time.perf_counter()
            loop_count += 1

            if events["exit_early"]:
                break

            # Get robot observation
            observation = robot.get_observation()

            # Convert observation to the format expected by policy
            # The policy expects specific key names like 'observation.state', 'observation.images.agentview'
            # but the robot provides raw keys like 'agentview', joint names, etc.
            formatted_observation = {}

            # Collect all non-image data into observation.state
            state_data = []
            for key, value in observation.items():
                if not key in ["agentview", "infrared"]:  # Skip camera images
                    if isinstance(value, (int, float, bool)):
                        state_data.append(float(value))
                    elif hasattr(value, "__len__") and not isinstance(value, str):
                        state_data.extend([float(x) for x in value])
                    else:
                        state_data.append(float(value))

            # Convert state data to numpy array
            if state_data:
                formatted_observation["observation.state"] = np.array(
                    state_data, dtype=np.float32
                )

            # Map camera images to expected format
            if "agentview" in observation:
                formatted_observation["observation.images.agentview"] = observation[
                    "agentview"
                ]

            if "infrared" in observation:
                formatted_observation["observation.images.infrared"] = observation[
                    "infrared"
                ]

            # Note: Do NOT add 'task' and 'robot_type' here - predict_action handles them internally

            # Predict action using policy
            action_values = predict_action(
                formatted_observation,
                policy,
                get_safe_torch_device(policy.config.device),
                policy.config.use_amp,
                task=args.task_description,
                robot_type=robot.robot_type,
            )
            action = {
                key: action_values[i].item()
                for i, key in enumerate(robot.action_features)
            }

            # --- SCALE ACTION DELTA TO USER-DEFINED FRACTION AND PRINT DELTAS ---
            scale = args.action_delta_scale
            if not hasattr(main, "_prev_action"):
                main._prev_action = None
            prev_action = main._prev_action
            deltas = {}
            if prev_action is not None:
                for k in action:
                    delta = action[k] - prev_action[k]
                    action[k] = prev_action[k] + scale * delta
                    deltas[k] = delta
                # print(f"Action deltas: {{}}".format({{k: f"{deltas[k]:.4f}" for k in deltas}}))
            main._prev_action = action.copy()
            # --- END SCALE ---
            # Send action to robot
            robot.send_action(action)

            # Optional visualization
            if loop_count % 10 == 0:  # Log every 10th loop to reduce spam
                log_rerun_data(observation, action)

            # Maintain control frequency
            loop_time = time.perf_counter() - loop_start
            target_dt = 1.0 / args.fps

            if loop_time > target_dt:
                print(
                    f"Loop {loop_count}: Slow control {loop_time*1000:.1f}ms (target: {target_dt*1000:.1f}ms)"
                )

            # Wait for next cycle
            remaining_time = target_dt - loop_time
            if remaining_time > 0:
                time.sleep(remaining_time)

            # Print status every 100 loops
            if loop_count % 100 == 0:
                elapsed = time.perf_counter() - start_time
                avg_hz = loop_count / elapsed
                print(f"Loop {loop_count}: Running at {avg_hz:.1f} Hz average")

    except KeyboardInterrupt:
        print("\n🛑 Execution interrupted by user")

    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback

        print("   Full traceback:")
        traceback.print_exc()
        return 1

    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        robot.disconnect()
        listener.stop()

    print("\n🎉 Continuous execution completed!")
    return 0


if __name__ == "__main__":
    exit(main())
