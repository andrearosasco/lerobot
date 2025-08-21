#!/usr/bin/env python

"""
Evaluate a trained policy on the ErgoCub robot.

Usage:
    python evaluate_ergocub_policy.py --policy_path=steb6/ergocub-pick-mustard --num_episodes=5

This script will:
1. Load your trained policy from Hugging Face
2. Connect to the ErgoCub robot
3. Run policy inference for the specified number of episodes
4. Record the evaluation episodes for analysis
"""

import argparse
import logging
import time
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.record import record_loop
from lerobot.robots.ergocub import ErgoCubConfig, ErgoCub
from lerobot.cameras.yarp import YarpCameraConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun


def main():
    parser = argparse.ArgumentParser(description="Evaluate policy on ErgoCub robot")
    parser.add_argument(
        "--policy_path", 
        type=str, 
        required=True,
        help="Path or Hugging Face repo ID of the trained policy (e.g., steb6/ergocub-pick-mustard)"
    )
    parser.add_argument(
        "--num_episodes", 
        type=int, 
        default=3,
        help="Number of evaluation episodes to run"
    )
    parser.add_argument(
        "--episode_time_s", 
        type=float, 
        default=60.0,
        help="Maximum time per episode in seconds"
    )
    parser.add_argument(
        "--fps", 
        type=int, 
        default=10,
        help="Frames per second for recording"
    )
    parser.add_argument(
        "--task_description", 
        type=str, 
        default="Policy evaluation episode",
        help="Task description for the evaluation"
    )
    parser.add_argument(
        "--eval_dataset_repo_id", 
        type=str, 
        default=None,
        help="Repo ID to save evaluation episodes (optional)"
    )
    parser.add_argument(
        "--remote_prefix", 
        type=str, 
        default="/ergocub",
        help="Remote prefix for ErgoCub robot"
    )
    parser.add_argument(
        "--local_prefix", 
        type=str, 
        default="/lerobot",
        help="Local prefix for ErgoCub robot"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print(f"🤖 Starting ErgoCub policy evaluation")
    print(f"   Policy: {args.policy_path}")
    print(f"   Episodes: {args.num_episodes}")
    print(f"   FPS: {args.fps}")
    print(f"   Episode duration: {args.episode_time_s}s")
    
    # Create robot configuration
    robot_config = ErgoCubConfig(
        remote_prefix=args.remote_prefix,
        local_prefix=args.local_prefix,
        cameras={
            "agentview": YarpCameraConfig(
                yarp_name="depthCamera/rgbImage:o", 
                width=1280, 
                height=720, 
                fps=30
            ),
            "infrared": YarpCameraConfig(
                yarp_name="depthCamera/ir:o", 
                width=1280, 
                height=720, 
                fps=30
            )
        },
        encoders_control_boards=["head", "left_arm", "right_arm", "torso"]
    )
    
    # Initialize robot
    print("🔧 Initializing robot...")
    robot = ErgoCub(robot_config)
    
    # Configure dataset features for recording evaluation episodes
    print("📊 Setting up dataset features...")
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}
    
    # Create evaluation dataset (optional)
    eval_dataset = None
    if args.eval_dataset_repo_id:
        print(f"📁 Creating evaluation dataset: {args.eval_dataset_repo_id}")
        
        # Clean up any existing evaluation dataset directory
        eval_dataset_path = Path.home() / ".cache" / "huggingface" / "lerobot" / args.eval_dataset_repo_id
        if eval_dataset_path.exists():
            print(f"   Removing existing evaluation dataset: {eval_dataset_path}")
            import shutil
            shutil.rmtree(eval_dataset_path)
        
        eval_dataset = LeRobotDataset.create(
            repo_id=args.eval_dataset_repo_id,
            fps=args.fps,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=4,
        )
    else:
        print("⚠️  No dataset repo ID provided, episodes will not be saved")
    
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
        
        # Ensure we have an eval_dataset for recording episodes
        if not eval_dataset:
            print("   Creating recording dataset...")
            import uuid
            import shutil
            temp_name = f"temp_eval_dataset_{uuid.uuid4().hex[:8]}"
            
            # Clean up any existing dataset with the same name
            temp_dataset_path = Path.home() / ".cache" / "huggingface" / "lerobot" / temp_name
            if temp_dataset_path.exists():
                print(f"   Removing existing dataset: {temp_dataset_path}")
                shutil.rmtree(temp_dataset_path)
            
            eval_dataset = LeRobotDataset.create(
                repo_id=temp_name,
                fps=args.fps,
                features=dataset_features,
                robot_type=robot.name,
                use_videos=True,
                image_writer_threads=1,
            )
        
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
    _init_rerun(session_name="ergocub_policy_evaluation")
    listener, events = init_keyboard_listener()
    
    print("\n" + "="*50)
    print("🎯 EVALUATION STARTED")
    print("="*50)
    print("Controls:")
    print("  - Press 's' to stop recording")
    print("  - Press 'r' to re-record episode")
    print("  - Press Ctrl+C to exit")
    print("="*50 + "\n")
    
    recorded_episodes = 0
    success_count = 0
    
    try:
        while recorded_episodes < args.num_episodes and not events["stop_recording"]:
            episode_num = recorded_episodes + 1
            log_say(f"🚀 Running episode {episode_num}/{args.num_episodes}")
            print(f"Episode {episode_num}: Running policy inference...")
            
            episode_start_time = time.time()
            
            print(f"   Policy device: {next(policy.parameters()).device}")
            print(f"   Dataset features: {list(eval_dataset.features.keys()) if eval_dataset.features else 'None'}")
            print(f"   Robot connected: {robot.is_connected}")
            print(f"   Robot action features: {list(robot.action_features.keys())}")
            print(f"   Policy input features: {list(policy.config.input_features.keys()) if hasattr(policy.config, 'input_features') else 'None'}")
            print(f"   Policy output features: {list(policy.config.output_features.keys()) if hasattr(policy.config, 'output_features') else 'None'}")
            
            # Add debugging for first few robot observation and action to understand mapping
            print("\n   🔍 Getting robot observation for action mapping...")
            try:
                obs = robot.get_observation()
                print(f"   First few observation keys: {list(obs.keys())[:10]}")
                
                # Get a sample observation to understand the data structure
                observation_batch = {}
                for key in policy.config.input_features.keys():
                    if key in obs:
                        obs_data = obs[key]
                        if hasattr(obs_data, 'shape'):
                            print(f"   {key} shape: {obs_data.shape}")
                        # Convert to tensor and add batch dimension
                        import torch
                        if hasattr(obs_data, 'to'):  # already a tensor
                            observation_batch[key] = obs_data.unsqueeze(0).to(next(policy.parameters()).device)
                        else:  # numpy array or other
                            observation_batch[key] = torch.tensor(obs_data, dtype=torch.float32).unsqueeze(0).to(next(policy.parameters()).device)
                
                # Get a sample action from the policy to see its structure
                print("   🧠 Testing policy action output...")
                with torch.inference_mode():
                    sample_action = policy.select_action(observation_batch)
                
                print(f"   Policy action type: {type(sample_action)}")
                if hasattr(sample_action, 'shape'):
                    print(f"   Policy action shape: {sample_action.shape}")
                if hasattr(sample_action, 'keys'):
                    print(f"   Policy action keys: {list(sample_action.keys())}")
                    
            except Exception as debug_error:
                print(f"   ⚠️  Debug failed: {debug_error}")
            
            # Run policy inference loop
            try:
                record_loop(
                    robot=robot,
                    events=events,
                    fps=args.fps,
                    policy=policy,
                    dataset=eval_dataset,  # Always pass the dataset (never None when we have a policy)
                    control_time_s=args.episode_time_s,
                    single_task=args.task_description,
                    display_data=True,
                )
            except Exception as record_error:
                print(f"   ❌ Error in record_loop: {record_error}")
                import traceback
                print("   Record loop traceback:")
                traceback.print_exc()
                raise record_error
            
            episode_duration = time.time() - episode_start_time
            
            # Handle episode completion
            if not events["stop_recording"] and not events["rerecord_episode"]:
                # Only save episode if we have a real evaluation dataset (not temporary)
                if args.eval_dataset_repo_id:
                    eval_dataset.save_episode()
                else:
                    # For temporary dataset, just clear the buffer without saving
                    eval_dataset.clear_episode_buffer()
                
                recorded_episodes += 1
                print(f"✅ Episode {episode_num} completed in {episode_duration:.1f}s")
                
                # Ask user if episode was successful (optional)
                user_input = input("Was this episode successful? (y/n/skip): ").strip().lower()
                if user_input == 'y':
                    success_count += 1
                elif user_input == 'n':
                    pass  # Don't increment success count
                # 'skip' or any other input doesn't affect success count
                
            elif events["rerecord_episode"]:
                print("🔄 Re-recording episode...")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                eval_dataset.clear_episode_buffer()
                continue
    
    except KeyboardInterrupt:
        print("\n🛑 Evaluation interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        print("   Full traceback:")
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        robot.disconnect()
        listener.stop()
    
    # Results summary
    print("\n" + "="*50)
    print("📊 EVALUATION RESULTS")
    print("="*50)
    print(f"Total episodes completed: {recorded_episodes}")
    print(f"Successful episodes: {success_count}")
    if recorded_episodes > 0:
        success_rate = (success_count / recorded_episodes) * 100
        print(f"Success rate: {success_rate:.1f}%")
    print("="*50)
    
    # Optionally push evaluation data to hub
    if args.eval_dataset_repo_id and recorded_episodes > 0:
        push_to_hub = input("Push evaluation episodes to Hugging Face Hub? (y/n): ").strip().lower()
        if push_to_hub == 'y':
            print("📤 Pushing to Hugging Face Hub...")
            try:
                eval_dataset.push_to_hub()
                print("✅ Successfully pushed to Hub!")
            except Exception as e:
                print(f"❌ Error pushing to Hub: {e}")
    elif not args.eval_dataset_repo_id:
        print("ℹ️  Episodes were not saved (no eval_dataset_repo_id provided)")
    
    print("🎉 Evaluation completed!")
    return 0


if __name__ == "__main__":
    exit(main())
