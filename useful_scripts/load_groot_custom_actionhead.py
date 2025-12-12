#!/usr/bin/env python3
"""
Load pretrained GROOT backbone with a custom action head.

This script creates a hybrid GROOT model by:
1. Loading the pretrained backbone (vision + LLM) weights
2. Initializing a new action head with custom dimensions (e.g., 36 actions)
3. Saving the hybrid model for training

Usage:
    python useful_scripts/load_groot_custom_actionhead.py \
        --pretrained-model nvidia/gr00t-n1.5 \
        --action-dim 36 \
        --action-horizon 16 \
        --output-dir checkpoints/groot_hybrid_36dim
"""

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from lerobot.policies.groot.groot_n1 import GR00TN15, GR00TN15Config


def load_hybrid_model(
    pretrained_model_path: str,
    action_dim: int,
    action_horizon: int,
    max_state_dim: int,
    output_dir: Path,
):
    """Load pretrained backbone and create new action head with custom dimensions."""
    
    print("="*80)
    print("GROOT Hybrid Model Creator")
    print("="*80)
    print(f"\nPretrained model: {pretrained_model_path}")
    print(f"Custom action_dim: {action_dim}")
    print(f"Custom action_horizon: {action_horizon}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Download pretrained model
    print("Downloading pretrained model...")
    try:
        local_model_path = snapshot_download(pretrained_model_path, repo_type="model")
    except Exception as e:
        print(f"Failed to download from hub, trying local path: {e}")
        local_model_path = pretrained_model_path
    
    # Load the pretrained config
    config_path = Path(local_model_path) / "config.json"
    with open(config_path, 'r') as f:
        pretrained_config_dict = json.load(f)
    
    print(f"\nPretrained model config:")
    print(f"  action_dim: {pretrained_config_dict.get('action_dim', 'N/A')}")
    print(f"  action_horizon: {pretrained_config_dict.get('action_horizon', 'N/A')}")
    
    # Create new config with custom action dimensions
    new_config_dict = pretrained_config_dict.copy()
    new_config_dict['action_dim'] = action_dim
    new_config_dict['action_horizon'] = action_horizon
    
    # Update action_head_cfg if it exists
    if 'action_head_cfg' in new_config_dict:
        new_config_dict['action_head_cfg']['action_dim'] = action_dim
        new_config_dict['action_head_cfg']['action_horizon'] = action_horizon
        new_config_dict['action_head_cfg']['max_action_dim'] = action_dim
    
    # Update backbone_cfg if it exists and has max_state_dim
    if 'backbone_cfg' in new_config_dict:
        if 'project_to_dim' in new_config_dict['backbone_cfg']:
            print(f"  backbone project_to_dim: {new_config_dict['backbone_cfg']['project_to_dim']}")
    
    print(f"\nNew config:")
    print(f"  action_dim: {action_dim}")
    print(f"  action_horizon: {action_horizon}")
    print(f"  max_state_dim: {max_state_dim}")
    
    # Create model with new config
    print("\nInitializing model with new action dimensions...")
    new_config = GR00TN15Config(**new_config_dict)
    model = GR00TN15(new_config, local_model_path=str(local_model_path))
    
    # Load pretrained weights
    print("\nLoading pretrained weights...")
    
    # Try different weight file formats
    pretrained_state_dict = None
    
    # Check for sharded safetensors (model-00001-of-00003.safetensors, etc.)
    index_path = Path(local_model_path) / "model.safetensors.index.json"
    if index_path.exists():
        print("Loading sharded safetensors...")
        from safetensors.torch import load_file
        
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        pretrained_state_dict = {}
        weight_files = set(index['weight_map'].values())
        
        for weight_file in weight_files:
            print(f"  Loading {weight_file}...")
            shard_path = Path(local_model_path) / weight_file
            shard_dict = load_file(str(shard_path))
            pretrained_state_dict.update(shard_dict)
    
    # Try single safetensors file
    elif (Path(local_model_path) / "model.safetensors").exists():
        from safetensors.torch import load_file
        pretrained_state_dict = load_file(str(Path(local_model_path) / "model.safetensors"))
    
    # Try pytorch_model.bin
    elif (Path(local_model_path) / "pytorch_model.bin").exists():
        pretrained_state_dict = torch.load(Path(local_model_path) / "pytorch_model.bin", map_location='cpu')
    
    else:
        raise FileNotFoundError(f"No model weights found in {local_model_path}")
    
    # Filter to load only backbone weights
    backbone_state_dict = {}
    action_head_keys_skipped = []
    
    for key, value in pretrained_state_dict.items():
        if key.startswith('backbone.'):
            backbone_state_dict[key] = value
        else:
            action_head_keys_skipped.append(key)
    
    print(f"\nLoading {len(backbone_state_dict)} backbone parameters...")
    print(f"Skipping {len(action_head_keys_skipped)} action head parameters (will be randomly initialized)")
    
    # Load backbone weights, allow missing keys for action head
    missing_keys, unexpected_keys = model.load_state_dict(backbone_state_dict, strict=False)
    
    # Filter out expected missing keys (action head)
    unexpected_missing = [k for k in missing_keys if not k.startswith('action_head.')]
    
    if unexpected_missing:
        print(f"\n⚠️  Warning: Unexpected missing keys: {unexpected_missing[:5]}...")
    if unexpected_keys:
        print(f"\n⚠️  Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")
    
    print(f"\n✓ Successfully loaded backbone weights")
    print(f"✓ Action head initialized with random weights for {action_dim} dimensions")
    
    # Save the hybrid model
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving hybrid model to {output_dir}...")
    
    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(new_config_dict, f, indent=2)
    
    # Save model weights
    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    
    print(f"\n✓ Hybrid model saved!")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Create GROOT hybrid model with custom action head')
    parser.add_argument('--pretrained-model', type=str, default='nvidia/GR00T-N1.5-3B',
                       help='Pretrained model name or path (default: nvidia/GR00T-N1.5-3B)')
    parser.add_argument('--action-dim', type=int, default=36,
                       help='Custom action dimension (default: 36)')
    parser.add_argument('--action-horizon', type=int, default=16,
                       help='Action horizon (default: 16, max for GROOT)')
    parser.add_argument('--max-state-dim', type=int, default=36,
                       help='Maximum state dimension (default: 36)')
    parser.add_argument('--output-dir', type=str, default='checkpoints/groot_hybrid_36dim',
                       help='Output directory for hybrid model (default: checkpoints/groot_hybrid_36dim)')
    parser.add_argument('--push-to-hub', action='store_true',
                       help='Push the model to HuggingFace Hub')
    parser.add_argument('--hub-repo', type=str, default=None,
                       help='HuggingFace Hub repository (default: steb6/{model_name}-head{action_dim})')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    saved_dir = load_hybrid_model(
        pretrained_model_path=args.pretrained_model,
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
        max_state_dim=args.max_state_dim,
        output_dir=output_dir,
    )
    
    # Push to hub if requested
    if args.push_to_hub:
        from huggingface_hub import HfApi
        
        # Generate repo name if not provided
        if args.hub_repo is None:
            # Extract model name from pretrained path (e.g., nvidia/GR00T-N1.5-3B -> GR00T-N1.5-3B)
            model_name = args.pretrained_model.split('/')[-1]
            hub_repo = f"steb6/{model_name}-head{args.action_dim}"
        else:
            hub_repo = args.hub_repo
        
        print("\n" + "="*80)
        print(f"Pushing model to HuggingFace Hub: {hub_repo}")
        print("="*80)
        
        api = HfApi()
        api.create_repo(repo_id=hub_repo, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(saved_dir),
            repo_id=hub_repo,
            repo_type="model",
        )
        
        print(f"\n✓ Model successfully pushed to {hub_repo}")
        print("\nYou can now train with:")
        print(f"  policy.base_model_path={hub_repo}")
        print(f"  policy.max_action_dim={args.action_dim}")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("You can now train with:")
        print(f"  --policy.pretrained_model_name_or_path={output_dir}")
        print(f"  --policy.tune_visual=false")
        print(f"  --policy.tune_llm=false")
        print("\nTo push to HuggingFace Hub, run again with --push-to-hub")
        print("="*80)


if __name__ == "__main__":
    main()
