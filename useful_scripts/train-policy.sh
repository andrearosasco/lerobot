#!/bin/bash

# Modular training script for different policy types
#
# Usage:
#   ./train_policy.sh act steb6/stand-wave-sim-split
#   ./train_policy.sh diffusion steb6/stand-wave-sim-split
#   ./train_policy.sh pi0 HSP-IIT/ergoPour_mergedV1

set -e

# Check arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <policy> <dataset_repo_id>"
    echo ""
    echo "Policies: pi0, smolvla, act, diffusion"
    echo ""
    echo "Examples:"
    echo "  $0 act steb6/stand-wave-sim-split"
    echo "  $0 diffusion steb6/stand-wave-sim-split"
    echo "  $0 pi0 HSP-IIT/ergoPour_mergedV1"
    exit 1
fi

POLICY=$1
DATASET=$2

# Fixed settings
WANDB="true"
DEVICE="cuda"

# Extract dataset parts
IFS='/' read -ra DATASET_PARTS <<< "$DATASET"
if [ ${#DATASET_PARTS[@]} -eq 2 ]; then
    DATASET_OWNER="${DATASET_PARTS[0]}"
    DATASET_NAME="${DATASET_PARTS[1]}"
else
    DATASET_OWNER="local"
    DATASET_NAME="$DATASET"
fi

# Set output and repo ID
OUTPUT="${POLICY}_${DATASET_NAME}"
REPO_ID="${DATASET_OWNER}/${POLICY}-${DATASET_NAME}"

# Base command
CMD=(
    python
    src/lerobot/scripts/lerobot_train.py
    --wandb.enable="$WANDB"
    --policy.type="$POLICY"
    --policy.repo_id="$REPO_ID"
    --dataset.repo_id="$DATASET"
    --policy.device="$DEVICE"
    --output_dir="checkpoints/$OUTPUT"
    --image_crop_params='{"observation.images.top": [0, 80, 480, 480]}' 
	--image_resize_size="[256,256]"
    --dataset.image_transforms.enable=true
    --dataset.image_transforms.tfs.greenscreen.weight=-1.0  # always apply greenscreen
    --dataset.image_transforms.max_num_transforms=3
    --dataset.image_transforms.tfs.affine.weight=0
    --dataset.image_transforms.random_order=true
)

# Add policy-specific parameters
case $POLICY in
    groot)  # inference every 1.6 seconds at 10 fps (max action_horizon=16 for pretrained models)
        # NOTE: To use pretrained GROOT with custom action dims (e.g., 36 instead of 32):
        # 1. Run: python useful_scripts/load_groot_custom_actionhead.py --action-dim 36 --pretrained-model nvidia/GR00T-N1.5-3B
        # 2. Then train with: --policy.base_model_path=steb6/GR00T-N1.5-3B-head36
        CMD+=(
            --policy.chunk_size=16
            --policy.n_action_steps=16
            --policy.max_action_dim=36
            --policy.base_model_path=steb6/GR00T-N1.5-3B-head36
        )
        ;;
    pi0)  # inference every 0.5 seconds at 10 fps
        CMD+=(
            --policy.chunk_size=5
            --policy.n_action_steps=5
            --policy.max_state_dim=36
            --policy.max_action_dim=36
        )
        ;;
    smolvla)  # inference every 1.6 seconds (50/30=1.6 fps)
        CMD+=(
            --policy.chunk_size=16
            --policy.n_action_steps=16
            --policy.max_state_dim=36
            --policy.max_action_dim=36
        )
        ;;
    act)  # inference every 2 seconds at 10 fps
        CMD+=(
            --policy.chunk_size=20
            --policy.n_action_steps=20
        )
        ;;
    diffusion)  # they use 10fps datasets, so default values are good
        CMD+=(
            --batch_size=8
            --num_workers=4
            --policy.crop_shape=null
            --policy.noise_scheduler_type=DDIM
            --policy.num_train_timesteps=100
        )
        ;;
    *)
        echo "Unknown policy: $POLICY"
        echo "Available policies: pi0, smolvla, act, diffusion, groot"
        exit 1
        ;;
esac

# Print command
echo "================================================================================"
echo "Training ${POLICY^^} policy on $DATASET"
echo "================================================================================"
echo ""
echo "Command:"
printf '%s \\\n' "${CMD[@]}" | sed '$ s/ \\$//'
echo ""
echo "================================================================================"
echo ""

# Execute command
"${CMD[@]}"
