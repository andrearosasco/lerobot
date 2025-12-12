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
)

# Add policy-specific parameters
case $POLICY in
    pi0)
        CMD+=(
            --policy.chunk_size=10
            --policy.n_action_steps=10
            --policy.max_state_dim=36
            --policy.max_action_dim=36
        )
        ;;
    smolvla)
        CMD+=(
            --policy.chunk_size=16
            --policy.n_action_steps=16
            --policy.max_state_dim=36
            --policy.max_action_dim=36
        )
        ;;
    act)
        CMD+=(
            --policy.chunk_size=20
            --policy.n_action_steps=20
        )
        ;;
    diffusion)
        CMD+=(
            --batch_size=8
            --num_workers=4
            --policy.crop_shape=null
            --policy.resize_shape='[240,320]'
            --policy.noise_scheduler_type=DDIM
            --policy.num_train_timesteps=100
        )
        ;;
    *)
        echo "Unknown policy: $POLICY"
        echo "Available policies: pi0, smolvla, act, diffusion"
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
