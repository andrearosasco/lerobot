#!/bin/bash
#SBATCH --job-name=lerobot_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=gpua
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mem=128G

# Usage: sbatch submit_training.sh <policy_name> <dataset_repo>
# Example: sbatch submit_training.sh groot HSP-IIT/ergoPour_mergedV1

# Check arguments
if [ $# -ne 2 ]; then
    echo "Error: Requires exactly 2 arguments"
    echo "Usage: sbatch submit_training.sh <policy_name> <dataset_repo>"
    echo "Example: sbatch submit_training.sh groot HSP-IIT/ergoPour_mergedV1"
    exit 1
fi

POLICY_NAME=$1
DATASET_REPO=$2

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Policy: $POLICY_NAME"
echo "Dataset: $DATASET_REPO"
echo "GPU info:"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

# Activate conda environment
source ~/.bashrc
conda activate lerobot

# Change to project directory
cd /home/sberti/lerobot

# Run training using train_policy.sh
bash useful_scripts/train_policy.sh "$POLICY_NAME" "$DATASET_REPO"

# Capture exit code
EXIT_CODE=$?

echo "Training finished at: $(date)"
echo "Exit code: $EXIT_CODE"

# Exit with the training script's exit code
exit $EXIT_CODE
