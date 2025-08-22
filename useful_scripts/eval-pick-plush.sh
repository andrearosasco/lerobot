
#!/bin/bash

# Set environment variables for ErgoCub
export DATA_DIRS="$CONDA_PREFIX/share/ergoCub/robots"

# Run the Python evaluation script with the original eval.sh parameters
python evaluate_ergocub_policy.py \
    --policy_path="steb6/trained-ergocub-pick-plush" \
    --num_episodes=1 \
    --episode_time_s=60 \
    --fps=10 \
    --remote_prefix="/ergocub" \
    --local_prefix="/lerobot" 
