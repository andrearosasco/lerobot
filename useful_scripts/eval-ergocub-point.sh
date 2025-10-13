#!/bin/bash

# Evaluate the diffusion policy trained for ergocub pointing task
# Uses lerobot-eval instead of record.py for proper evaluation metrics

python -m lerobot.scripts.eval \
    --policy.path=steb6/train-ergocub-point \
    --policy.device=cuda \
    --policy.num_inference_steps=10 \
    --env.type=ergocub \
    --env.remote_prefix="/ergocubSim" \
    --env.local_prefix="/lerobot" \
    --env.cameras='{ agentview: {"type": "yarp", "yarp_name": "camera", "width": 640, "height": 480, "fps": 30}}' \
    --env.control_boards='[neck,left_arm,right_arm,fingers]' \
    --eval.n_episodes=10 \
    --eval.batch_size=1 \
    --seed=1000 \
    --output_dir=outputs/eval/ergocub-point \
    --job_name=eval_ergocub_point