#!/bin/bash

# Evaluate the diffusion policy trained for pick-plush task
# with DDIM scheduler, 10 inference steps, and 240x320 image resolution

python evaluate_ergocub_policy.py \
    --policy_path=steb6/trained-ergocub-pick-plush-dp \
    --noise_scheduler_type=DDIM \
    --num_inference_steps=10 \
    --crop_height=240 \
    --crop_width=320 \
    --fps=10 \
    --task_description="Pick plush toy with diffusion policy"
