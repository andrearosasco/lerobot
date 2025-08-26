#!/bin/bash

# Evaluate the ACT policy trained for pick-plush task
# with 240x320 image resolution

python evaluate_ergocub_policy.py \
    --policy_path=steb6/trained-ergocub-pick-plush-act \
    --crop_height=240 \
    --crop_width=320 \
    --fps=10 \
    --task_description="Pick plush toy with ACT policy"
