#!/bin/bash
rm -rf checkpoints/ergocub-pick-plush-dp

/home/sberti/.conda/envs/lerobot/bin/python \
	-m lerobot.scripts.train \
	--wandb.enable=true \
	--policy.type=diffusion \
	--policy.repo_id=steb6/traind-ergocub-pick-plush-dp \
	--dataset.repo_id=ar0s/ergocub-pick-plush \
	--policy.device=cuda \
	--output_dir=checkpoints/ergocub-pick-plush-dp \
	--policy.noise_scheduler_type=DDIM \
	--policy.num_inference_steps=10 \
	--dataset.image_keys='[observation.images.rgb]' \
	--policy.crop_shape='[240, 320]'
