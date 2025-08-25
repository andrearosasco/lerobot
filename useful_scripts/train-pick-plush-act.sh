#!/bin/bash
rm -rf checkpoints/ergocub-pick-plush-act

/home/sberti/.conda/envs/lerobot/bin/python \
	-m lerobot.scripts.train \
	--wandb.enable=true \
	--policy.type=act \
	--policy.repo_id=steb6/traind-ergocub-pick-plush-act \
	--dataset.repo_id=ar0s/ergocub-pick-plush \
	--policy.device=cuda \
	--output_dir=checkpoints/ergocub-pick-plush-act \
	--dataset.image_keys='[observation.images.rgb]' \
	--policy.crop_shape='[240, 320]'
