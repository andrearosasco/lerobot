/home/sberti/.conda/envs/lerobot/bin/python \
	-m lerobot.scripts.train \
	--wandb.enable=true \
	--policy.type=diffusion \
	--policy.repo_id=steb6/traind-ergocub-pick-plush \
	--dataset.repo_id=ar0s/ergocub-pick-plush \
	--policy.device=cuda \
	--output_dir=checkpoints/ergocub-pick-plush
