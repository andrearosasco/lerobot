python \
	-m lerobot.scripts.train \
	--wandb.enable=true \
	--policy.type=diffusion \
	--policy.repo_id=steb6/traind-ergocub-pick-mustard \
	--dataset.repo_id=ar0s/ergocub-pick-mustard \
	--policy.device=cuda \
	--output_dir=checkpoints/ergocub-pick-mustard
