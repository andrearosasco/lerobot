python \
	-m lerobot.scripts.train \
	--wandb.enable=true \
	--policy.type=diffusion \
	--policy.repo_id=steb6/traind-ergocub-move-plush \
	--dataset.repo_id=steb6/ergocub-move-plush  \
	--policy.device=cuda \
	--output_dir=checkpoints/ergocub-move-plush
