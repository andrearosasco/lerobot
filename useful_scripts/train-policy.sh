python \
	src/lerobot/scripts/lerobot_train.py \
	--wandb.enable=true \
	--batch_size=8 \
	--num_workers=4 \
	--policy.type=diffusion \
	--policy.repo_id=steb6/dp_redball_sim_reactive \
	--policy.crop_shape=null \
	--policy.resize_shape=[240,320] \
	--policy.noise_scheduler_type=DDIM \
	--policy.num_train_timesteps=100 \
	--policy.use_language_conditioning=true \
	--policy.language_dropout=0.0 \
	--dataset.repo_id=steb6/redball_sim_reactive \
	--policy.device=cuda \
	--output_dir=checkpoints/redball_sim_reactive