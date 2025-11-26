python \
	src/lerobot/scripts/lerobot_train.py \
	--wandb.enable=true \
	--policy.type=act \
	--policy.repo_id=steb6/act_hri_ste_carmela_split \
	--policy.use_language_conditioning=true \
	--policy.chunk_size=10 \
	--policy.n_action_steps=5 \
	--policy.language_dropout=0.0 \
	--dataset.repo_id=steb6/hri_ste_carmela_split \
	--policy.device=cuda \
	--output_dir=checkpoints/ergoPohri_ste_carmela_splitur_resolved
