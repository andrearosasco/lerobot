python \
	src/lerobot/scripts/lerobot_train.py \
	--wandb.enable=true \
	--policy.type=act \
	--policy.repo_id=HSP-IIT/trained_ergoPour_resolved \
	--dataset.repo_id=HSP-IIT/ergoPour_resolved \
	--policy.device=cuda \
	--output_dir=checkpoints/ergoPour_resolved
