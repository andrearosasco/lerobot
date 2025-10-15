python \
-m lerobot.scripts.train \
--wandb.enable=true \
--policy.type=diffusion \
--policy.repo_id=steb6/train-redballppp \
--dataset.repo_id=steb6/redball\
--policy.device=cuda \
--output_dir=checkpoints/redballppp