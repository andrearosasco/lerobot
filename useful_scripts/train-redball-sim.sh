python \
-m lerobot.scripts.train \
--wandb.enable=true \
--policy.type=diffusion \
--policy.repo_id=steb6/train-ergocub-point \
--dataset.repo_id=ar0s/ergocub-point \
--policy.device=cuda \
--output_dir=checkpoints/ergocub-point