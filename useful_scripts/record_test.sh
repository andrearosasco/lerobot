DATA_DIRS="$CONDA_PREFIX/share/ergoCub/robots"
python -m lerobot.record \
   --robot.type=ergocub \
   --robot.remote_prefix="/ergocubSim" \
   --robot.local_prefix="/lerobot" \
   --robot.cameras='{ agentview: {"type": "yarp", "yarp_name": "camera", "width": 640, "height": 480, "fps": 30}}' \
   --robot.control_boards='[head,left_arm,right_arm,fingers]' \
   --dataset.repo_id=steb6/eval_ergocub-point \
   --dataset.fps=10 \
   --policy.path=steb6/train-ergocub-point \
   --policy.num_inference_steps=10 \
   --dataset.single_task="Wave the hand" \
   --dataset.reset_time=0 \
   --dataset.episode_time=10 \
