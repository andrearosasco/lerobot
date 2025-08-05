DATA_DIRS="$CONDA_PREFIX/share/ergoCub/robots"
python -m lerobot.record \
   --robot.type=ergocub \
   --robot.remote_prefix="/ergocub" \
   --robot.local_prefix="/lerobot" \
   --robot.cameras='{ agentview: {"type": "yarp", "yarp_name": "depthCamera", "width": 1280, "height": 720, "fps": 30} }' \
   --robot.encoders_control_boards='[head,left_arm,right_arm,torso]' \
   --teleop.type=metaquest \
   --teleop.control_boards='[neck,left_arm,right_arm,fingers]' \
   --dataset.repo_id=ar0s/ergocub-pick-plush \
   --dataset.num_episodes=50 \
    --dataset.fps=10 \
   --dataset.single_task="Pick up the orange plush from the table"