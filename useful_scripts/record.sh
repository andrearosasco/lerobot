DATA_DIRS="$CONDA_PREFIX/share/ergoCub/robots"
python -m lerobot.record \
   --robot.type=ergocub \
   --robot.remote_prefix="/ergocubSim" \
   --robot.local_prefix="/lerobot" \
   --robot.cameras='{ agentview: {"type": "yarp", "yarp_name": "camera", "width": 640, "height": 480, "fps": 30}}' \
   --robot.control_boards='[neck,left_arm,right_arm,fingers]' \
   --teleop.type=metaquest \
   --dataset.repo_id=ar0s/ergocub-point \
   --dataset.num_episodes=50 \
   --dataset.fps=10 \
   --dataset.reset_time=2 \
   --dataset.episode_time=10 \
   --dataset.single_task="Point your finger at the red ball"