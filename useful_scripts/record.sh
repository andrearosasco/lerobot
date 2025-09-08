DATA_DIRS="$CONDA_PREFIX/share/ergoCub/robots"
python -m lerobot.record \
   --robot.type=ergocub \
   --robot.remote_prefix="/ergocubSim" \
   --robot.local_prefix="/lerobot" \
   --robot.cameras='{ agentview: {"type": "yarp", "yarp_name": "ergocubSim/depthCamera/rgbImage:o", "width": 640, "height": 480, "fps": 30}}' \
   --robot.control_boards='[right_arm]' \
   --teleop.type=metaquest \
   --dataset.repo_id=ar0s/ergocub-sim-wave \
   --dataset.num_episodes=40 \
   --dataset.fps=10 \
   --dataset.reset_time=2 \
   --dataset.episode_time=5 \
   --resume=True \
   --dataset.single_task="Wave the hand"