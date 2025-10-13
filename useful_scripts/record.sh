DATA_DIRS="$CONDA_PREFIX/share/ergoCub/robots"
python -m lerobot.record \
   --robot.type=ergocub \
   --robot.remote_prefix="/ergocubSim" \
   --robot.local_prefix="/lerobot" \
<<<<<<< Updated upstream
   --robot.cameras='{ external: {"type": "yarp", "yarp_name": "camera", "width": 640, "height": 480, "fps": 30}, egocentric:{"type": "yarp", "yarp_name": "ergocubSim/depthCamera/rgbImage:o", "width": 640, "height": 480, "fps": 30}}' \
   --robot.control_boards='[neck,left_arm,right_arm,fingers]' \
   --teleop.type=metaquest \
   --dataset.repo_id=steb6/redball \
   --dataset.num_episodes=100 \
   --dataset.fps=10 \
   --dataset.reset_time=1 \
   --dataset.episode_time=8 \
   --dataset.single_task="get the ball"
=======
   --robot.cameras='{ agentview: {"type": "yarp", "yarp_name": "camera", "width": 640, "height": 480, "fps": 30}}' \
   --robot.control_boards='[neck,left_arm,right_arm,fingers]' \
   --teleop.type=metaquest \
   --dataset.repo_id=ar0s/ergocub-point \
   --dataset.num_episodes=46 \
   --dataset.fps=10 \
   --dataset.reset_time=2 \
   --dataset.episode_time=10 \
   --dataset.single_task="Point to target" \
   --resume=True
>>>>>>> Stashed changes
