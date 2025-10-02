DATA_DIRS="$CONDA_PREFIX/share/ergoCub/robots"
python -m lerobot.record \
   --robot.type=ergocub \
   --robot.remote_prefix="/ergocub" \
   --robot.local_prefix="/lerobot" \
   --robot.cameras='{ external: {"type": "yarp", "yarp_name": "depthCamera/rgbImage:o", "width": 640, "height": 480, "fps": 30}}' \
   --robot.control_boards='[neck,left_arm,right_arm,fingers]' \
   --teleop.type=metaquest \
   --dataset.repo_id=steb6/test_teleop \
   --dataset.num_episodes=3 \
   --dataset.fps=10 \
   --dataset.reset_time=3 \
   --dataset.episode_time=30 \
   --dataset.single_task="test"