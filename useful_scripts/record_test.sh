DATA_DIRS="$CONDA_PREFIX/share/ergoCub/robots"
python -m lerobot.record \
   --robot.type=ergocub \
   --robot.remote_prefix="/ergocubSim" \
   --robot.local_prefix="/lerobot" \
   --robot.cameras='{ external: {"type": "yarp", "yarp_name": "camera", "width": 640, "height": 480, "fps": 30}, egocentric:{"type": "yarp", "yarp_name": "ergocubSim/depthCamera/rgbImage:o", "width": 640, "height": 480, "fps": 30}}' \
   --robot.control_boards='[neck,left_arm,right_arm,fingers]' \
   --dataset.repo_id=steb6/eval_redball_long \
   --dataset.fps=10 \
   --policy.path=steb6/trained-redball-40k \
   --policy.num_inference_steps=100 \
   --dataset.single_task="get the ball" \
   --dataset.reset_time=0 \
   --dataset.episode_time=30 \
   --dataset.num_episodes=50
   # --display_data=true
