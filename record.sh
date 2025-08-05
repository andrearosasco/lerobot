DATA_DIRS="$CONDA_PREFIX/share/ergoCub/robots"
python -m lerobot.record \
   --robot.type=ergocub \
   --robot.remote_prefix="/ergocub" \
   --robot.local_prefix="/lerobot" \
   --robot.cameras='{ agentview: {"type": "yarp", "yarp_name": "depthCamera/rgbImage:o", "width": 1280, "height": 720, "fps": 30},
                      infrared: {"type": "yarp", "yarp_name": "depthCamera/ir:o", "width": 1280, "height": 720, "fps": 30} }' \
   --robot.encoders_control_boards='[head,left_arm,right_arm,torso]' \
   --teleop.type=metaquest \
   --teleop.control_boards='[neck,left_arm,right_arm,fingers]' \
   --dataset.repo_id=ar0s/ergocub-pick-cubes \
   --dataset.num_episodes=50 \
    --dataset.fps=10 \
   --dataset.single_task="Pick all the cubes from the table and place them in the plastic container."