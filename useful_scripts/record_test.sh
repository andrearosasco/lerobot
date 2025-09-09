DATA_DIRS="$CONDA_PREFIX/share/ergoCub/robots"
python -m lerobot.record \
   --robot.type=ergocub \
   --robot.remote_prefix="/ergocubSim" \
   --robot.local_prefix="/lerobot" \
   --robot.cameras='{ agentview: {"type": "yarp", "yarp_name": "ergocubSim/depthCamera/rgbImage:o", "width": 640, "height": 480, "fps": 30}}' \
   --robot.control_boards='[right_arm]' \
   --dataset.repo_id=ar0s/eval_ergocub-sim-wave \
   --dataset.fps=10 \
   --policy.path=steb6/traind-ergocub-sim-wave \
   --dataset.single_task="Wave the hand"
