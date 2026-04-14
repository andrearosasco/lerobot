# Custom Manipulator Quickstart

Minimal commands to start the `custom_manipulator` setup from the repository root.

## Setup

```bash
conda env create -f panda.yml
conda activate lerobot
pip install -e .
```

Also install:

- `oculus_reader` by following the instructions in its repository
- `panda-ros2`

```bash
git clone https://github.com/andrearosasco/panda-ros2.git
cd panda-ros2
colcon build --merge-install --install-base ${CONDA_PREFIX}
cd ../lerobot
```

## Start the robot interface

In two separate terminals:

```bash
ros2 launch panda_control panda_control.launch.py
```

```bash
ros2 launch robotiq_85_driver gripper_driver.launch.py
```

## Teleop / record dataset

This uses the configuration in `cfgs/record.yaml`:

```bash
python -m lerobot.robots.custom_manipulator.record
```

## Run a policy

Pass the checkpoint with `--policy.path`:

```bash
python -m lerobot.robots.custom_manipulator.record \
  --policy.path=outputs/train/<run>/checkpoints/last/pretrained_model
```

Or load it from Hugging Face Hub:

```bash
python -m lerobot.robots.custom_manipulator.record \
  --policy.path=<hf-user>/<policy-repo>
```

## Quick notes

- The main config file is `cfgs/record.yaml`.
- If `teleop` is enabled in the config, the policy is used when teleop is not engaged.
- For a policy-only test, set `teleop: null` in `cfgs/record.yaml`.
