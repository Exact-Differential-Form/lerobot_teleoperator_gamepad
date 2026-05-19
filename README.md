# LeRobot Trossen Gamepad Teleoperator

This package adds a LeRobot-compatible Xbox/gamepad teleoperator and a Cartesian
Trossen follower wrapper for collecting datasets with a single follower arm.

The recorder entry point keeps the LeRobot dataset flow, but changes the action
schema from follower joint positions to absolute Cartesian targets:

```text
target_x, target_y, target_z, target_roll, target_pitch, target_yaw, gripper
```

## Install

Use the same environment as `lerobot_trossen`:

```bash
cd /home/jiahao/code/lerobot_trossen
uv sync
uv pip install -e /home/jiahao/code/lerobot_teleoperator_gamepad
```

## Preflight

```bash
uv run lerobot-find-cameras realsense --record-time-s=3
uv run python /home/jiahao/code/trossen_ctrl/xbox_test.py --device /dev/input/eventX
uv run python -c "import lerobot, trossen_arm, evdev, pyrealsense2, lerobot_teleoperator_gamepad"
```

## Record

Dual-camera smoke command, using D405 as wrist camera and D455F as external camera:

```bash
uv run lerobot-record-trossen-gamepad \
  --robot.type=trossen_cartesian_follower_robot \
  --robot.ip_address=192.168.1.3 \
  --robot.id=follower \
  --robot.stage_on_connect=false \
  --robot.sleep_on_disconnect=false \
  --robot.cartesian_goal_time=0.5 \
  --robot.cameras='{cam_wrist: {type: intelrealsense, serial_number_or_name: "419122271402", width: 640, height: 480, fps: 30}, cam_external: {type: intelrealsense, serial_number_or_name: "254122301124", width: 640, height: 480, fps: 30}}' \
  --teleop.type=trossen_gamepad_cartesian_teleop \
  --teleop.device=/dev/input/by-id/usb-PowerA_Xbox_Series_X_Wired_Controller_Black_00000101A4059B03-event-joystick \
  --teleop.id=xbox \
  --teleop.config_path=/home/jiahao/code/trossen_ctrl/controller/config.yaml \
  --teleop.deadzone_fraction=0.2 \
  --dataset.repo_id=jiahao/widowxai-gamepad-dual-smoke \
  --dataset.root=/tmp/lerobot_trossen_gamepad_dual_smoke \
  --dataset.push_to_hub=false \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=10 \
  --dataset.reset_time_s=3 \
  --dataset.single_task="Move the end effector slightly with the gamepad" \
  --display_data=false
```

The recorded image features should appear as `observation.images.cam_wrist` and
`observation.images.cam_external`.
