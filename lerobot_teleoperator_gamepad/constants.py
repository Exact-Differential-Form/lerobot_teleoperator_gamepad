from __future__ import annotations

ACTION_TARGET_KEYS = (
    "target_x",
    "target_y",
    "target_z",
    "target_roll",
    "target_pitch",
    "target_yaw",
    "gripper",
)

DELTA_ACTION_KEYS = (
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_roll",
    "delta_pitch",
    "delta_yaw",
    "delta_gripper",
)

OBS_CARTESIAN_KEYS = (
    "ee_x",
    "ee_y",
    "ee_z",
    "ee_roll",
    "ee_pitch",
    "ee_yaw",
    "ee_gripper",
)

POSE_LABELS = ("x", "y", "z", "roll", "pitch", "yaw")

X_INDEX = 0
Y_INDEX = 1
Z_INDEX = 2
ROLL_INDEX = 3
PITCH_INDEX = 4
YAW_INDEX = 5

DEFAULT_ARM_IP = "192.168.1.3"
DEFAULT_LINEAR_STEP = 0.01
DEFAULT_ANGULAR_STEP = 0.05
DEFAULT_GRIPPER_STEP = 0.005
DEFAULT_SENSITIVITY = 5.0
DEFAULT_MIN_SENSITIVITY = 0.1
DEFAULT_MAX_SENSITIVITY = 15.0
DEFAULT_SENSITIVITY_FACTOR = 1.25
DEFAULT_MAX_REACH_RADIUS = 0.9
DEFAULT_GRIPPER_MIN = 0.0
DEFAULT_GRIPPER_MAX = 0.044
