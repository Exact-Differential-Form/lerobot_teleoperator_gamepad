from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig

from .constants import (
    DEFAULT_ANGULAR_STEP,
    DEFAULT_ARM_IP,
    DEFAULT_GRIPPER_MAX,
    DEFAULT_GRIPPER_MIN,
    DEFAULT_GRIPPER_STEP,
    DEFAULT_LINEAR_STEP,
    DEFAULT_MAX_REACH_RADIUS,
    DEFAULT_SENSITIVITY_FACTOR,
)


@TeleoperatorConfig.register_subclass("trossen_gamepad_cartesian_teleop")
@dataclass
class TrossenGamepadCartesianTeleopConfig(TeleoperatorConfig):
    device: str | None = None
    config_path: str | None = None
    linear_step: float = DEFAULT_LINEAR_STEP
    angular_step: float = DEFAULT_ANGULAR_STEP
    gripper_step: float = DEFAULT_GRIPPER_STEP
    sensitivity: float | None = None
    sensitivity_factor: float = DEFAULT_SENSITIVITY_FACTOR
    min_sensitivity: float | None = None
    max_sensitivity: float | None = None
    deadzone_fraction: float = 0.125
    grab_device: bool = True


@RobotConfig.register_subclass("trossen_cartesian_follower_robot")
@dataclass
class TrossenCartesianFollowerConfig(RobotConfig):
    ip_address: str = DEFAULT_ARM_IP
    min_time_to_move_multiplier: float = 3.0
    loop_rate: int = 30
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    joint_names: list[str] = field(
        default_factory=lambda: [
            "joint_0",
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "left_carriage_joint",
        ]
    )
    staged_positions: list[float] = field(
        default_factory=lambda: [0.0, math.pi / 3.0, math.pi / 6.0, math.pi / 5.0, 0.0, 0.0, 0.0]
    )
    stage_on_connect: bool = False
    sleep_on_disconnect: bool = False
    max_reach_radius: float = DEFAULT_MAX_REACH_RADIUS
    gripper_min: float = DEFAULT_GRIPPER_MIN
    gripper_max: float = DEFAULT_GRIPPER_MAX
    cartesian_goal_time: float | None = None
    orientation_origin: Literal["base", "initial", "point"] = "base"
    orientation_origin_x: float = 0.0
    orientation_origin_y: float = 0.0
    controller_yaw_follow_xy: bool = False
    mode_init_wait: float = 0.5
    mode_probe_goal_time: float = 0.5
    skip_mode_probe: bool = False
