from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .constants import (
    ACTION_TARGET_KEYS,
    DEFAULT_GRIPPER_MAX,
    DEFAULT_GRIPPER_MIN,
    DEFAULT_MAX_REACH_RADIUS,
    OBS_CARTESIAN_KEYS,
    X_INDEX,
    Y_INDEX,
    Z_INDEX,
)

Pose = tuple[float, float, float, float, float, float]
BASE_ORIENTATION_ORIGIN = (0.0, 0.0)
ZERO_POSE: Pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class CartesianTarget:
    pose: Pose
    gripper: float

    def as_action_dict(self) -> dict[str, float]:
        return dict(zip(ACTION_TARGET_KEYS, (*self.pose, self.gripper), strict=True))

    def as_observation_dict(self) -> dict[str, float]:
        return dict(zip(OBS_CARTESIAN_KEYS, (*self.pose, self.gripper), strict=True))


def target_from_action(action: dict[str, float]) -> CartesianTarget:
    return CartesianTarget(
        pose=tuple(float(action[key]) for key in ACTION_TARGET_KEYS[:6]),  # type: ignore[arg-type]
        gripper=float(action[ACTION_TARGET_KEYS[6]]),
    )


def target_from_observation(observation: dict, fallback: CartesianTarget | None = None) -> CartesianTarget:
    if all(key in observation for key in OBS_CARTESIAN_KEYS):
        return CartesianTarget(
            pose=tuple(float(observation[key]) for key in OBS_CARTESIAN_KEYS[:6]),  # type: ignore[arg-type]
            gripper=float(observation[OBS_CARTESIAN_KEYS[6]]),
        )
    if fallback is not None:
        return fallback
    return CartesianTarget(ZERO_POSE, 0.0)


def clamp_target(
    target: CartesianTarget,
    max_reach_radius: float = DEFAULT_MAX_REACH_RADIUS,
    gripper_min: float = DEFAULT_GRIPPER_MIN,
    gripper_max: float = DEFAULT_GRIPPER_MAX,
) -> CartesianTarget:
    return CartesianTarget(
        pose=(*limit_xyz_radius(target.pose[:3], max_reach_radius), *target.pose[3:]),
        gripper=clamp(target.gripper, gripper_min, gripper_max),
    )


def target_pose_to_driver_pose(
    pose: Iterable[float],
    max_reach_radius: float = DEFAULT_MAX_REACH_RADIUS,
    orientation_origin_xy=BASE_ORIENTATION_ORIGIN,
) -> tuple[float, float, float, float, float, float]:
    x, y, z, roll, pitch, yaw = tuple(float(value) for value in pose)
    limited_xyz = limit_xyz_radius((x, y, z), max_reach_radius)
    base_yaw = orientation_base_yaw(limited_xyz, orientation_origin_xy)
    rotation = matmul(
        rotation_z(base_yaw + yaw),
        matmul(rotation_y(pitch), rotation_x(roll)),
    )
    return (*limited_xyz, *matrix_to_axis_angle(rotation))


def driver_pose_to_target_pose(
    pose: Iterable[float],
    orientation_origin_xy=BASE_ORIENTATION_ORIGIN,
) -> Pose:
    values = tuple(float(value) for value in pose)
    x, y, z = values[:3]
    rotation = axis_angle_to_matrix(values[3:])
    base_rotation = rotation_z(-orientation_base_yaw((x, y, z), orientation_origin_xy))
    relative_rotation = matmul(base_rotation, rotation)
    roll, pitch, yaw = matrix_to_roll_pitch_yaw(relative_rotation)
    return (x, y, z, roll, pitch, yaw)


def resolve_orientation_origin_xy(mode: str, pose: Iterable[float], point: tuple[float, float]) -> tuple[float, float]:
    values = tuple(float(value) for value in pose)
    if mode == "base":
        return BASE_ORIENTATION_ORIGIN
    if mode == "initial":
        return (values[X_INDEX], values[Y_INDEX])
    return tuple(float(value) for value in point)


def orientation_base_yaw(xyz: Iterable[float], orientation_origin_xy) -> float:
    values = tuple(float(value) for value in xyz)
    origin_x, origin_y = orientation_origin_xy
    return xy_yaw(values[X_INDEX] - origin_x, values[Y_INDEX] - origin_y)


def limit_xyz_radius(xyz: Iterable[float], max_radius: float) -> tuple[float, float, float]:
    values = tuple(float(value) for value in xyz)
    if max_radius <= 0.0:
        return values  # type: ignore[return-value]
    radius = xyz_radius(values)
    if radius <= max_radius or radius < 1e-12:
        return values  # type: ignore[return-value]
    scale = max_radius / radius
    return tuple(value * scale for value in values)  # type: ignore[return-value]


def xyz_radius(pose_or_xyz: Iterable[float]) -> float:
    values = tuple(float(value) for value in pose_or_xyz)
    return math.sqrt(values[X_INDEX] ** 2 + values[Y_INDEX] ** 2 + values[Z_INDEX] ** 2)


def xy_yaw(x: float, y: float) -> float:
    if math.hypot(x, y) < 1e-12:
        return 0.0
    return math.atan2(y, x)


def normalize_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def rotation_x(angle: float):
    cos_value = math.cos(angle)
    sin_value = math.sin(angle)
    return (
        (1.0, 0.0, 0.0),
        (0.0, cos_value, -sin_value),
        (0.0, sin_value, cos_value),
    )


def rotation_y(angle: float):
    cos_value = math.cos(angle)
    sin_value = math.sin(angle)
    return (
        (cos_value, 0.0, sin_value),
        (0.0, 1.0, 0.0),
        (-sin_value, 0.0, cos_value),
    )


def rotation_z(angle: float):
    cos_value = math.cos(angle)
    sin_value = math.sin(angle)
    return (
        (cos_value, -sin_value, 0.0),
        (sin_value, cos_value, 0.0),
        (0.0, 0.0, 1.0),
    )


def matmul(left, right):
    return tuple(
        tuple(
            sum(left[row][inner] * right[inner][col] for inner in range(3))
            for col in range(3)
        )
        for row in range(3)
    )


def axis_angle_to_matrix(vector):
    angle = math.sqrt(sum(value * value for value in vector))
    if angle < 1e-12:
        return identity_matrix()
    axis = tuple(value / angle for value in vector)
    x, y, z = axis
    cos_value = math.cos(angle)
    sin_value = math.sin(angle)
    one_minus_cos = 1.0 - cos_value
    return (
        (
            cos_value + x * x * one_minus_cos,
            x * y * one_minus_cos - z * sin_value,
            x * z * one_minus_cos + y * sin_value,
        ),
        (
            y * x * one_minus_cos + z * sin_value,
            cos_value + y * y * one_minus_cos,
            y * z * one_minus_cos - x * sin_value,
        ),
        (
            z * x * one_minus_cos - y * sin_value,
            z * y * one_minus_cos + x * sin_value,
            cos_value + z * z * one_minus_cos,
        ),
    )


def matrix_to_axis_angle(matrix):
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    angle = math.acos(clamp((trace - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-12:
        return (0.0, 0.0, 0.0)

    sin_angle = math.sin(angle)
    if abs(sin_angle) < 1e-8:
        return matrix_to_axis_angle_near_pi(matrix, angle)

    scale = angle / (2.0 * sin_angle)
    return (
        (matrix[2][1] - matrix[1][2]) * scale,
        (matrix[0][2] - matrix[2][0]) * scale,
        (matrix[1][0] - matrix[0][1]) * scale,
    )


def matrix_to_axis_angle_near_pi(matrix, angle):
    axis = [
        math.sqrt(max(0.0, (matrix[index][index] + 1.0) / 2.0))
        for index in range(3)
    ]
    if matrix[2][1] - matrix[1][2] < 0.0:
        axis[0] = -axis[0]
    if matrix[0][2] - matrix[2][0] < 0.0:
        axis[1] = -axis[1]
    if matrix[1][0] - matrix[0][1] < 0.0:
        axis[2] = -axis[2]
    length = math.sqrt(sum(value * value for value in axis))
    if length < 1e-12:
        return (angle, 0.0, 0.0)
    return tuple(angle * value / length for value in axis)


def matrix_to_roll_pitch_yaw(matrix):
    pitch = math.asin(clamp(-matrix[2][0], -1.0, 1.0))
    cos_pitch = math.cos(pitch)
    if abs(cos_pitch) > 1e-8:
        roll = math.atan2(matrix[2][1], matrix[2][2])
        yaw = math.atan2(matrix[1][0], matrix[0][0])
    else:
        roll = 0.0
        yaw = math.atan2(-matrix[0][1], matrix[1][1])
    return roll, pitch, yaw


def identity_matrix():
    return (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )


def clamp(value: float, min_value: float, max_value: float) -> float:
    return min(max(float(value), float(min_value)), float(max_value))
