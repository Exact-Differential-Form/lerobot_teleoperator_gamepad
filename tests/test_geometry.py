from __future__ import annotations

import math

from lerobot_teleoperator_gamepad.geometry import (
    CartesianTarget,
    clamp_target,
    driver_pose_to_target_pose,
    target_pose_to_driver_pose,
    xyz_radius,
)


def assert_close_tuple(left, right, tol=1e-6):
    assert len(left) == len(right)
    for lval, rval in zip(left, right, strict=True):
        assert math.isclose(lval, rval, abs_tol=tol)


def test_driver_target_pose_round_trip_base_origin():
    target_pose = (0.35, 0.12, 0.42, 0.1, -0.2, 0.3)

    driver_pose = target_pose_to_driver_pose(target_pose, max_reach_radius=0.9)
    recovered = driver_pose_to_target_pose(driver_pose)

    assert_close_tuple(recovered, target_pose)


def test_clamp_target_limits_xyz_radius_and_gripper():
    target = CartesianTarget((2.0, 0.0, 0.0, 0.1, 0.2, 0.3), 0.2)

    clamped = clamp_target(target, max_reach_radius=0.5, gripper_min=0.0, gripper_max=0.044)

    assert math.isclose(xyz_radius(clamped.pose), 0.5, abs_tol=1e-9)
    assert clamped.pose[3:] == target.pose[3:]
    assert clamped.gripper == 0.044
