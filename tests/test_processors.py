from __future__ import annotations

import math

from lerobot_teleoperator_gamepad.constants import ACTION_TARGET_KEYS, OBS_CARTESIAN_KEYS
from lerobot_teleoperator_gamepad.processors import TrossenGamepadActionProcessorStep
from lerobot_teleoperator_gamepad.record_processors import make_trossen_gamepad_teleop_action_processor


def observation(values):
    return dict(zip(OBS_CARTESIAN_KEYS, values, strict=True))


def test_processor_initializes_from_observation_and_applies_delta():
    processor = make_trossen_gamepad_teleop_action_processor()
    obs = observation((0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 0.01))
    raw_action = {
        "delta_x": 0.05,
        "delta_y": -0.01,
        "delta_z": 0.02,
        "delta_roll": 0.1,
        "delta_pitch": 0.0,
        "delta_yaw": -0.2,
        "delta_gripper": 0.005,
        "reset": False,
        "quit": False,
    }

    action = processor((raw_action, obs))

    assert set(action) == set(ACTION_TARGET_KEYS)
    assert math.isclose(action["target_x"], 0.15)
    assert math.isclose(action["target_y"], 0.19)
    assert math.isclose(action["target_z"], 0.32)
    assert math.isclose(action["target_roll"], 0.11)
    assert math.isclose(action["target_yaw"], -0.17)
    assert math.isclose(action["gripper"], 0.015)


def test_processor_reset_returns_initial_target():
    processor = make_trossen_gamepad_teleop_action_processor()
    obs = observation((0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.02))

    processor(({"delta_x": 0.1, "delta_gripper": -0.01}, obs))
    reset_action = processor(({"reset": True}, obs))

    assert reset_action == dict(zip(ACTION_TARGET_KEYS, (0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.02), strict=True))


def test_processor_clamps_target():
    step = TrossenGamepadActionProcessorStep(max_reach_radius=0.2, gripper_min=0.0, gripper_max=0.044)
    processor = make_trossen_gamepad_teleop_action_processor()
    processor.steps = [step]
    obs = observation((0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04))

    action = processor(({"delta_x": 1.0, "delta_gripper": 1.0}, obs))

    assert math.isclose(action["target_x"], 0.2)
    assert math.isclose(action["gripper"], 0.044)


def test_yaw_follow_xy_updates_yaw_from_target_xy_and_offset():
    step = TrossenGamepadActionProcessorStep(controller_yaw_follow_xy=True)
    processor = make_trossen_gamepad_teleop_action_processor()
    processor.steps = [step]
    obs = observation((1.0, 0.0, 0.2, 0.0, 0.0, 0.1, 0.01))

    action = processor(({"delta_y": 1.0, "delta_yaw": 0.2}, obs))

    assert math.isclose(action["target_yaw"], math.atan2(1.0, 1.0) + 0.3)


def test_yaw_follow_xy_normalizes_yaw():
    step = TrossenGamepadActionProcessorStep(controller_yaw_follow_xy=True)
    processor = make_trossen_gamepad_teleop_action_processor()
    processor.steps = [step]
    obs = observation((-1.0, 0.0, 0.2, 0.0, 0.0, math.pi - 0.01, 0.01))

    action = processor(({"delta_yaw": 0.1}, obs))

    assert -math.pi <= action["target_yaw"] <= math.pi
    assert math.isclose(action["target_yaw"], -math.pi + 0.09, abs_tol=1e-6)


def test_reset_action_schema_stays_absolute_7d():
    processor = make_trossen_gamepad_teleop_action_processor()
    obs = observation((0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.02))

    processor(({"delta_x": 0.1, "reset": False, "quit": False}, obs))
    reset_action = processor(({"reset": True, "quit": False}, obs))

    assert tuple(reset_action) == ACTION_TARGET_KEYS
    assert "reset" not in reset_action
    assert "quit" not in reset_action
