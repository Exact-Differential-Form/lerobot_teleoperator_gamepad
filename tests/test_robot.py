from __future__ import annotations

import math

from types import SimpleNamespace

from lerobot_teleoperator_gamepad.config import TrossenCartesianFollowerConfig
from lerobot_teleoperator_gamepad.robot import TrossenCartesianFollower


class FakeDriver:
    def __init__(self):
        self.cartesian_calls = []
        self.gripper_calls = []
        self.mode_calls = []
        self.all_position_calls = []
        self.fail_cartesian = False
        self.cleared = False
        self.cartesian_pose = (0.1, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.gripper_position = 0.01

    def get_is_configured(self):
        return True

    def set_all_modes(self, mode):
        self.mode_calls.append(mode)

    def get_arm_positions(self):
        return [0.0] * 6

    def set_all_positions(self, **kwargs):
        self.all_position_calls.append(kwargs)

    def get_cartesian_positions(self):
        return self.cartesian_pose

    def get_gripper_position(self):
        return self.gripper_position

    def set_cartesian_positions(self, **kwargs):
        if self.fail_cartesian:
            raise RuntimeError("blocked")
        self.cartesian_calls.append(kwargs)

    def set_gripper_position(self, **kwargs):
        self.gripper_calls.append(kwargs)

    def clear_error(self):
        self.cleared = True


def make_robot():
    config = TrossenCartesianFollowerConfig(
        ip_address="0.0.0.0",
        max_reach_radius=0.5,
        gripper_max=0.044,
        cartesian_goal_time=0.2,
        cameras={},
        mode_init_wait=0.0,
    )
    robot = TrossenCartesianFollower(config)
    robot.driver = FakeDriver()
    return robot


def test_send_action_calls_cartesian_api_and_returns_clamped_action():
    robot = make_robot()
    action = {
        "target_x": 2.0,
        "target_y": 0.0,
        "target_z": 0.0,
        "target_roll": 0.0,
        "target_pitch": 0.0,
        "target_yaw": 0.0,
        "gripper": 1.0,
    }

    sent = robot.send_action(action)

    assert math.isclose(sent["target_x"], 0.5)
    assert math.isclose(sent["gripper"], 0.044)
    assert robot.driver.cartesian_calls
    assert robot.driver.gripper_calls
    assert robot.driver.cartesian_calls[-1]["goal_time"] == 0.2
    assert robot.driver.gripper_calls[-1]["goal_position"] == 0.044


def test_send_action_failure_recovers_to_measured_safe_action():
    robot = make_robot()
    first = {
        "target_x": 0.1,
        "target_y": 0.0,
        "target_z": 0.0,
        "target_roll": 0.0,
        "target_pitch": 0.0,
        "target_yaw": 0.0,
        "gripper": 0.01,
    }
    assert robot.send_action(first) == first

    robot.driver.fail_cartesian = True
    robot.driver.cartesian_pose = (0.3, 0.0, 0.0, 0.0, 0.0, 0.0)
    robot.driver.gripper_position = 0.02
    failed = dict(first, target_x=0.2, gripper=0.03)
    returned = robot.send_action(failed)

    assert math.isclose(returned["target_x"], 0.3)
    assert math.isclose(returned["gripper"], 0.02)
    assert robot.driver.cleared
    assert robot.driver.mode_calls


def test_configure_runs_position_mode_probe():
    robot = make_robot()

    robot.configure()

    assert robot.driver.mode_calls
    assert robot.driver.all_position_calls
    assert robot.driver.all_position_calls[0]["goal_time"] == robot.config.mode_probe_goal_time


def test_configure_can_skip_position_mode_probe():
    robot = make_robot()
    robot.config.skip_mode_probe = True

    robot.configure()

    assert robot.driver.mode_calls
    assert not robot.driver.all_position_calls


def test_dual_camera_observation_features_include_both_camera_keys():
    robot = make_robot()
    robot.config.cameras = {
        "observation.images.cam_wrist": SimpleNamespace(height=480, width=640),
        "observation.images.cam_external": SimpleNamespace(height=480, width=640),
    }
    robot.cameras = {"observation.images.cam_wrist": object(), "observation.images.cam_external": object()}

    features = robot.observation_features

    assert features["observation.images.cam_wrist"] == (480, 640, 3)
    assert features["observation.images.cam_external"] == (480, 640, 3)
