from __future__ import annotations

import logging
import time
from functools import cached_property
from typing import Any

import trossen_arm
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config import TrossenCartesianFollowerConfig
from .constants import ACTION_TARGET_KEYS, OBS_CARTESIAN_KEYS
from .geometry import (
    CartesianTarget,
    clamp_target,
    driver_pose_to_target_pose,
    resolve_orientation_origin_xy,
    target_from_action,
    target_pose_to_driver_pose,
)

logger = logging.getLogger(__name__)


class TrossenCartesianFollower(Robot):
    config_class = TrossenCartesianFollowerConfig
    name = "trossen_cartesian_follower_robot"

    def __init__(self, config: TrossenCartesianFollowerConfig):
        super().__init__(config)
        self.config = config
        self.driver = trossen_arm.TrossenArmDriver()
        self.cameras = make_cameras_from_configs(config.cameras)
        self.min_time_to_move = config.min_time_to_move_multiplier / config.loop_rate
        self.model = get_attr(trossen_arm.Model, ["wxai_v0", "WXAI_V0"])
        self.end_effector = get_attr(
            trossen_arm.StandardEndEffector,
            ["wxai_v0_follower", "WXAI_V0_FOLLOWER"],
        )
        self.position_mode = get_attr(trossen_arm.Mode, ["position", "POSITION"])
        self.cartesian_interp = get_attr(
            trossen_arm.InterpolationSpace,
            ["cartesian", "CARTESIAN"],
        )
        self._orientation_origin_xy = (config.orientation_origin_x, config.orientation_origin_y)
        self._last_sent_action: dict[str, float] | None = None

    @property
    def _joint_ft(self) -> dict[str, type]:
        return {f"{joint_name}.pos": float for joint_name in self.config.joint_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {
            **self._joint_ft,
            **{key: float for key in OBS_CARTESIAN_KEYS},
            **self._cameras_ft,
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {key: float for key in ACTION_TARGET_KEYS}

    @property
    def is_connected(self) -> bool:
        return self.driver.get_is_configured() and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        self.driver.configure(
            model=self.model,
            end_effector=self.end_effector,
            serv_ip=self.config.ip_address,
            clear_error=True,
        )
        if not self.is_calibrated and calibrate:
            self.calibrate()
        for cam in self.cameras.values():
            cam.connect()
        self.configure()
        logger.info("%s connected.", self)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        self._set_position_mode()
        if self.config.mode_init_wait > 0.0:
            time.sleep(self.config.mode_init_wait)
        if not self.config.skip_mode_probe:
            self._verify_position_mode()
        if self.config.stage_on_connect:
            self.driver.set_all_positions(
                self.config.staged_positions,
                goal_time=2.0,
                blocking=True,
            )
        cartesian_pose = self.driver.get_cartesian_positions()
        self._orientation_origin_xy = resolve_orientation_origin_xy(
            self.config.orientation_origin,
            cartesian_pose,
            (self.config.orientation_origin_x, self.config.orientation_origin_y),
        )

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        robot_all_joint_outputs = self.driver.get_robot_output().joint.all
        obs_dict = {
            f"{joint_name}.pos": pos
            for joint_name, pos in zip(
                self.config.joint_names,
                robot_all_joint_outputs.positions,
                strict=True,
            )
        }
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug("%s read joint state: %.1fms", self, dt_ms)

        cartesian_pose = self.driver.get_cartesian_positions()
        target_pose = driver_pose_to_target_pose(cartesian_pose, self._orientation_origin_xy)
        gripper_position = float(self.driver.get_gripper_position())
        obs_dict.update(dict(zip(OBS_CARTESIAN_KEYS, (*target_pose, gripper_position), strict=True)))

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug("%s read %s: %.1fms", self, cam_key, dt_ms)

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        target = clamp_target(
            target_from_action(action),
            self.config.max_reach_radius,
            self.config.gripper_min,
            self.config.gripper_max,
        )
        driver_pose = target_pose_to_driver_pose(
            target.pose,
            self.config.max_reach_radius,
            self._orientation_origin_xy,
        )
        goal_time = (
            self.config.cartesian_goal_time
            if self.config.cartesian_goal_time is not None
            else self.min_time_to_move
        )

        try:
            self.driver.set_cartesian_positions(
                goal_positions=list(driver_pose),
                interpolation_space=self.cartesian_interp,
                goal_time=goal_time,
                blocking=False,
            )
            self.driver.set_gripper_position(
                goal_position=target.gripper,
                goal_time=goal_time,
                blocking=False,
            )
        except Exception as exc:
            logger.warning("%s failed to send Cartesian target; trying recovery: %s", self, exc)
            recovered_action = self._recover_after_driver_error()
            if recovered_action is not None:
                self._last_sent_action = recovered_action
                return dict(recovered_action)
            if self._last_sent_action is not None:
                logger.warning("%s recovery failed; holding last sent action", self)
                return dict(self._last_sent_action)
            logger.exception("%s failed to send initial Cartesian target", self)
            raise

        self._last_sent_action = target.as_action_dict()
        return dict(self._last_sent_action)

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.sleep_on_disconnect:
            self.driver.set_all_positions(
                self.config.staged_positions,
                goal_time=2.0,
                blocking=True,
            )
            self.driver.set_all_positions(
                [0.0] * len(self.config.joint_names),
                goal_time=2.0,
                blocking=True,
            )
        self.driver.cleanup()
        for cam in self.cameras.values():
            cam.disconnect()
        logger.info("%s disconnected.", self)


    def _set_position_mode(self) -> None:
        if hasattr(self.driver, "set_all_modes"):
            self.driver.set_all_modes(self.position_mode)
            return
        self.driver.set_arm_modes(self.position_mode)
        self.driver.set_gripper_mode(self.position_mode)

    def _verify_position_mode(self) -> None:
        arm_positions = list(self.driver.get_arm_positions())
        gripper_position = float(self.driver.get_gripper_position())
        if hasattr(self.driver, "set_all_positions"):
            self.driver.set_all_positions(
                goal_positions=[*arm_positions, gripper_position],
                goal_time=self.config.mode_probe_goal_time,
                blocking=True,
            )
            return
        self.driver.set_arm_positions(
            goal_positions=arm_positions,
            goal_time=self.config.mode_probe_goal_time,
            blocking=True,
        )
        self.driver.set_gripper_position(
            goal_position=gripper_position,
            goal_time=self.config.mode_probe_goal_time,
            blocking=True,
        )

    def _recover_after_driver_error(self) -> dict[str, float] | None:
        if not try_clear_driver_error(self.driver):
            return None
        try:
            self._set_position_mode()
            if self.config.mode_init_wait > 0.0:
                time.sleep(self.config.mode_init_wait)
            if not self.config.skip_mode_probe:
                self._verify_position_mode()
            return self._read_current_target_action()
        except Exception:
            logger.exception("%s could not recover position mode after driver error", self)
            return None

    def _read_current_target_action(self) -> dict[str, float]:
        cartesian_pose = self.driver.get_cartesian_positions()
        target_pose = driver_pose_to_target_pose(cartesian_pose, self._orientation_origin_xy)
        gripper_position = float(self.driver.get_gripper_position())
        return CartesianTarget(target_pose, gripper_position).as_action_dict()

    @property
    def last_sent_target(self) -> CartesianTarget | None:
        if self._last_sent_action is None:
            return None
        return target_from_action(self._last_sent_action)


def get_attr(obj, names: list[str]):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"Cannot find any of {names} in {obj}")


def try_clear_driver_error(driver) -> bool:
    if not hasattr(driver, "clear_error"):
        return False
    try:
        driver.clear_error()
    except Exception:
        logger.exception("Could not clear Trossen driver error")
        return False
    return True
