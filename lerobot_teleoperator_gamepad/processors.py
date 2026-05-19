from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.processor import RobotActionProcessorStep

from .compat import PipelineFeatureType, RobotAction, TransitionKey

from .config import TrossenCartesianFollowerConfig
from .constants import (
    ACTION_TARGET_KEYS,
    DEFAULT_GRIPPER_MAX,
    DEFAULT_GRIPPER_MIN,
    DEFAULT_MAX_REACH_RADIUS,
    DELTA_ACTION_KEYS,
)
from .geometry import (
    CartesianTarget,
    clamp_target,
    normalize_angle,
    target_from_action,
    target_from_observation,
    xy_yaw,
)


@dataclass
class TrossenGamepadActionProcessorStep(RobotActionProcessorStep):
    max_reach_radius: float = DEFAULT_MAX_REACH_RADIUS
    gripper_min: float = DEFAULT_GRIPPER_MIN
    gripper_max: float = DEFAULT_GRIPPER_MAX
    reset_to_initial_target: bool = True
    controller_yaw_follow_xy: bool = False
    _target: CartesianTarget | None = field(default=None, init=False, repr=False)
    _initial_target: CartesianTarget | None = field(default=None, init=False, repr=False)
    _yaw_offset: float = field(default=0.0, init=False, repr=False)

    @classmethod
    def from_robot_config(cls, config) -> "TrossenGamepadActionProcessorStep":
        if isinstance(config, TrossenCartesianFollowerConfig):
            return cls(
                max_reach_radius=config.max_reach_radius,
                gripper_min=config.gripper_min,
                gripper_max=config.gripper_max,
                controller_yaw_follow_xy=config.controller_yaw_follow_xy,
            )
        return cls()

    def action(self, action: RobotAction) -> RobotAction:
        if all(key in action for key in ACTION_TARGET_KEYS) and not any(key in action for key in DELTA_ACTION_KEYS):
            target = clamp_target(
                target_from_action(action),
                self.max_reach_radius,
                self.gripper_min,
                self.gripper_max,
            )
            self._target = target
            self._sync_yaw_offset(self._target)
            self._target = self._with_followed_yaw(self._target)
            return self._target.as_action_dict()

        observation = self.transition.get(TransitionKey.OBSERVATION) or {}
        if self._target is None:
            self._target = target_from_observation(observation)
            self._sync_yaw_offset(self._target)
            if self.controller_yaw_follow_xy:
                self._target = self._with_followed_yaw(self._target)
            self._initial_target = self._target

        if action.get("reset", False):
            self._target = (
                self._initial_target
                if self.reset_to_initial_target
                else target_from_observation(observation)
            )
            self._sync_yaw_offset(self._target)
            return clamp_target(
                self._target,
                self.max_reach_radius,
                self.gripper_min,
                self.gripper_max,
            ).as_action_dict()

        pose_delta = list(float(action.get(key, 0.0)) for key in DELTA_ACTION_KEYS[:6])
        gripper_delta = float(action.get("delta_gripper", 0.0))
        if self.controller_yaw_follow_xy:
            self._yaw_offset = normalize_angle(self._yaw_offset + pose_delta[5])
            pose_delta[5] = 0.0
        next_target = CartesianTarget(
            pose=tuple(
                value + delta for value, delta in zip(self._target.pose, pose_delta, strict=True)
            ),
            gripper=self._target.gripper + gripper_delta,
        )
        next_target = self._with_followed_yaw(next_target)
        self._target = clamp_target(
            next_target,
            self.max_reach_radius,
            self.gripper_min,
            self.gripper_max,
        )
        return self._target.as_action_dict()

    def reset(self) -> None:
        self._target = None
        self._initial_target = None
        self._yaw_offset = 0.0

    def _sync_yaw_offset(self, target: CartesianTarget | None) -> None:
        if not self.controller_yaw_follow_xy or target is None:
            return
        x, y = target.pose[:2]
        self._yaw_offset = normalize_angle(target.pose[5] - xy_yaw(x, y))

    def _with_followed_yaw(self, target: CartesianTarget) -> CartesianTarget:
        if not self.controller_yaw_follow_xy:
            return target
        pose = list(target.pose)
        pose[5] = normalize_angle(xy_yaw(pose[0], pose[1]) + self._yaw_offset)
        return CartesianTarget(pose=tuple(pose), gripper=target.gripper)

    def transform_features(self, features):
        features[PipelineFeatureType.ACTION] = {key: float for key in ACTION_TARGET_KEYS}
        return features
