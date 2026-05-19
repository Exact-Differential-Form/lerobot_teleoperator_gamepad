from __future__ import annotations

from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from .compat import RobotAction

from .config import TrossenGamepadCartesianTeleopConfig
from .constants import DEFAULT_SENSITIVITY, DELTA_ACTION_KEYS
from .gamepad import EvdevXboxController, MotionCommand


class TrossenGamepadCartesianTeleop(Teleoperator):
    config_class = TrossenGamepadCartesianTeleopConfig
    name = "trossen_gamepad_cartesian_teleop"

    def __init__(self, config: TrossenGamepadCartesianTeleopConfig):
        super().__init__(config)
        self.config = config
        self.controller: EvdevXboxController | None = None
        self._last_command = MotionCommand(
            source="xbox",
            action="idle",
            sensitivity=config.sensitivity if config.sensitivity is not None else DEFAULT_SENSITIVITY,
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "delta_x": float,
            "delta_y": float,
            "delta_z": float,
            "delta_roll": float,
            "delta_pitch": float,
            "delta_yaw": float,
            "delta_gripper": float,
            "reset": bool,
            "quit": bool,
            "sensitivity": float,
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.controller is not None

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            return
        self.controller = EvdevXboxController(
            device_path=self.config.device,
            linear_step=self.config.linear_step,
            angular_step=self.config.angular_step,
            gripper_step=self.config.gripper_step,
            deadzone_fraction=self.config.deadzone_fraction,
            grab_device=self.config.grab_device,
            config_path=self.config.config_path,
            sensitivity=self.config.sensitivity,
            sensitivity_factor=self.config.sensitivity_factor,
            min_sensitivity=self.config.min_sensitivity,
            max_sensitivity=self.config.max_sensitivity,
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> RobotAction:
        if self.controller is None:
            raise RuntimeError(f"{self} is not connected.")

        command = self.controller.read_command(timeout=0.0)
        if command is None:
            command = MotionCommand(
                source="xbox",
                action="idle",
                sensitivity=self.controller.sensitivity,
            )
        self._last_command = command
        return motion_command_to_action(command)

    def get_teleop_events(self) -> dict[str, Any]:
        return {
            TeleopEvents.IS_INTERVENTION: self._last_command.action != "idle",
            TeleopEvents.TERMINATE_EPISODE: self._last_command.quit,
            TeleopEvents.SUCCESS: False,
            TeleopEvents.RERECORD_EPISODE: False,
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        return None

    def disconnect(self) -> None:
        if self.controller is not None:
            self.controller.close()
            self.controller = None


def motion_command_to_action(command: MotionCommand) -> RobotAction:
    return {
        **dict(zip(DELTA_ACTION_KEYS[:6], command.pose_delta, strict=True)),
        "delta_gripper": float(command.gripper_delta),
        "reset": bool(command.reset),
        "quit": bool(command.quit),
        "sensitivity": float(command.sensitivity),
    }
