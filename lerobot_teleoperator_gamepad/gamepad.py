from __future__ import annotations

import select
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .constants import (
    DEFAULT_ANGULAR_STEP,
    DEFAULT_GRIPPER_STEP,
    DEFAULT_LINEAR_STEP,
    DEFAULT_MAX_SENSITIVITY,
    DEFAULT_MIN_SENSITIVITY,
    DEFAULT_SENSITIVITY,
    DEFAULT_SENSITIVITY_FACTOR,
)
from .controller_config import load_keybind_config, sensitivity_config, validate_sensitivity_bounds
from .geometry import Pose, ZERO_POSE, clamp

try:
    from evdev import InputDevice, categorize, ecodes, list_devices
except ImportError:  # pragma: no cover - exercised in hardware env
    InputDevice = None
    categorize = None
    ecodes = None
    list_devices = None


ACTION_DIRECTIONS: dict[str, Pose] = {
    "forward": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "backward": (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "leftward": (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    "rightward": (0.0, -1.0, 0.0, 0.0, 0.0, 0.0),
    "upward": (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    "downward": (0.0, 0.0, -1.0, 0.0, 0.0, 0.0),
    "pitchup": (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    "pitchdown": (0.0, 0.0, 0.0, 0.0, -1.0, 0.0),
    "yawleft": (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    "yawright": (0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
    "rollleft": (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    "rollright": (0.0, 0.0, 0.0, -1.0, 0.0, 0.0),
}

GRIPPER_DIRECTIONS = {
    "gripperopen": 1.0,
    "gripperclose": -1.0,
}

SENSITIVITY_ACTIONS = {"sensitivityup", "sensitivitydown"}

DEFAULT_XBOX_BUTTONS = {
    "gripperclose": "ABS_HAT0X:-",
    "gripperopen": "ABS_HAT0X:+",
    "reset": "BTN_SELECT",
    "rollleft": "BTN_TR",
    "rollright": "BTN_TL",
    "sensitivityup": "ABS_HAT0Y:-",
    "sensitivitydown": "ABS_HAT0Y:+",
    "quit": "BTN_START",
}

DEFAULT_XBOX_AXES = {
    "leftward": "ABS_X:-",
    "rightward": "ABS_X:+",
    "forward": "ABS_Y:-",
    "backward": "ABS_Y:+",
    "yawleft": "ABS_RX:-",
    "yawright": "ABS_RX:+",
    "pitchup": "ABS_RY:+",
    "pitchdown": "ABS_RY:-",
    "downward": "ABS_Z:+",
    "upward": "ABS_RZ:+",
}


@dataclass(frozen=True)
class MotionCommand:
    source: str
    action: str
    pose_delta: Pose = ZERO_POSE
    gripper_delta: float = 0.0
    reset: bool = False
    quit: bool = False
    sensitivity: float = 1.0


def find_xbox_controller(device_path: Optional[str] = None):
    if InputDevice is None:
        raise SystemExit("Missing Python package 'evdev'. Install it in the LeRobot environment.")
    if device_path:
        return _open_input_device_with_retry(device_path)

    paths = list(list_devices())
    paths.extend(str(path) for path in Path("/dev/input/by-id").glob("*event-joystick"))
    paths.extend(str(path) for path in Path("/dev/input").glob("event*"))

    seen: set[str] = set()
    devices = []
    for path in paths:
        real_path = str(Path(path).resolve())
        if real_path in seen:
            continue
        seen.add(real_path)
        try:
            devices.append(InputDevice(path))
        except PermissionError as exc:
            raise SystemExit(
                f"Permission denied reading {path}. Try adding your user to the input group, "
                "then restart WSL:\n  sudo usermod -aG input $USER"
            ) from exc

    if not devices:
        raise SystemExit("No /dev/input event devices found. Is the controller attached?")

    for device in devices:
        name = device.name.lower()
        if "xbox" in name or "controller" in name or "x-input" in name or "xinput" in name:
            return device

    print("Input devices found:")
    for device in devices:
        print(f"  {device.path}: {device.name}")
    raise SystemExit("Could not find an Xbox controller input device.")


def _open_input_device_with_retry(device_path: str, timeout_s: float = 5.0):
    deadline = time.monotonic() + timeout_s
    last_exc: OSError | None = None
    while time.monotonic() < deadline:
        try:
            return InputDevice(device_path)
        except PermissionError as exc:
            raise SystemExit(
                f"Permission denied reading {device_path}. Try adding your user to the input group, "
                "then restart WSL:\n  sudo usermod -aG input $USER"
            ) from exc
        except FileNotFoundError as exc:
            last_exc = exc
        except OSError as exc:
            if exc.errno not in {19}:  # ENODEV: controller was briefly re-enumerated.
                raise
            last_exc = exc
        time.sleep(0.1)
    raise SystemExit(f"Could not open controller device {device_path}: {last_exc}") from last_exc


class EvdevXboxController:
    source_name = "xbox"

    def __init__(
        self,
        device_path: Optional[str] = None,
        linear_step: float = DEFAULT_LINEAR_STEP,
        angular_step: float = DEFAULT_ANGULAR_STEP,
        gripper_step: float = DEFAULT_GRIPPER_STEP,
        deadzone_fraction: float = 0.125,
        grab_device: bool = True,
        config_path: str | None = None,
        sensitivity: float | None = None,
        sensitivity_factor: float = DEFAULT_SENSITIVITY_FACTOR,
        min_sensitivity: float | None = None,
        max_sensitivity: float | None = None,
        button_bindings: dict[str, str] | None = None,
        axis_bindings: dict[str, str] | None = None,
    ):
        file_config = load_keybind_config(config_path) if config_path else {}
        sensitivity_settings = sensitivity_config(file_config)
        self.button_bindings = dict(button_bindings or file_config.get("xbox_buttons") or DEFAULT_XBOX_BUTTONS)
        self.axis_binding_config = dict(axis_bindings or file_config.get("xbox_axes") or DEFAULT_XBOX_AXES)
        self.button_to_action = {
            button: action
            for action, button in self.button_bindings.items()
            if ":" not in button
        }
        self.axis_button_bindings = parse_axis_bindings(
            {
                action: binding
                for action, binding in self.button_bindings.items()
                if ":" in binding
            }
        )
        self.axis_bindings = parse_axis_bindings(self.axis_binding_config)
        self.linear_rate = float(linear_step)
        self.angular_rate = float(angular_step)
        self.gripper_rate = float(gripper_step)
        self.deadzone_fraction = float(deadzone_fraction)
        self.sensitivity_factor = float(sensitivity_factor)
        self.min_sensitivity = float(
            min_sensitivity
            if min_sensitivity is not None
            else sensitivity_settings["min_sensitivity"]
        )
        self.max_sensitivity = float(
            max_sensitivity
            if max_sensitivity is not None
            else sensitivity_settings["max_sensitivity"]
        )
        validate_sensitivity_bounds(self.min_sensitivity, self.max_sensitivity)
        self.sensitivity = clamp(
            sensitivity
            if sensitivity is not None
            else sensitivity_settings["default_sensitivity"],
            self.min_sensitivity,
            self.max_sensitivity,
        )
        self.active_buttons: set[str] = set()
        self.active_axis_buttons: dict[str, float] = {}
        self.axis_amounts: dict[str, float] = {}
        self.last_tick = time.monotonic()
        self.device = find_xbox_controller(device_path)
        self._grabbed = False
        if grab_device:
            try:
                self.device.grab()
                self._grabbed = True
            except OSError:
                pass

    def close(self):
        if self._grabbed:
            try:
                self.device.ungrab()
            except OSError:
                pass
            self._grabbed = False

    def __enter__(self):
        self.last_tick = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def read_command(self, timeout: Optional[float] = None) -> MotionCommand | None:
        readable, _, _ = select.select([self.device.fd], [], [], timeout)
        now = time.monotonic()
        had_active_controls = bool(self.active_buttons or self.axis_amounts or self.active_axis_buttons)

        if readable:
            for event in self.device.read():
                command = self._process_event(event)
                if command is not None:
                    self.last_tick = now
                    return command

        if not had_active_controls and (self.active_buttons or self.axis_amounts or self.active_axis_buttons):
            self.last_tick = now
            return None
        dt = now - self.last_tick
        self.last_tick = now
        return self._continuous_command(dt)

    def _process_event(self, event) -> MotionCommand | None:
        if event.type == ecodes.EV_KEY:
            key_event = categorize(event)
            keycode = key_event.keycode[0] if isinstance(key_event.keycode, list) else key_event.keycode
            action = self.button_to_action.get(keycode)
            if action is None:
                return None

            if action == "quit" and key_event.keystate:
                self._clear_active_controls()
                return MotionCommand(source=self.source_name, action=action, quit=True)
            if action == "reset" and key_event.keystate:
                self._clear_active_controls()
                return MotionCommand(source=self.source_name, action=action, reset=True)
            if action in SENSITIVITY_ACTIONS and key_event.keystate:
                self._change_sensitivity(action)
                return MotionCommand(source=self.source_name, action=action, sensitivity=self.sensitivity)

            if key_event.keystate:
                self.active_buttons.add(action)
            else:
                self.active_buttons.discard(action)
            return None

        if event.type == ecodes.EV_ABS:
            command = self._update_axis_buttons(event.code, event.value)
            if command is not None:
                return command
            self._update_axis_amounts(event.code, event.value)
        return None

    def _update_axis_buttons(self, code: int, value: int) -> MotionCommand | None:
        code_name = ecodes.ABS.get(code, str(code))
        command = None

        for action, axis_name, sign in self.axis_button_bindings:
            if axis_name != code_name:
                continue
            amount = self._axis_amount(code_name, code, value, sign)
            was_active = action in self.active_axis_buttons

            if amount <= 0.0:
                self.active_axis_buttons.pop(action, None)
                continue

            self.active_axis_buttons[action] = amount
            if was_active:
                continue

            if action == "quit":
                self._clear_active_controls()
                return MotionCommand(source=self.source_name, action=action, quit=True)
            if action == "reset":
                self._clear_active_controls()
                return MotionCommand(source=self.source_name, action=action, reset=True)
            if action in SENSITIVITY_ACTIONS:
                self._change_sensitivity(action)
                command = MotionCommand(
                    source=self.source_name,
                    action=action,
                    sensitivity=self.sensitivity,
                )

        return command

    def _update_axis_amounts(self, code: int, value: int):
        code_name = ecodes.ABS.get(code, str(code))
        for action, axis_name, sign in self.axis_bindings:
            if axis_name != code_name:
                continue
            amount = self._axis_amount(code_name, code, value, sign)
            if amount > 0.0:
                self.axis_amounts[action] = amount
            else:
                self.axis_amounts.pop(action, None)

    def _axis_amount(self, code_name: str, code: int, value: int, sign: int) -> float:
        if code_name.startswith("ABS_HAT"):
            if sign < 0 and value < 0:
                return 1.0
            if sign > 0 and value > 0:
                return 1.0
            return 0.0

        absinfo = self.device.absinfo(code)
        span = absinfo.max - absinfo.min
        deadzone = max(1, int(span * self.deadzone_fraction))

        if code_name in {"ABS_Z", "ABS_RZ"}:
            raw = max(0, value - absinfo.min)
            if raw <= deadzone:
                return 0.0
            return min(1.0, (raw - deadzone) / max(1, span - deadzone))

        center = (absinfo.max + absinfo.min) / 2.0
        distance = (value - center) * sign
        if distance <= deadzone:
            return 0.0
        max_distance = max(abs(absinfo.max - center), abs(absinfo.min - center))
        return min(1.0, (distance - deadzone) / max(1.0, max_distance - deadzone))

    def _continuous_command(self, dt: float) -> MotionCommand | None:
        action_amounts = dict(self.axis_amounts)
        for action in self.active_buttons:
            action_amounts[action] = max(action_amounts.get(action, 0.0), 1.0)
        for action, amount in self.active_axis_buttons.items():
            if action not in SENSITIVITY_ACTIONS:
                action_amounts[action] = max(action_amounts.get(action, 0.0), amount)

        if dt <= 0.0 or not action_amounts:
            return None

        pose_delta = [0.0] * 6
        gripper_delta = 0.0
        scale = self.sensitivity * dt

        for action, amount in action_amounts.items():
            if action in GRIPPER_DIRECTIONS:
                gripper_delta += self.gripper_rate * GRIPPER_DIRECTIONS[action] * amount * scale
                continue
            direction = ACTION_DIRECTIONS.get(action)
            if direction is None:
                continue
            for index, value in enumerate(direction):
                rate = self.linear_rate if index < 3 else self.angular_rate
                pose_delta[index] += rate * value * amount * scale

        return MotionCommand(
            source=self.source_name,
            action="+".join(sorted(action_amounts)),
            pose_delta=tuple(pose_delta),  # type: ignore[arg-type]
            gripper_delta=gripper_delta,
            sensitivity=self.sensitivity,
        )

    def _change_sensitivity(self, action: str):
        if action == "sensitivityup":
            self.sensitivity *= self.sensitivity_factor
        elif action == "sensitivitydown":
            self.sensitivity /= self.sensitivity_factor
        self.sensitivity = clamp(self.sensitivity, self.min_sensitivity, self.max_sensitivity)

    def _clear_active_controls(self):
        self.active_buttons.clear()
        self.active_axis_buttons.clear()
        self.axis_amounts.clear()


def parse_axis_bindings(bindings: dict[str, str]) -> list[tuple[str, str, int]]:
    parsed = []
    for action_name, spec in bindings.items():
        axis_name, separator, direction = spec.partition(":")
        if not separator or direction not in {"+", "-"}:
            raise ValueError(f"Invalid Xbox axis binding: {action_name}: {spec}")
        parsed.append((action_name, axis_name, 1 if direction == "+" else -1))
    return parsed
